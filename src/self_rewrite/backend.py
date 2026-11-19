"""Transformers execution adapter.

This serial adapter is an executable reference implementation. The paper used
vLLM; throughput and sampling streams therefore differ. No quantization is used.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
import torch
from .evaluation import grade, problem_seed


@dataclass(frozen=True)
class Problem:
    id: str
    question: str
    answer: str
    image: str | None = None
    grading: str = "math"
    benchmark: str | None = None


class TransformersBackend:
    def __init__(self, model, tokenizer, checkpoint, edits, config, processor=None):
        self.model = model.eval()
        self.model.requires_grad_(False)
        self.tokenizer = tokenizer
        self.processor = processor
        self.checkpoint = checkpoint
        self.edits = edits
        self.config = config
        self._active = False
        self._parameters = dict(model.named_parameters())
        self._parameters.update(dict(model.named_buffers()))

    @classmethod
    def from_checkpoint(cls, checkpoint, edits, config, *, device='cuda', dtype=torch.bfloat16):
        from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor
        path = str(checkpoint.root)
        architecture = AutoConfig.from_pretrained(path, trust_remote_code=False)
        multimodal = hasattr(architecture, 'vision_config')
        loader = AutoModelForImageTextToText if multimodal else AutoModelForCausalLM
        model = loader.from_pretrained(path, dtype=dtype, device_map=device, trust_remote_code=False)
        processor = AutoProcessor.from_pretrained(path, trust_remote_code=False) if multimodal else None
        tokenizer = processor.tokenizer if processor else AutoTokenizer.from_pretrained(path, trust_remote_code=False)
        return cls(model, tokenizer, checkpoint, edits, config, processor)

    @property
    def device(self):
        return self.model.get_input_embeddings().weight.device

    @contextmanager
    def weights(self, mode):
        """Always restore clean editor weights, including after failed inference."""
        if mode not in {'editor', 'original', 'edited'}:
            raise ValueError('weights must be editor, original, or edited')
        if self._active:
            raise RuntimeError('Nested/concurrent weight contexts are not supported')
        self._active = True
        backups = {}
        try:
            names = set() if mode == 'editor' else set(self.checkpoint.damage_names)
            edited_names = set(self.edits.list_edits()) if mode == 'edited' else set()
            names |= edited_names
            with torch.no_grad():
                for name in sorted(names):
                    if name not in self._parameters:
                        raise KeyError('Checkpoint tensor does not match loaded model: ' + name)
                    target = self._parameters[name]
                    value = self.edits.load(name) if name in edited_names else self.checkpoint.load(name)
                    if value.shape != target.shape or not torch.isfinite(value).all():
                        raise ValueError('Invalid inference replacement: ' + name)
                    backups[name] = target.detach().cpu().clone()
                    target.copy_(value.to(device=target.device, dtype=target.dtype))
            yield
        finally:
            with torch.no_grad():
                for name, value in backups.items():
                    self._parameters[name].copy_(value.to(self._parameters[name].device))
            self._active = False

    def _chat_inputs(self, messages):
        template = self.processor if self.processor else self.tokenizer
        return template.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                    enable_thinking=True, return_dict=True, return_tensors='pt').to(self.device)

    def _problem_inputs(self, problem):
        question = problem.question + '\nPlease reason step by step, and put your final answer within \\boxed{}.'
        if problem.grading=='point':
            import json
            scale=json.loads(problem.answer)['coordinate_scale']
            question+=f' Return one click as (x, y), with x in [0, {scale[0]}] and y in [0, {scale[1]}].' 
        if problem.image:
            if self.processor is None:
                raise ValueError('Image problem requires a multimodal processor')
            content = [{'type': 'image', 'image': problem.image}, {'type': 'text', 'text': question}]
        else:
            content = question
        return self._chat_inputs([{'role': 'user', 'content': content}])

    @torch.inference_mode()
    def _sample(self, inputs, max_tokens, seed):
        cuda_devices = sorted({p.device.index for p in self.model.parameters() if p.device.type == 'cuda'})
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            if cuda_devices:
                torch.cuda.manual_seed_all(seed)
            output = self.model.generate(**inputs, max_new_tokens=max_tokens,
                do_sample=True, temperature=self.config.temperature, top_k=self.config.top_k,
                top_p=self.config.top_p, pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id)
        tokens = output[0, inputs['input_ids'].shape[1]:].tolist()
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        reason = 'stop' if tokens and tokens[-1] in eos else 'length' if len(tokens) >= max_tokens else 'stop'
        return self.tokenizer.decode(tokens, skip_special_tokens=False), len(tokens), reason

    def generate(self, problems, *, weights='original', samples=1, max_tokens=None, seed_base=0, comparator=None):
        max_tokens = self.config.max_tokens if max_tokens is None else max_tokens
        if weights not in {'original','edited'}:
            raise ValueError('Requested generations cannot use clean editor weights')
        if not problems or samples < 1 or not 16 <= max_tokens <= self.config.max_tokens:
            raise ValueError('Invalid generation request')
        from .data import EXCLUSION_ANSWER
        if any(p.answer==EXCLUSION_ANSWER for p in problems):
            raise ValueError('Exclusion-only data cannot be used for generation/evaluation')
        rows = []
        with self.weights(weights):
            for problem in problems:
                inputs = self._problem_inputs(problem)
                for sample in range(samples):
                    seed = (problem_seed(problem.id, seed_base) + sample) % 2**31
                    text, count, reason = self._sample(inputs, max_tokens, seed)
                    extra = {} if comparator is None else {'comparator': comparator}
                    if problem.grading=='math':
                        rows.append(grade(problem.id, sample, seed, text, count, max_tokens,
                                          problem.answer, finish_reason=reason, **extra))
                    else:
                        from .vision import grade_image
                        rows.append(grade_image(problem,sample,seed,text,count,max_tokens,reason))
        return rows

    def reply(self, messages, turn):
        with self.weights('editor'):
            inputs = self._chat_inputs(messages)
            text=self._sample(inputs, self.config.editor_tokens, self.config.turn_seed + turn)[0]
            # Preserve an opening thinking tag supplied by the template so a truncated
            # reasoning passage cannot accidentally execute a draft tool block.
            prompt=self.tokenizer.decode(inputs['input_ids'][0],skip_special_tokens=False)
            if prompt.rfind('<think>')>prompt.rfind('</think>'):
                text='<think>\n'+text
            import re
            return re.sub(r'<\|[^>]+\|>','',text)

    def count_tokens(self, messages):
        return len(self.tokenizer.apply_chat_template(messages, tokenize=True, enable_thinking=True))

    @torch.inference_mode()
    def read(self, texts, *, layers, weights='original', kind='hidden', positions='mean', stride=1, last=1, answer_starts=None):
        """Read post-layer residuals, MLP intermediate activations, or token log probabilities.

        Texts are literal tokenized strings. answer_starts gives token offsets
        into each text, excluding prompt tokens from the returned measurements.
        Hidden layer 0 means AFTER transformer block 0, not the embedding.
        """
        if weights not in {'original','edited'} or not 1 <= len(texts) <= self.config.max_read_texts:
            raise ValueError('Invalid read weights or text count')
        if kind not in {'hidden','mlp','logprobs'} or positions not in {'mean','all','stride','last'}:
            raise ValueError('Invalid read kind or position reduction')
        if stride < 1 or last < 1:
            raise ValueError('stride/last must be positive')
        starts = [0] * len(texts) if answer_starts is None else answer_starts
        if len(starts) != len(texts):
            raise ValueError('One answer start per text is required')
        modules = dict(self.model.named_modules())
        targets = {}
        if kind != 'logprobs':
            for layer in layers:
                suffix = f'.layers.{int(layer)}' + ('.mlp.down_proj' if kind == 'mlp' else '')
                matches = [name for name in modules if ('.'+name).endswith(suffix) and 'vision' not in name]
                if len(matches) != 1:
                    raise ValueError(f'Cannot uniquely resolve language layer {layer}: {matches}')
                targets[str(layer)] = modules[matches[0]]
            if not targets:
                raise ValueError('At least one layer is required')
        results = []
        with self.weights(weights):
            for text, start in zip(texts, starts):
                inputs = self.tokenizer(text, return_tensors='pt', truncation=True,
                                        max_length=self.config.max_read_tokens).to(self.device)
                length = inputs['input_ids'].shape[1]
                if not 0 <= start < length:
                    raise ValueError('Answer start outside truncated text')
                capture = {}
                handles = []
                def capture_hook(key):
                    def hook(module, args, output):
                        value = output[0] if isinstance(output, tuple) else output
                        capture[key] = value[0].detach().float().cpu()
                    return hook
                try:
                    for key, module in targets.items():
                        if kind == 'mlp':
                            def pre_hook(module, args, key=key):
                                capture[key] = args[0][0].detach().float().cpu()
                            handles.append(module.register_forward_pre_hook(pre_hook))
                        else:
                            handles.append(module.register_forward_hook(capture_hook(key)))
                    outputs = self.model(**inputs, use_cache=False)
                    if kind == 'logprobs':
                        logits = outputs.logits[0, :-1].float()
                        ids = inputs['input_ids'][0, 1:]
                        capture['logprobs'] = (logits.gather(-1, ids[:,None]).squeeze(-1) - torch.logsumexp(logits, -1)).cpu()
                finally:
                    for handle in handles:
                        handle.remove()
                reduced = {}
                for key, value in capture.items():
                    # logprob index t-1 predicts token t (token zero has no probability).
                    value = value[max(0,start-1) if kind == 'logprobs' else start:]
                    if value.shape[0] == 0:
                        raise ValueError('Read has no measurable positions')
                    if positions == 'mean': value = value.mean(0)
                    elif positions == 'stride': value = value[::stride]
                    elif positions == 'last': value = value[-last:]
                    reduced[key] = value
                results.append(reduced)
        return results
