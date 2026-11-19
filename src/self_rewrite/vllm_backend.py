"""Optional vLLM serving backend with transactional checkpoint overlays.

Generation/editor replies use vLLM; hidden reads use a lazy CPU Transformers copy.
This avoids competing GPU allocations but makes reads slower than GPU reads.
Only local, synchronous, single-GPU execution is supported. Runtime validation
requires a separately installed compatible vLLM build; contract tests are not a
substitute for running that build with Qwen.
"""
from contextlib import contextmanager
from pathlib import Path
import re
import tempfile
import torch
from .checkpoint import Checkpoint
from .evaluation import grade,problem_seed
from .vision import grade_image


def apply_overlay(worker,path):
    """Trusted harness RPC; editor code cannot call this function."""
    from safetensors import safe_open
    with safe_open(path,framework='pt',device='cpu') as handle:
        names=list(handle.keys())
        if not names: return []
        loaded=worker.model_runner.model.load_weights((name,handle.get_tensor(name)) for name in names)
        if not loaded: raise RuntimeError('vLLM accepted no overlay tensors')
    if torch.cuda.is_available(): torch.cuda.synchronize()
    return sorted(loaded)


class VLLMBackend:
    def __init__(self,engine,tokenizer,checkpoint,edits,config,*,processor=None,sampling_factory=None):
        self.engine,self.tokenizer=engine,tokenizer
        self.processor=processor
        self.checkpoint,self.edits,self.config=checkpoint,edits,config
        self.clean=Checkpoint(checkpoint.root)
        self._reference=None
        self._active=False
        self._broken=False
        if sampling_factory is None:
            from vllm import SamplingParams
            sampling_factory=SamplingParams
        self.sampling_factory=sampling_factory

    @classmethod
    def from_checkpoint(cls,checkpoint,edits,config,*,gpu_memory_utilization=.9,max_model_len=None,device='cuda'):
        if device!='cuda': raise ValueError('vLLM adapter currently supports a local CUDA engine only')
        from vllm import LLM
        from transformers import AutoConfig,AutoProcessor,AutoTokenizer
        architecture=AutoConfig.from_pretrained(checkpoint.root,trust_remote_code=False)
        processor=AutoProcessor.from_pretrained(checkpoint.root,trust_remote_code=False) if hasattr(architecture,'vision_config') else None
        tokenizer=processor.tokenizer if processor else AutoTokenizer.from_pretrained(checkpoint.root,trust_remote_code=False)
        # Permit the advertised editor reset threshold plus one full reply.
        max_model_len=max_model_len or config.context_reset_tokens+config.editor_tokens+4096
        engine=LLM(model=str(checkpoint.root),tokenizer=str(checkpoint.root),dtype='bfloat16',
                   quantization=None,tensor_parallel_size=1,trust_remote_code=False,
                   enable_prefix_caching=False,enforce_eager=True,
                   max_model_len=max_model_len,gpu_memory_utilization=gpu_memory_utilization)
        return cls(engine,tokenizer,checkpoint,edits,config,processor=processor)

    def _update(self,tensors,directory,label):
        from safetensors.torch import save_file
        if not tensors: return
        path=Path(directory)/(label+'.safetensors')
        save_file({name:value.detach().cpu().contiguous() for name,value in tensors.items()},path)
        self.engine.collective_rpc(apply_overlay,args=(str(path),))
        if self.engine.reset_prefix_cache() is False:
            raise RuntimeError('vLLM cache reset failed after weight update')

    @contextmanager
    def weights(self,mode):
        if mode not in {'original','edited','editor'}: raise ValueError('Invalid weights')
        if self._active or self._broken: raise RuntimeError('Engine busy or restoration previously failed')
        self._active=True
        names=set() if mode=='editor' else set(self.checkpoint.damage_names)
        edited=set(self.edits.list_edits()) if mode=='edited' else set()
        names|=edited
        if any('mtp' in n.split('.') for n in names):
            self._active=False
            raise ValueError('Auxiliary MTP tensors are not served by this engine')
        try:
            with tempfile.TemporaryDirectory(prefix='rewrite-overlay-') as directory:
                # Restore every possibly changed tensor even if loading fails halfway.
                clean={name:self.clean.load(name) for name in names}
                try:
                    current={name:self.edits.load(name) if name in edited else self.checkpoint.load(name) for name in names}
                    self._update(current,directory,'requested')
                    yield
                finally:
                    try: self._update(clean,directory,'clean')
                    except Exception:
                        self._broken=True
                        raise
        finally: self._active=False

    def _render(self,messages):
        renderer=self.processor or self.tokenizer
        return renderer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=True)

    def _params(self,limit,seed):
        return self.sampling_factory(n=1,temperature=self.config.temperature,top_p=self.config.top_p,
            top_k=self.config.top_k,max_tokens=limit,seed=seed,skip_special_tokens=False)

    def generate(self,problems,*,weights='original',samples=1,max_tokens=None,seed_base=0,comparator=None):
        limit=self.config.max_tokens if max_tokens is None else max_tokens
        if weights not in {'original','edited'} or not problems or samples<1 or not 16<=limit<=self.config.max_tokens:
            raise ValueError('Invalid generation request')
        from .data import EXCLUSION_ANSWER
        if any(p.answer==EXCLUSION_ANSWER for p in problems):
            raise ValueError('Exclusion-only data cannot be used for generation/evaluation')
        prompts=[];params=[];identities=[]
        for problem in problems:
            question=problem.question+'\nPlease reason step by step, and put your final answer within \\boxed{}.'
            image=None
            if problem.image:
                from PIL import Image
                if self.processor is None: raise ValueError('Multimodal processor required')
                image=Image.open(problem.image).convert('RGB')
                if problem.grading=='point':
                    import json
                    scale=json.loads(problem.answer)['coordinate_scale']
                    question+=f' Return one click as (x, y), with x in [0, {scale[0]}] and y in [0, {scale[1]}].'
                content=[{'type':'image'},{'type':'text','text':question}]
            else: content=question
            prompt={'prompt':self._render([{'role':'user','content':content}])}
            if image is not None: prompt['multi_modal_data']={'image':image}
            for sample in range(samples):
                seed=(problem_seed(problem.id,seed_base)+sample)%2**31
                prompts.append(prompt);params.append(self._params(limit,seed));identities.append((problem,sample,seed))
        with self.weights(weights): outputs=self.engine.generate(prompts,params,use_tqdm=False)
        if len(outputs)!=len(identities): raise RuntimeError('vLLM response count mismatch')
        rows=[]
        for result,(problem,sample,seed) in zip(outputs,identities):
            if len(result.outputs)!=1: raise RuntimeError('Expected one completion per request')
            answer=result.outputs[0]
            if problem.grading=='math':
                extra={} if comparator is None else {'comparator':comparator}
                rows.append(grade(problem.id,sample,seed,answer.text,len(answer.token_ids),limit,
                                  problem.answer,finish_reason=answer.finish_reason,**extra))
            else: rows.append(grade_image(problem,sample,seed,answer.text,len(answer.token_ids),limit,answer.finish_reason))
        return rows

    def reply(self,messages,turn):
        prompt=self._render(messages)
        with self.weights('editor'):
            results=self.engine.generate([{'prompt':prompt}],self._params(self.config.editor_tokens,self.config.turn_seed+turn),use_tqdm=False)
        text=results[0].outputs[0].text
        if prompt.rfind('<think>')>prompt.rfind('</think>'): text='<think>\n'+text
        return re.sub(r'<\|[^>]+\|>','',text)

    def count_tokens(self,messages):
        return len(self.tokenizer.encode(self._render(messages),add_special_tokens=False))

    def read(self,texts,**kwargs):
        if self._active or self._broken: raise RuntimeError('Engine unavailable for reads')
        if self._reference is None:
            from .backend import TransformersBackend
            self._reference=TransformersBackend.from_checkpoint(self.checkpoint,self.edits,self.config,device='cpu')
        self._reference.edits=self.edits
        return self._reference.read(texts,**kwargs)
