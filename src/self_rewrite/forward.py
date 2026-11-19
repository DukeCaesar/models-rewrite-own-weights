"""Validated experiment requests and auditable forward-pass artifacts."""
from __future__ import annotations
import json
from pathlib import Path
import torch
from .evaluation import paired_report


class ForwardRunner:
    def __init__(self, backend, practice, work):
        self.backend = backend
        self.config = backend.config
        self.practice = {p.id:p for p in practice}
        if len(self.practice) != len(practice):
            raise ValueError('Practice IDs must be unique')
        self.work = Path(work)
        self.work.mkdir(parents=True,exist_ok=True)
        self.used = 0

    def validate(self, request):
        if not isinstance(request,dict):
            raise ValueError('Forward request must be a JSON object')
        kind = request.get('kind')
        weights = request.get('weights','original')
        if kind == 'generate':
            allowed = {'kind','weights','ids','samples','max_tokens','candidates'}
            ids = request.get('ids',[])
            samples = request.get('samples',1)
            limit = request.get('max_tokens',self.config.max_tokens)
            if not isinstance(ids,list) or not ids or any(not isinstance(i,str) for i in ids):
                raise ValueError('ids must be a nonempty list of practice identifiers')
            if len(set(ids)) != len(ids) or any(i not in self.practice for i in ids):
                raise ValueError('IDs must be unique and belong to the visible practice set')
            if type(samples) is not int or samples < 1:
                raise ValueError('samples must be a positive integer')
            if type(limit) is not int or not 16 <= limit <= self.config.max_tokens:
                raise ValueError('Invalid generation token budget')
            if weights not in {'original','edited','both'}:
                raise ValueError('Invalid generation weights')
            candidates=request.get('candidates')
            variants=2 if weights=='both' else 1
            if candidates is not None:
                if self.config.protocol!=2 or weights!='both':
                    raise ValueError('Candidate comparisons require protocol 2 and weights=both')
                if not isinstance(candidates,list) or not 1<=len(candidates)<=4 or any(not isinstance(c,str) for c in candidates) or len(set(candidates))!=len(candidates):
                    raise ValueError('Choose one to four distinct saved candidates')
                for label in candidates: self.backend.edits.candidate(label)
                variants=1+len(candidates)
            if len(ids)*samples*variants > self.config.max_generation_answers:
                raise ValueError('Request exceeds total answer allowance')
        elif kind == 'read':
            allowed = {'kind','weights','texts','layers','read_kind','positions','stride','last','answer_starts'}
            texts = request.get('texts',[])
            if not isinstance(texts,list) or not 1 <= len(texts) <= self.config.max_read_texts or any(not isinstance(t,str) or not t for t in texts):
                raise ValueError('Invalid read texts')
            if weights not in {'original','edited'}:
                raise ValueError('Read requires original or edited weights')
            layers = request.get('layers',[])
            if not isinstance(layers,list) or any(type(i) is not int or i < 0 for i in layers):
                raise ValueError('layers must be nonnegative integer indices')
        else:
            raise ValueError('kind must be generate or read')
        if request.keys() - allowed:
            raise ValueError('Unknown request fields: '+str(sorted(request.keys()-allowed)))
        return kind

    def run(self,request):
        kind = self.validate(request)
        if self.used >= self.config.passes:
            raise ValueError('No forward passes remain')
        number = self.used + 1
        weights = request.get('weights','original')
        if kind == 'generate':
            problems = [self.practice[i] for i in request['ids']]
            modes = ['original','edited'] if weights == 'both' else [weights]
            candidates=request.get('candidates')
            options=dict(samples=request.get('samples',1),max_tokens=request.get('max_tokens'))
            if candidates is None:
                results={mode:self.backend.generate(problems,weights=mode,**options) for mode in modes}
            else:
                results={'original':self.backend.generate(problems,weights='original',**options)}
                current=self.backend.edits
                try:
                    for label in candidates:
                        self.backend.edits=current.candidate(label)
                        results['candidate:'+label]=self.backend.generate(problems,weights='edited',**options)
                finally:
                    self.backend.edits=current
            artifact = self.work/f'gen_{number}.jsonl'
            rows = [dict(weights=mode,**r.to_dict()) for mode,answers in results.items() for r in answers]
            artifact.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            if candidates is not None:
                observation={'candidates':{label:paired_report(results['original'],results['candidate:'+label]) for label in candidates}}
            elif weights == 'both':
                observation = paired_report(results['original'],results['edited'])
            else:
                answers = results[weights]
                observation = dict(accuracy=sum(r.correct for r in answers)/len(answers),
                    cut_share=sum(r.cut for r in answers)/len(answers),
                    mean_tokens=sum(r.tokens for r in answers)/len(answers))
        else:
            data = self.backend.read(request['texts'],layers=request.get('layers',[]),weights=weights,
                kind=request.get('read_kind','hidden'),positions=request.get('positions','mean'),
                stride=request.get('stride',1),last=request.get('last',1),answer_starts=request.get('answer_starts'))
            artifact = self.work/f'read_{number}.pt'
            torch.save(data,artifact)
            observation = {'shapes':[{k:list(v.shape) for k,v in item.items()} for item in data]}
        # Failed requests do not consume a pass; artifacts are only committed on success.
        self.used += 1
        observation.update(artifact=artifact.name,passes_used=self.used,passes_total=self.config.passes)
        with (self.work/'forward.jsonl').open('a') as handle:
            handle.write(json.dumps({'request':request,'observation':observation})+'\n')
        return observation
