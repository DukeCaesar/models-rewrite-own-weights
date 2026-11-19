"""Shared baseline answers, keyed by model/data/sampling identity, with file locks."""
import fcntl
import json
from pathlib import Path
from dataclasses import asdict
from .data import digest_json


class BaselineCache:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)

    def get(self,backend,problem,*,model,samples,max_tokens,seed_base):
        from .heldout import atomic_json
        baseline={k:v for k,v in model.items() if k!='edits'}
        key=dict(model=baseline,backend=type(backend).__name__,problem=asdict(problem),
                 samples=samples,max_tokens=max_tokens,seed_base=seed_base,
                 sampling={k:getattr(backend.config,k) for k in ('temperature','top_p','top_k')})
        if problem.image:
            from .data import file_digest
            key['image_hash']=file_digest(problem.image)
        digest=digest_json(key);path=self.root/(digest+'.json')
        with (self.root/(digest+'.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if path.exists():
                record=json.loads(path.read_text())
                if record['key']!=key or record['answers_digest']!=digest_json(record['answers']):
                    raise ValueError('Shared baseline cache integrity mismatch')
                return record['answers']
            answers=[r.to_dict() for r in backend.generate([problem],weights='original',samples=samples,
                     max_tokens=max_tokens,seed_base=seed_base)]
            atomic_json(path,{'key':key,'answers':answers,'answers_digest':digest_json(answers)})
            return answers
