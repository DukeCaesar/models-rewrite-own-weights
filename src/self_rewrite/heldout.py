"""Precommitted paired evaluation with per-problem atomic resume artifacts."""
from dataclasses import asdict
import json
from pathlib import Path
from .data import digest_json, file_digest, require_disjoint
from .evaluation import Answer, paired_report, permutation_test, problem_seed


def atomic_json(path,value):
    path=Path(path); temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2)); temp.replace(path)


def model_identity(backend):
    checkpoint=backend.checkpoint
    files=set(checkpoint.files.values())
    # Include inference configuration and tokenizer assets, not just weights.
    files|={p.name for p in checkpoint.root.iterdir() if p.is_file() and p.suffix in {'.json','.jinja','.txt','.model'}}
    return dict(checkpoint={name:file_digest(checkpoint.root/name) for name in sorted(files)},
        damage=file_digest(checkpoint.damage) if checkpoint.damage else None,
        edits={name:file_digest(backend.edits._path(name)) for name in backend.edits.list_edits()})


class HeldoutEvaluation:
    def __init__(self,backend,problems,practice,output,*,samples=8,seed_base=1000003,
                 max_tokens=16384,flips=100000,test_seed=0,alpha=.05,identity=None,baseline_cache=None,allow_practice_overlap=False):
        if not problems or len({p.id for p in problems})!=len(problems): raise ValueError('Invalid held-out problems')
        from .data import question_key
        if {p.id for p in practice}&{p.id for p in problems}: raise ValueError('Held-out IDs must not be practice IDs')
        practice_questions={question_key(p.question) for p in practice}
        overlaps=[p.id for p in problems if question_key(p.question) in practice_questions]
        if overlaps and not allow_practice_overlap: require_disjoint(practice,problems)
        if seed_base==0: raise ValueError('Evaluation must use a seed base not used by sessions')
        if samples<1 or flips<1 or not 0<alpha<1 or not 16<=max_tokens<=backend.config.max_tokens:
            raise ValueError('Invalid evaluation settings')
        self.backend,self.problems,self.root=backend,problems,Path(output)
        from .baseline import BaselineCache
        self.baseline_cache=BaselineCache(baseline_cache) if baseline_cache is not None else None
        self.root.mkdir(parents=True,exist_ok=True)
        self.samples,self.seed_base,self.max_tokens=samples,seed_base,max_tokens
        self.flips,self.test_seed,self.alpha=flips,test_seed,alpha
        self.plan=dict(schema=1,problems=[asdict(p) for p in problems],
            allow_practice_overlap=allow_practice_overlap,practice_overlap_ids=overlaps,
            image_hashes={p.id:file_digest(p.image) for p in problems if p.image},
            practice_digest=digest_json([asdict(p) for p in practice]),samples=samples,
            seed_base=seed_base,max_tokens=max_tokens,flips=flips,test_seed=test_seed,alpha=alpha,
            sampling={k:getattr(backend.config,k) for k in ('temperature','top_p','top_k')},
            backend=type(backend).__name__,model=model_identity(backend) if identity is None else identity,
            primary_test='problem_sign_flip' if samples>1 else 'paired_sign',
            shared_baseline=baseline_cache is not None,
            look_rule='One analysis after all paired answers are committed')
        path=self.root/'plan.json'
        if path.exists():
            if json.loads(path.read_text())!=self.plan: raise ValueError('Evaluation plan changed; use a new output directory')
        elif any(self.root.iterdir()): raise ValueError('Cannot adopt an existing directory without its plan')
        else: atomic_json(path,self.plan)

    def _validate(self,rows,problem):
        if set(rows)!={'original','edited'}: raise ValueError('Incomplete paired artifact')
        for mode in rows:
            expected={(problem.id,j,(problem_seed(problem.id,self.seed_base)+j)%2**31) for j in range(self.samples)}
            actual={(r['problem_id'],r['sample'],r['seed']) for r in rows[mode]}
            if actual!=expected or len(rows[mode])!=self.samples: raise ValueError('Invalid stored pair identities')

    def run(self):
        original=[]; edited=[]
        for index,problem in enumerate(self.problems):
            path=self.root/f'pair_{index:06d}.json'
            if path.exists(): rows=json.loads(path.read_text())
            else:
                rows={}
                for mode in ('original','edited'):
                    if mode=='original' and self.baseline_cache is not None:
                        rows[mode]=self.baseline_cache.get(self.backend,problem,model=self.plan['model'],
                            samples=self.samples,max_tokens=self.max_tokens,seed_base=self.seed_base)
                    else:
                        rows[mode]=[a.to_dict() for a in self.backend.generate([problem],weights=mode,
                            samples=self.samples,max_tokens=self.max_tokens,seed_base=self.seed_base)]
                self._validate(rows,problem)
                atomic_json(path,rows)
            self._validate(rows,problem)
            original.extend(Answer(**r) for r in rows['original'])
            edited.extend(Answer(**r) for r in rows['edited'])
        report=paired_report(original,edited)
        benchmarks=sorted({p.benchmark for p in self.problems if p.benchmark})
        if benchmarks:
            report['benchmarks']={}
            for name in benchmarks:
                ids={p.id for p in self.problems if p.benchmark==name}
                report['benchmarks'][name]=paired_report([r for r in original if r.problem_id in ids],
                                                       [r for r in edited if r.problem_id in ids])
        report['permutation_p']=permutation_test(original,edited,flips=self.flips,seed=self.test_seed)
        primary=report['permutation_p'] if self.samples>1 else report['sign_p']
        report.update(problems=len(self.problems),samples=self.samples,primary_p=primary,
            significantly_better=primary<self.alpha and report['correct']['delta']>0,
            plan_digest=digest_json(self.plan),complete=True)
        atomic_json(self.root/'report.json',report)
        return report
