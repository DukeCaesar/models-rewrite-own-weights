"""Second-protocol held-out workflow (Appendix E.9).

Every rule, seed pool and look is written before any generation. The optional
second look is fixed at construction, never enabled after reading first results.
"""
from dataclasses import asdict,replace
import json
from pathlib import Path
from .data import require_disjoint,digest_json
from .evaluation import Answer,paired_report,permutation_test
from .heldout import HeldoutEvaluation,atomic_json,model_identity


def load_pairs(root):
    original=[];edited=[]
    plan=json.loads((Path(root)/'plan.json').read_text())
    for i in range(len(plan['problems'])):
        pair=json.loads((Path(root)/f'pair_{i:06d}.json').read_text())
        original.extend(Answer(**r) for r in pair['original'])
        edited.extend(Answer(**r) for r in pair['edited'])
    return original,edited


def pool_runs(runs,samples_per_run):
    a=[];b=[]
    for index,(base,edit) in enumerate(runs):
        a.extend(replace(r,sample=r.sample+index*samples_per_run) for r in base)
        b.extend(replace(r,sample=r.sample+index*samples_per_run) for r in edit)
    return a,b


class StagedEvaluation:
    def __init__(self,backend,initial,confirmatory,practice,output,*,two_looks=False,
                 seed_bases=(1000003,2000003,3000003,4000003,5000003),baseline_cache=None,
                 max_tokens=16384,flips=100000,allow_initial_overlap=False):
        if len(seed_bases)!=5 or len(set(seed_bases))!=5 or 0 in seed_bases:
            raise ValueError('Five distinct non-session seed bases are required')
        if not initial or not confirmatory: raise ValueError('Both evaluation subsets must be nonempty')
        if not allow_initial_overlap: require_disjoint(practice,initial)
        if {p.id for p in practice}&{p.id for p in initial}: raise ValueError('Held-out IDs cannot be practice IDs')
        require_disjoint(practice,confirmatory)
        self.allow_initial_overlap=allow_initial_overlap
        require_disjoint(initial,confirmatory)
        self.backend,self.initial,self.confirmatory,self.practice=backend,initial,confirmatory,practice
        self.root=Path(output);self.root.mkdir(parents=True,exist_ok=True)
        self.two_looks,self.seeds,self.max_tokens,self.flips=two_looks,seed_bases,max_tokens,flips
        self.baseline_cache=baseline_cache
        self.identity=model_identity(backend)
        self.plan=dict(model=self.identity,backend=type(backend).__name__,initial=[asdict(p) for p in initial],
            confirmatory=[asdict(p) for p in confirmatory],practice_digest=digest_json([asdict(p) for p in practice]),
            allow_initial_overlap=allow_initial_overlap,
            two_looks=two_looks,seed_bases=list(seed_bases),max_tokens=max_tokens,flips=flips,
            sampling={k:getattr(backend.config,k) for k in ('temperature','top_p','top_k')},
            shared_baseline=baseline_cache is not None,
            rules={'screen':'Two one-sample runs; pooled edited accuracy must exceed baseline',
                'cases':'Success/reverse pairs must hold in both screening runs; retest each on 16 new samples',
                'confirmation':'Eight samples per problem; one-sided problem sign-flip test',
                'alpha':.0294 if two_looks else .05,
                'second_look':'If first look fails, add eight samples, pool all sixteen and test at 0.0294' if two_looks else None})
        path=self.root/'staged_plan.json'
        if path.exists():
            if json.loads(path.read_text())!=self.plan: raise ValueError('Staged precommitment changed')
        elif any(self.root.iterdir()): raise ValueError('Output must be empty or contain matching staged plan')
        else: atomic_json(path,self.plan)

    def evaluate(self,name,problems,samples,seed,alpha=.05):
        result=HeldoutEvaluation(self.backend,problems,self.practice,self.root/name,samples=samples,
            seed_base=seed,max_tokens=self.max_tokens,flips=self.flips,alpha=alpha,
            identity=self.identity,baseline_cache=self.baseline_cache,
            allow_practice_overlap=self.allow_initial_overlap and not name.startswith('confirm')).run()
        return result,load_pairs(self.root/name)

    def run(self):
        screens=[]
        for index in range(2):
            _,answers=self.evaluate(f'screen{index+1}',self.initial,1,self.seeds[index])
            screens.append(answers)
        pooled=pool_runs(screens,1)
        screening=paired_report(*pooled)
        maps=[({r.problem_id:r.correct for r in a},{r.problem_id:r.correct for r in b}) for a,b in screens]
        success=[p for p in self.initial if all(not a[p.id] and b[p.id] for a,b in maps)]
        reverse=[p for p in self.initial if all(a[p.id] and not b[p.id] for a,b in maps)]
        gate=screening['correct']['delta']>0
        # Only the gate and case counts are recorded before the rest completes.
        atomic_json(self.root/'gate.json',{'passed':gate,'success_cases':len(success),'reverse_cases':len(reverse)})
        case_results={}
        for label,problems in [('success',success),('reverse',reverse)]:
            if problems:
                _,(a,b)=self.evaluate('cases-'+label,problems,16,self.seeds[2])
                case_results[label]={p.id:paired_report([r for r in a if r.problem_id==p.id],
                    [r for r in b if r.problem_id==p.id]) for p in problems}
            else: case_results[label]={}
        confirmation=None;second=None
        if gate:
            alpha=self.plan['rules']['alpha']
            confirmation,first_answers=self.evaluate('confirm1',self.confirmatory,8,self.seeds[3],alpha)
            if self.two_looks and not confirmation['significantly_better']:
                _,next_answers=self.evaluate('confirm2',self.confirmatory,8,self.seeds[4],alpha)
                a,b=pool_runs([first_answers,next_answers],8)
                second=paired_report(a,b)
                p=permutation_test(a,b,flips=self.flips,seed=0)
                second.update(primary_p=p,significantly_better=p<alpha and second['correct']['delta']>0)
        result=dict(complete=True,plan_digest=digest_json(self.plan),screen=screening,gate_passed=gate,
            cases=case_results,confirmation=confirmation,second_look=second)
        atomic_json(self.root/'staged_report.json',result)
        return result
