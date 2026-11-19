"""Local checkpoint session entry point. No model/data download is implicit."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .backend import Problem, TransformersBackend
from .checkpoint import Checkpoint
from .config import preset
from .edits import EditStore, damage_rows, clip_outliers
from .session import Session
from .worker import CodeWorker
from .factory import load_backend
from .data import read_problems, prepare, import_deepscaler
from .heldout import HeldoutEvaluation
from .config import SessionConfig



def parser():
    root=argparse.ArgumentParser(description='Independent self-rewriting reproduction')
    commands=root.add_subparsers(dest='command',required=True)
    session=commands.add_parser('session',help='Run a model-driven session under either protocol')
    session.add_argument('--checkpoint',required=True)
    session.add_argument('--practice',required=True,help='JSONL with id, question, answer strings')
    session.add_argument('--work',required=True)
    session.add_argument('--size',choices=['2B','4B','9B','27B'],default='27B')
    session.add_argument('--protocol',type=int,choices=[1,2],default=1)
    session.add_argument('--reserved',help='Protocol 2 only: inaccessible reserved practice JSONL')
    session.add_argument('--short',action='store_true',help='27B first protocol, second session (4096 tokens)')
    session.add_argument('--damage',help='Damage overlay; enables repair-session settings')
    session.add_argument('--backend',choices=['transformers','vllm'],default='transformers')
    session.add_argument('--device',default='cuda')
    session.add_argument('--turn-seed',type=int,default=0)
    session.add_argument('--resume',action='store_true')
    damage=commands.add_parser('damage',help='Create a damaged tensor overlay, separate from editor work')
    damage.add_argument('--checkpoint',required=True)
    damage.add_argument('--tensor',required=True)
    damage.add_argument('--row-start',type=int,required=True)
    damage.add_argument('--seed',type=int,required=True)
    damage.add_argument('--strength',type=float,default=200)
    damage.add_argument('--output',required=True)
    export=commands.add_parser('export',help='Export a standalone checkpoint with validated edits')
    export.add_argument('--checkpoint',required=True)
    export.add_argument('--damage')
    export.add_argument('--edits',required=True)
    export.add_argument('--output',required=True)
    convert=commands.add_parser('import-deepscaler',help='Convert a downloaded DeepScaleR JSON file to canonical JSONL')
    convert.add_argument('--source',required=True)
    convert.add_argument('--output',required=True)
    prepare_cmd=commands.add_parser('prepare',help='Create seeded practice/held-out splits and audit overlaps')
    prepare_cmd.add_argument('--pool',required=True)
    prepare_cmd.add_argument('--exclude',action='append',default=[],metavar='BENCHMARK=JSONL')
    prepare_cmd.add_argument('--manual-exclusions',help='JSON list of pool IDs identified by a semantic overlap audit')
    prepare_cmd.add_argument('--seed',type=int,default=0)
    prepare_cmd.add_argument('--output',required=True)
    evaluate=commands.add_parser('evaluate',help='Precommit and execute held-out paired evaluation')
    evaluate.add_argument('--checkpoint',required=True)
    evaluate.add_argument('--damage')
    evaluate.add_argument('--edits',required=True)
    evaluate.add_argument('--problems',required=True)
    evaluate.add_argument('--practice',required=True)
    evaluate.add_argument('--output',required=True)
    evaluate.add_argument('--report-practice-overlap',action='store_true',help='Retain and explicitly audit repeated questions (never repeated IDs)')
    evaluate.add_argument('--baseline-cache',help='Shared immutable baseline answer directory')
    evaluate.add_argument('--samples',type=int,default=8)
    evaluate.add_argument('--seed-base',type=int,default=1000003)
    evaluate.add_argument('--max-tokens',type=int,default=16384)
    evaluate.add_argument('--backend',choices=['transformers','vllm'],default='transformers')
    evaluate.add_argument('--device',default='cuda')
    replay=commands.add_parser('replay',help='Replay reported rules; this is not model-driven discovery')
    replay.add_argument('--checkpoint',required=True)
    replay.add_argument('--damage')
    replay.add_argument('--edits',required=True)
    replay.add_argument('--rule',required=True,choices=['repair','direction','time-bias'])
    replay.add_argument('--part',choices=['all','named','early'],default='all')
    replay.add_argument('--direction',help='CPU tensor file containing a practice-derived direction')
    replay.add_argument('--layer',type=int,default=8)
    replay.add_argument('--alpha',type=float,default=.05)
    images=commands.add_parser('prepare-images',help='Select fixed image benchmark samples from canonical manifests')
    images.add_argument('--source',action='append',required=True,metavar='BENCHMARK=JSONL')
    images.add_argument('--output',required=True)
    images.add_argument('--seed',type=int,default=0)
    plan=commands.add_parser('plan',help='Write the paper session schedule without running models')
    plan.add_argument('--output',required=True)
    run=commands.add_parser('run-plan',help='Execute an independent session from the paper schedule')
    run.add_argument('--plan',required=True)
    run.add_argument('--job',required=True,help='Named job in the plan')
    run.add_argument('--checkpoints',required=True,help='Directory containing Qwen3.5-{size} folders')
    run.add_argument('--data',required=True,help='Prepared split directory')
    run.add_argument('--output',required=True)
    run.add_argument('--backend',choices=['transformers','vllm'],default='transformers')
    run.add_argument('--device',default='cuda')
    history=commands.add_parser('history',help='Report protocol-2 replacement decisions, not held-out accuracy')
    history.add_argument('--output',required=True,help='Root of experiment session directories')
    source=commands.add_parser('import-benchmark',help='Convert original JSON/CSV/Parquet benchmark exports')
    source.add_argument('--benchmark',required=True)
    source.add_argument('--source',action='append',required=True)
    source.add_argument('--output',required=True)
    source.add_argument('--image-root')
    source.add_argument('--bbox-format',choices=['xyxy','xywh'])
    source.add_argument('--bbox-space',choices=['normalized','pixels'])
    source.add_argument('--exclusion-only',action='store_true',help='Extract question-only exclusion records, not evaluation answers')
    source.add_argument('--repository')
    source.add_argument('--revision')
    staged=commands.add_parser('evaluate-staged',help='Precommit and run the second-protocol evaluation stages')
    staged.add_argument('--checkpoint',required=True)
    staged.add_argument('--edits',required=True)
    staged.add_argument('--initial',required=True,help='First held-out subset, normally 200 problems')
    staged.add_argument('--confirmatory',required=True,help='Filtered held-out positions 500–999')
    staged.add_argument('--practice',required=True)
    staged.add_argument('--output',required=True)
    staged.add_argument('--baseline-cache',required=True)
    staged.add_argument('--report-initial-overlap',action='store_true',help='Retain repeated practice questions in initial screen, as audited in the paper')
    staged.add_argument('--two-looks',action='store_true',help='Precommit session-8 two-look rule before sampling')
    staged.add_argument('--device',default='cuda')
    staged.add_argument('--backend',choices=['transformers','vllm'],default='transformers')
    audit=commands.add_parser('audit-overlap',help='Retrieve lexical near-duplicate candidates for semantic review')
    audit.add_argument('--pool',required=True)
    audit.add_argument('--exclude',action='append',required=True,metavar='BENCHMARK=JSONL')
    audit.add_argument('--output',required=True)
    audit.add_argument('--threshold',type=float,default=.5)
    review=commands.add_parser('reviewed-exclusions',help='Validate completed overlap decisions and export exclusion IDs')
    review.add_argument('--audit',required=True)
    review.add_argument('--output',required=True)
    return root


def main(argv=None):
    args=parser().parse_args(argv)
    if args.command=='audit-overlap':
        from .overlap import overlap_candidates
        sources={}
        for item in args.exclude:
            name,path=item.split('=',1)
            if name in sources: raise ValueError('Duplicate benchmark name')
            sources[name]=read_problems(path)
        print(json.dumps(overlap_candidates(read_problems(args.pool),sources,args.output,threshold=args.threshold),indent=2)); return
    if args.command=='reviewed-exclusions':
        from .overlap import reviewed_exclusions
        print(json.dumps({'excluded_ids':reviewed_exclusions(args.audit,args.output)})); return
    if args.command=='evaluate-staged':
        from .staged import StagedEvaluation
        checkpoint=Checkpoint(args.checkpoint)
        edits=EditStore(args.edits,checkpoint.files,checkpoint.load)
        config=SessionConfig(size='27B',max_tokens=16384)
        backend=load_backend(checkpoint,edits,config,backend=args.backend,device=args.device)
        run=StagedEvaluation(backend,read_problems(args.initial),read_problems(args.confirmatory),
            read_problems(args.practice),args.output,two_looks=args.two_looks,baseline_cache=args.baseline_cache,allow_initial_overlap=args.report_initial_overlap)
        print(json.dumps(run.run(),indent=2)); return
    if args.command=='import-benchmark':
        from .adapters import import_benchmark
        print(json.dumps(import_benchmark(args.benchmark,args.source,args.output,
            image_root=args.image_root,bbox_space=args.bbox_space,bbox_format=args.bbox_format,repository=args.repository,revision=args.revision,exclusion_only=args.exclusion_only),indent=2)); return
    if args.command=='plan':
        from .experiments import write_plan
        print(json.dumps(write_plan(args.output),indent=2)); return
    if args.command=='history':
        from .experiments import deployment_history
        print(json.dumps(deployment_history(args.output),indent=2)); return
    if args.command=='run-plan':
        from .experiments import execute_job,session_schedule
        plan=json.loads(Path(args.plan).read_text())
        matches=[job for job in plan['jobs'] if job['name']==args.job]
        if len(matches)!=1 or matches[0] not in session_schedule():
            raise ValueError('Select one unchanged paper-schedule job')
        result=execute_job(matches[0],checkpoint_root=args.checkpoints,data_root=args.data,
                           output_root=args.output,damage_seed=plan['damage_seed'],device=args.device,backend_name=args.backend)
        print(json.dumps(result,indent=2)); return
    if args.command=='prepare-images':
        from .vision import prepare_images
        sources={}
        for item in args.source:
            name,path=item.split('=',1)
            if name in sources: raise ValueError('Repeated image benchmark')
            sources[name]=read_problems(path)
        manifest=prepare_images(sources,args.output,seed=args.seed)
        print(json.dumps({'benchmarks':list(manifest['benchmarks']),'output':args.output})); return
    if args.command=='import-deepscaler':
        print(json.dumps({'rows':import_deepscaler(args.source,args.output)})); return
    if args.command=='prepare':
        exclusions={}
        for item in args.exclude:
            name,path=item.split('=',1)
            if name in exclusions: raise ValueError('Repeated exclusion benchmark')
            exclusions[name]=read_problems(path)
        manual=json.loads(Path(args.manual_exclusions).read_text()) if args.manual_exclusions else []
        result=prepare(read_problems(args.pool),exclusions,args.output,seed=args.seed,manual_exclusions=manual)
        print(json.dumps({'confirmatory_count':result['confirmatory_count'],'output':args.output})); return
    checkpoint=Checkpoint(args.checkpoint,getattr(args,'damage',None))
    if args.command=='replay':
        from .replay import repair_replay,direction_replay,time_bias_replay
        import torch
        edits=EditStore(args.edits,checkpoint.files,checkpoint.load)
        if args.rule=='repair': names=repair_replay(checkpoint,edits,part=args.part)
        elif args.rule=='time-bias': names=time_bias_replay(checkpoint,edits)
        else:
            if not args.direction: raise ValueError('Direction replay needs a measured direction tensor')
            direction=torch.load(args.direction,map_location='cpu',weights_only=True)
            names=direction_replay(checkpoint,edits,direction,layer=args.layer,alpha=args.alpha)
        print(json.dumps({'replayed_edits':names})); return
    if args.command=='evaluate':
        config=SessionConfig(max_tokens=args.max_tokens)
        edits=EditStore(args.edits,checkpoint.files,checkpoint.load)
        backend=load_backend(checkpoint,edits,config,backend=args.backend,device=args.device)
        evaluation=HeldoutEvaluation(backend,read_problems(args.problems),read_problems(args.practice),
            args.output,samples=args.samples,seed_base=args.seed_base,max_tokens=args.max_tokens,baseline_cache=args.baseline_cache,allow_practice_overlap=args.report_practice_overlap)
        print(json.dumps(evaluation.run(),indent=2)); return
    if args.command=='damage':
        from safetensors.torch import save_file
        output=Path(args.output)
        if output.exists(): raise ValueError('Refusing to overwrite damage output')
        output.parent.mkdir(parents=True,exist_ok=True)
        tensor=damage_rows(checkpoint.load(args.tensor),args.row_start,strength=args.strength,seed=args.seed)
        save_file({args.tensor:tensor},output)
        output.with_suffix('.metadata.json').write_text(json.dumps(vars(args),indent=2))
        print(json.dumps({'damage':str(output)})); return
    if args.command=='export':
        edits=EditStore(args.edits,checkpoint.files,checkpoint.load)
        checkpoint.export(args.output,edits)
        print(json.dumps({'checkpoint':args.output})); return
    config=preset(args.size,repair=bool(args.damage),short=args.short,protocol=args.protocol,turn_seed=args.turn_seed)
    practice=read_problems(args.practice)
    reserved=read_problems(args.reserved) if args.reserved else []
    if args.protocol==2 and len(reserved)<config.reserved_count:
        raise ValueError('Protocol 2 requires 300 reserved practice problems')
    work=Path(args.work)
    if (work/'state.json').exists() and not args.resume:
        raise ValueError('Session already exists; specify --resume')
    edits=EditStore(work/'edited',checkpoint.files,checkpoint.load)
    backend=load_backend(checkpoint,edits,config,backend=args.backend,device=args.device)
    worker=CodeWorker(checkpoint,edits,practice,work,timeout=config.code_timeout,output_limit=config.output_chars,tokenizer=backend.tokenizer)
    try:
        session=Session(backend,worker,practice,work,reserved=reserved,resume=args.resume)
        print(json.dumps(session.run(),indent=2))
    finally: worker.close()


if __name__=='__main__':
    main()
