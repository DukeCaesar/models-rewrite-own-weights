"""Paper session schedules: independent base starts, never cumulative edits."""
import json
from pathlib import Path
from .config import preset
from .replay import DAMAGE_LOCATIONS


def session_schedule():
    jobs=[]
    for size in ('2B','4B','9B','27B'):
        jobs.append(dict(name=f'protocol1-{size}-self',size=size,protocol=1,repair=False,short=False,turn_seed=0))
        jobs.append(dict(name=f'protocol1-{size}-repair',size=size,protocol=1,repair=True,short=False,turn_seed=0))
    jobs.append(dict(name='protocol1-27B-self-short',size='27B',protocol=1,repair=False,short=True,turn_seed=0))
    for seed in range(8):
        jobs.append(dict(name=f'protocol2-27B-session{seed+1}',size='27B',protocol=2,repair=False,short=False,turn_seed=seed))
    return jobs


def write_plan(output):
    path=Path(output)
    if path.exists(): raise ValueError('Plan already exists')
    plan={'paper':'https://arxiv.org/html/2611.21885','jobs':session_schedule(),
          'damage_seed':0,'damage_seed_note':'Reconstruction choice; original seed unavailable',
          'damage_locations':DAMAGE_LOCATIONS,
          'policy':'Each session starts from base weights. A submitted protocol-2 edit replaces the model in use; withdrawal retains it. Edits are never composed.'}
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(plan,indent=2))
    return plan


def execute_job(job,*,checkpoint_root,data_root,output_root,damage_seed=0,device='cuda',backend_name='transformers'):
    from .checkpoint import Checkpoint
    from .edits import EditStore,damage_rows
    from .factory import load_backend
    from .data import read_problems
    from .worker import CodeWorker
    from .session import Session
    from safetensors.torch import save_file
    checkpoint_path=Path(checkpoint_root)/f'Qwen3.5-{job["size"]}'
    work=Path(output_root)/job['name']
    config=preset(job['size'],protocol=job['protocol'],repair=job['repair'],short=job['short'],turn_seed=job['turn_seed'])
    damage=None
    if job['repair']:
        clean=Checkpoint(checkpoint_path)
        layer,row=DAMAGE_LOCATIONS[job['size']]
        names=[n for n in clean.output_matrices(layer) if n.endswith('mlp.down_proj.weight')]
        if len(names)!=1: raise ValueError('Cannot resolve damage tensor')
        damage=Path(output_root)/'damage'/f'{job["size"]}-seed{damage_seed}.safetensors'
        damage.parent.mkdir(parents=True,exist_ok=True)
        if not damage.exists(): save_file({names[0]:damage_rows(clean.load(names[0]),row,seed=damage_seed)},damage)
    checkpoint=Checkpoint(checkpoint_path,damage)
    data_root=Path(data_root)
    practice=read_problems(data_root/('visible.jsonl' if job['protocol']==2 else 'practice.jsonl'))
    reserved=read_problems(data_root/'reserved.jsonl') if job['protocol']==2 else []
    edits=EditStore(work/'edited',checkpoint.files,checkpoint.load)
    backend=load_backend(checkpoint,edits,config,backend=backend_name,device=device)
    worker=CodeWorker(checkpoint,edits,practice,work,timeout=config.code_timeout,output_limit=config.output_chars,tokenizer=backend.tokenizer)
    try: return Session(backend,worker,practice,work,reserved=reserved,resume=(work/'state.json').exists()).run()
    finally: worker.close()


def deployment_history(output_root):
    """Audit submitted replacement selection without looking at held-out scores."""
    active='base';history=[]
    for job in session_schedule():
        if job['protocol']!=2: continue
        state_path=Path(output_root)/job['name']/'state.json'
        if not state_path.exists(): break
        state=json.loads(state_path.read_text())
        if state['phase']!='done': break
        if state['result']['submitted']: active=job['name']
        history.append({'session':job['name'],'submitted':state['result']['submitted'],'model_in_use':active})
    return history
