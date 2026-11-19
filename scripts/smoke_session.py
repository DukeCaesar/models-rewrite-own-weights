"""Short actual-editor session; explicitly NOT a paper-budget experiment."""
import json
from pathlib import Path
from self_rewrite.backend import TransformersBackend,Problem
from self_rewrite.checkpoint import Checkpoint
from self_rewrite.config import SessionConfig
from self_rewrite.edits import EditStore
from self_rewrite.worker import CodeWorker
from self_rewrite.session import Session

if __name__=='__main__':
    work=Path('runs/qwen2b-editor-smoke')
    checkpoint=Checkpoint('checkpoints/Qwen3.5-2B')
    config=SessionConfig(size='2B',max_tokens=512,editor_tokens=2048,
                         diagnosis_count=2,passes=1,turns=2,hours=.25,code_runs=3)
    practice=[Problem('smoke-0','What is 2+2?','4'),Problem('smoke-1','What is 3 times 7?','21')]
    edits=EditStore(work/'edited',checkpoint.files,checkpoint.load)
    backend=TransformersBackend.from_checkpoint(checkpoint,edits,config,device='cuda')
    worker=CodeWorker(checkpoint,edits,practice,work,tokenizer=backend.tokenizer)
    try:
        result=Session(backend,worker,practice,work,resume=(work/'state.json').exists()).run()
        report={'scope':'Two-turn integration smoke with synthetic practice and shortened limits, not a paper result','result':result}
        (work/'smoke_report.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2),flush=True)
    finally: worker.close()
