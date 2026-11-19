"""Run in an environment with a compatible vLLM/Qwen3.5 CUDA installation."""
import json
from pathlib import Path
from self_rewrite.backend import Problem
from self_rewrite.checkpoint import Checkpoint
from self_rewrite.config import SessionConfig
from self_rewrite.edits import EditStore
from self_rewrite.vllm_backend import VLLMBackend

if __name__=='__main__':
    work=Path('runs/vllm-smoke');work.mkdir(parents=True,exist_ok=True)
    checkpoint=Checkpoint('checkpoints/Qwen3.5-2B')
    edits=EditStore(work/'edited',checkpoint.files,checkpoint.load)
    config=SessionConfig(size='2B',max_tokens=512,editor_tokens=512,context_reset_tokens=2048)
    backend=VLLMBackend.from_checkpoint(checkpoint,edits,config,max_model_len=4096,gpu_memory_utilization=.8)
    problem=Problem('smoke-vllm','What is 2+2?','4')
    before=backend.generate([problem])
    name=checkpoint.output_matrices(0)[0]
    weight=checkpoint.load(name).float();weight[0]*=.99
    edits.save_edit(name,weight)
    changed=backend.generate([problem],weights='edited')
    after=backend.generate([problem])
    report={'scope':'vLLM execution smoke only, not a paper benchmark',
            'before':[r.to_dict() for r in before],'edited':[r.to_dict() for r in changed],
            'after':[r.to_dict() for r in after]}
    (work/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
