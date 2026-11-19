"""Actual Qwen smoke verification, not a paper-budget accuracy experiment."""
import json
from pathlib import Path
import torch
from PIL import Image
from self_rewrite.backend import TransformersBackend,Problem
from self_rewrite.checkpoint import Checkpoint
from self_rewrite.config import SessionConfig
from self_rewrite.edits import EditStore

if __name__=='__main__':
    work=Path('runs/qwen2b-smoke');work.mkdir(parents=True,exist_ok=True)
    checkpoint=Checkpoint('checkpoints/Qwen3.5-2B')
    edits=EditStore(work/'edited',checkpoint.files,checkpoint.load)
    config=SessionConfig(size='2B',max_tokens=128,editor_tokens=128)
    backend=TransformersBackend.from_checkpoint(checkpoint,edits,config,device='cuda')
    print('Loaded Qwen',flush=True)
    text=backend.generate([Problem('smoke-math','What is 2+2?','4')])
    print('Text generation finished',flush=True)
    read=backend.read(['The answer is four.'],layers=[0,3],positions='last',last=2)
    print('Read finished',flush=True)
    name=checkpoint.output_matrices(0)[0]
    original=backend._parameters[name].detach().cpu().clone()
    w=checkpoint.load(name).float();w[0]*=.99
    edits.save_edit(name,w)
    with backend.weights('edited'):
        applied=torch.equal(backend._parameters[name].cpu(),edits.load(name).to(original.dtype))
    restored=torch.equal(backend._parameters[name].cpu(),original)
    edits.clear()
    image=(work/'white.png').resolve();Image.new('RGB',(64,64),'white').save(image)
    vision=backend.generate([Problem('smoke-image','Is this image white? Answer yes or no.','yes',str(image),'yesno','pope')])
    report={'scope':'Integration smoke only: short budgets, synthetic questions, not a benchmark result',
            'checkpoint':json.loads((checkpoint.root/'download_provenance.json').read_text()),
            'text_answers':[a.to_dict() for a in text],'image_answers':[a.to_dict() for a in vision],
            'hidden_shapes':[{k:list(v.shape) for k,v in item.items()} for item in read],
            'edit_applied':applied,'clean_weights_restored':restored,
            'peak_gpu_bytes':torch.cuda.max_memory_allocated()}
    (work/'report.json').write_text(json.dumps(report,indent=2))
    assert applied and restored
    print(json.dumps(report,indent=2),flush=True)
