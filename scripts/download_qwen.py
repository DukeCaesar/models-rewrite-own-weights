"""Download pinned official Qwen assets; record resolved upstream revision."""
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

if __name__=='__main__':
    repo='Qwen/Qwen3.5-2B'
    revision=HfApi().model_info(repo).sha
    target=Path('checkpoints/Qwen3.5-2B')
    print('Resolved revision:',revision,flush=True)
    snapshot_download(repo,revision=revision,local_dir=target,
        allow_patterns=['*.json','*.jinja','*.safetensors','*.txt','*.model'],max_workers=4)
    (target/'download_provenance.json').write_text(json.dumps(dict(repository=repo,revision=revision),indent=2))
    print('Download complete',flush=True)
