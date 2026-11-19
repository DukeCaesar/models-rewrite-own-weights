"""Fetch recorded exclusion-source revisions; no silent fallback for gated data."""
import concurrent.futures
import json
from pathlib import Path
import urllib.request
from self_rewrite.data import file_digest


def fetch(item):
    name,metadata=item
    repo=metadata['id']; revision=metadata['sha']
    files=[r['rfilename'] for r in metadata['siblings'] if r['rfilename'].endswith(('.jsonl','.csv','.parquet'))]
    if name=='gpqa-diamond': files=['gpqa_diamond.csv']
    output=Path('data/exclusion-raw')/name;output.mkdir(parents=True,exist_ok=True)
    manifest={'repository':repo,'revision':revision,'files':{}}
    try:
        for filename in files:
            target=output/filename;target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():
                if metadata.get('gated'):
                    from huggingface_hub import hf_hub_download
                    fetched=hf_hub_download(repo,filename,repo_type='dataset',revision=revision,local_dir=output)
                    assert Path(fetched)==target
                else:
                    with urllib.request.urlopen(f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}',timeout=60) as response:
                        temp=target.with_suffix(target.suffix+'.partial')
                        with temp.open('wb') as handle:
                            while chunk:=response.read(1024*1024): handle.write(chunk)
                        temp.replace(target)
            manifest['files'][filename]=file_digest(target)
        (output/'download.json').write_text(json.dumps(manifest,indent=2))
        return name,{'status':'complete','files':len(files)}
    except Exception as exc:
        return name,{'status':'failed','reason':str(exc)[:800]}

if __name__=='__main__':
    names=['math500','aime2024','aime2025','hmmt2025','amc2023','olympiadbench','gpqa-diamond']
    sources=[(name,json.loads((Path('data/source-metadata')/(name+'.json')).read_text())) for name in names]
    result={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for name,status in pool.map(fetch,sources):
            result[name]=status;print(name,status,flush=True)
    Path('data/exclusion-download-status.json').write_text(json.dumps(result,indent=2))
