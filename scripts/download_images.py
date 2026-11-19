"""Download only the image benchmark splits needed by the paper, at recorded revisions."""
import concurrent.futures
import json
from pathlib import Path
import urllib.request
from self_rewrite.data import file_digest

SELECTION={
 'mathvista':['data/testmini-00000-of-00001-725687bf7a18d64b.parquet'],
 'mathverse':['testmini.parquet'],
 'mmstar':['mmstar.parquet'],
 'hallusionbench':['data/image-00000-of-00001.parquet'],
 'pope':['data/test-00000-of-00003.parquet','data/test-00001-of-00003.parquet','data/test-00002-of-00003.parquet'],
 'screenspot-v2':['screenspot_desktop_v2.json','screenspot_mobile_v2.json','screenspot_web_v2.json','screenspotv2_image.zip']}


def fetch(name):
    metadata=json.loads((Path('data/source-metadata')/(name+'.json')).read_text())
    tree=json.loads((Path('data/source-metadata')/(name+'-files.json')).read_text())
    sizes={r['path']:r['size'] for r in tree if r.get('type')=='file'}
    repo,revision=metadata['id'],metadata['sha']
    root=Path('data/image-raw')/name;root.mkdir(parents=True,exist_ok=True)
    manifest={'repository':repo,'revision':revision,'files':{}}
    for filename in SELECTION[name]:
        path=root/filename;path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            partial=path.with_suffix(path.suffix+'.partial')
            offset=partial.stat().st_size if partial.exists() else 0
            request=urllib.request.Request(f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}',
                headers={'Range':f'bytes={offset}-'} if offset else {})
            with urllib.request.urlopen(request,timeout=60) as response:
                mode='ab' if offset and response.status==206 else 'wb'
                with partial.open(mode) as handle:
                    while chunk:=response.read(1024*1024): handle.write(chunk)
            if partial.stat().st_size!=sizes[filename]: raise ValueError('Downloaded file size mismatch: '+filename)
            partial.replace(path)
        if path.stat().st_size!=sizes[filename]: raise ValueError('Existing file size mismatch: '+filename)
        manifest['files'][filename]=file_digest(path)
        print(name,filename,'ready',flush=True)
    (root/'download.json').write_text(json.dumps(manifest,indent=2))
    return name

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for name in pool.map(fetch,SELECTION): print('Completed:',name,flush=True)
