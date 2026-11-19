"""Convert the six pinned downloads; run after scripts/download_images.py.

Completed imports are checked and reused. An incomplete output directory must be
moved aside explicitly before retrying, so this script never discards artifacts.
"""
import argparse
import json
from pathlib import Path
import shutil
import zipfile
import zlib
from self_rewrite.adapters import import_benchmark
from self_rewrite.data import file_digest, read_problems
from self_rewrite.vision import BENCHMARKS, validate_image_problem


def extract_images(archive, root):
    root=root.resolve()
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            destination=(root/member.filename).resolve()
            if not destination.is_relative_to(root):
                raise ValueError('Archive path escapes extraction directory')
            if member.is_dir():
                continue
            if destination.exists():
                crc=0
                with destination.open('rb') as handle:
                    while chunk:=handle.read(1024*1024): crc=zlib.crc32(chunk,crc)
                if destination.stat().st_size!=member.file_size or crc!=member.CRC:
                    raise ValueError('Existing extracted image differs: '+member.filename)
                continue
            destination.parent.mkdir(parents=True,exist_ok=True)
            temporary=destination.with_suffix(destination.suffix+'.partial')
            with source.open(member) as inp, temporary.open('wb') as out:
                shutil.copyfileobj(inp,out)
            temporary.replace(destination)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,default=Path('data/image-raw'))
    parser.add_argument('--output',type=Path,default=Path('data/image-benchmarks'))
    args=parser.parse_args()
    for name in BENCHMARKS:
        raw=args.raw/name;output=args.output/name
        metadata=json.loads((raw/'download.json').read_text())
        for filename,digest in metadata['files'].items():
            if file_digest(raw/filename)!=digest:
                raise ValueError('Downloaded source hash changed: '+str(raw/filename))
        sources=[raw/p for p in metadata['files'] if Path(p).suffix in {'.parquet','.json'}]
        if (output/'manifest.json').exists():
            manifest=json.loads((output/'manifest.json').read_text())
            expected={str(p):file_digest(p) for p in sources}
            if manifest['source_files']!=expected or manifest['revision']!=metadata['revision']:
                raise ValueError('Completed import belongs to different source files')
            problems=read_problems(output/'problems.jsonl')
            if len(problems)!=manifest['imported_rows']: raise ValueError('Imported row count changed')
            for problem in problems: validate_image_problem(problem)
            print(name,'verified existing import:',len(problems),flush=True)
            continue
        kwargs={}
        if name=='screenspot-v2':
            extract_images(raw/'screenspotv2_image.zip',raw)
            kwargs=dict(image_root=raw/'screenspotv2_image',bbox_space='pixels',bbox_format='xywh')
        manifest=import_benchmark(name,sources,output,repository=metadata['repository'],
                                  revision=metadata['revision'],**kwargs)
        print(name,'imported:',manifest['imported_rows'],flush=True)


if __name__=='__main__': main()
