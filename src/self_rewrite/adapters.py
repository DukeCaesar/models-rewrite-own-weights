"""Import original-format benchmark exports without executing dataset scripts.

Inputs: JSON arrays, JSONL, CSV, Parquet (including embedded image bytes).
Network fetch is separate; record upstream repository/revision when importing.
Schema mismatch is an error, never an empty successful benchmark.
"""
from __future__ import annotations
import csv
import hashlib
import io
import json
import re
from pathlib import Path
from .backend import Problem
from .data import write_problems,file_digest,EXCLUSION_ANSWER
from .vision import BENCHMARKS,validate_image_problem

TEXT_BENCHMARKS=('math500','aime2024','aime2025','hmmt2025','amc2023','olympiadbench','gpqa-diamond')


def raw_rows(path):
    path=Path(path)
    if path.suffix=='.parquet':
        import pyarrow.parquet as pq
        yield from pq.read_table(path).to_pylist()
    elif path.suffix=='.csv':
        with path.open(newline='') as f: yield from csv.DictReader(f)
    elif path.suffix=='.jsonl':
        for line in path.read_text().splitlines():
            if line.strip(): yield json.loads(line)
    elif path.suffix=='.json':
        data=json.loads(path.read_text())
        if not isinstance(data,list): raise ValueError('JSON input must be an array of rows')
        yield from data
    else: raise ValueError('Expected .json, .jsonl, .csv, or .parquet')


def required(row,*names):
    for name in names:
        if name in row and row[name] is not None: return row[name]
    raise ValueError('Missing required source field: '+' / '.join(names))


def text_problem(benchmark,row,index,*,exclusion_only=False):
    if benchmark not in TEXT_BENCHMARKS: raise ValueError('Unknown text benchmark')
    question=required(row,'problem','question','Question','Problem')
    answer=EXCLUSION_ANSWER if exclusion_only else required(row,'answer','Answer','final_answer','Correct Answer')
    if isinstance(answer,list):
        if len(answer)!=1: raise ValueError('Multiple reference answers need an explicit conversion rule')
        answer=answer[0]
    if not str(question).strip() or not str(answer).strip(): raise ValueError('Empty question or answer')
    # GPQA text is preserved without generated option shuffling for exclusion use.
    return Problem(f'{benchmark}-{index:06d}',str(question),str(answer))


def image_problem(benchmark,row,index,image_root,output_images,*,bbox_space=None,bbox_format=None,image_cache=None):
    from PIL import Image
    if benchmark not in BENCHMARKS: raise ValueError('Unknown image benchmark')
    if benchmark=='mathverse' and str(required(row,'problem_version')).casefold()!='vision only': return None
    if benchmark=='hallusionbench' and str(row.get('visual_input','1'))=='0': return None
    image_data=required(row,'decoded_image','image','img_filename')
    if isinstance(image_data,dict):
        if image_data.get('bytes') is not None: image_data=io.BytesIO(image_data['bytes'])
        else: image_data=required(image_data,'path')
    if isinstance(image_data,(bytes,bytearray)):
        image_data=io.BytesIO(image_data)
    if isinstance(image_data,(str,Path)):
        image_path=Path(image_data)
        if not image_path.is_absolute(): image_path=Path(image_root)/image_path
        image=Image.open(image_path)
    elif isinstance(image_data,Image.Image): image=image_data
    else: image=Image.open(image_data)
    image=image.convert('RGB')
    identifier=f'{benchmark}-{index:06d}'
    destination=Path(output_images)/(identifier+'.png')
    question=str(required(row,'question','instruction'))
    if benchmark=='screenspot-v2':
        box=list(required(row,'bbox'))
        if bbox_format not in {'xyxy','xywh'}: raise ValueError('ScreenSpot import requires explicit bbox_format')
        if bbox_format=='xywh': box=[box[0],box[1],box[0]+box[2],box[1]+box[3]]
        if bbox_space not in {'normalized','pixels'}: raise ValueError('ScreenSpot import requires explicit bbox_space')
        correction={}
        if bbox_space=='pixels':
            # Some original annotations extend one pixel beyond the screenshot.
            # Record this narrowly bounded correction; reject larger errors.
            limits=[image.width,image.height,image.width,image.height]
            if any(v<0 or v>limit for v,limit in zip(box,limits)):
                if any(v < -1 or v > limit+1 for v,limit in zip(box,limits)):
                    raise ValueError('ScreenSpot box exceeds image by more than one pixel')
                correction={'original_bbox_pixels':box[:]}
                box=[max(0,min(v,limit)) for v,limit in zip(box,limits)]
            box=[box[0]/image.width,box[1]/image.height,box[2]/image.width,box[3]/image.height]
        answer=json.dumps({'bbox':box,'coordinate_scale':[1000,1000],**correction})
        question='Locate the following target in the screenshot: '+question
        grading='point'
    elif benchmark=='mathvista':
        answer=str(required(row,'answer'));choices=row.get('choices')
        if choices:
            matches=[i for i,value in enumerate(choices) if str(value)==answer]
            if not matches: raise ValueError('MathVista answer does not match any choice')
            question+='\n'+'\n'.join(f'{chr(65+i)}. {choice}' for i,choice in enumerate(choices))
            answer=','.join(chr(65+i) for i in matches);grading='choice'
        else:
            grading='image-math'
            precision=row.get('precision')
            if precision is not None:
                import math
                if math.isfinite(float(precision)):
                    question+=f' Round the numeric answer to {int(precision)} decimal places.'
            if row.get('unit'): question+=' Give the answer in '+str(row['unit'])+'.'
    elif benchmark=='mathverse':
        question=str(required(row,'query_cot'))
        answer=str(required(row,'answer'))
        grading='choice' if str(row.get('question_type')).replace('_','-')=='multi-choice' else 'image-math'
        if grading=='choice':
            labels=re.fullmatch(r'\(?([A-Z])\)?',answer.strip())
            if labels: answer=labels[1]
            elif re.fullmatch(r'[A-Z](?:\s+[A-Z])+',answer.strip()):
                answer=','.join(sorted(set(answer.split())));grading='choice-set'
            else:
                # Evaluation-only source text is used to map a value to its label;
                # it is never added to the vision-only model prompt.
                choices=re.findall(r'^\s*([A-Z])[.:]\s*(.*?)\s*$',str(row.get('question_for_eval','')),re.M)
                matches=sorted({label for label,value in choices if value==answer.strip()})
                if not matches: raise ValueError('MathVerse reference does not match any option')
                answer=','.join(matches)
    elif benchmark=='mmstar': answer=str(required(row,'answer'));grading='choice'
    else:
        answer=str(required(row,'answer','gt_answer','label')).lower()
        answer={'1':'yes','0':'no'}.get(answer,answer)
        grading='yesno'
    destination.parent.mkdir(parents=True,exist_ok=True)
    if image_cache is None:
        image.save(destination)
    else:
        key=(image.size,hashlib.sha256(image.tobytes()).hexdigest())
        if key in image_cache:
            destination=image_cache[key]
        else:
            image.save(destination)
            image_cache[key]=destination
    problem=Problem(identifier,question,answer,str(destination.resolve()),grading,benchmark)
    validate_image_problem(problem)
    return problem


def import_benchmark(benchmark,sources,output,*,image_root=None,bbox_space=None,bbox_format=None,repository=None,revision=None,exclusion_only=False):
    root=Path(output)
    if root.exists() and any(root.iterdir()): raise ValueError('Import output must be empty')
    root.mkdir(parents=True,exist_ok=True)
    rows=[];count=0;excluded=[];image_cache={}
    for source in sources:
        for raw in raw_rows(source):
            if benchmark=='amc2023':
                import re
                year=raw.get('year')
                if year is None:
                    match=re.search(r'(20\d{2})_AMC_',str(raw.get('url','')))
                    year=match[1] if match else None
                if year is None: raise ValueError('AMC source needs a year field or dated source URL')
                if str(year)!='2023':
                    excluded.append(count);count+=1;continue
            if benchmark in TEXT_BENCHMARKS: problem=text_problem(benchmark,raw,count,exclusion_only=exclusion_only)
            else: problem=image_problem(benchmark,raw,count,image_root or Path(source).parent,root/'images',bbox_space=bbox_space,bbox_format=bbox_format,image_cache=image_cache)
            if problem is not None: rows.append(problem)
            else: excluded.append(count)
            count+=1
    if not rows: raise ValueError('No benchmark rows imported')
    write_problems(root/'problems.jsonl',rows)
    manifest={'benchmark':benchmark,'source_files':{str(p):file_digest(p) for p in sources},
              'repository':repository,'revision':revision,'input_rows':count,'imported_rows':len(rows),
              'filtered_row_indices':excluded,'bbox_space':bbox_space,'bbox_format':bbox_format,
              'purpose':'exclusion-only' if exclusion_only else 'evaluation',
              'note':'Source-row IDs are reconstruction identifiers. Original paper selections unavailable.'}
    if benchmark in BENCHMARKS: manifest['unique_images']=len(image_cache)
    if benchmark=='screenspot-v2':
        manifest['pixel_boundary_tolerance']=1
        manifest['box_corrections']=[{'id':p.id,**json.loads(p.answer)} for p in rows
                                     if 'original_bbox_pixels' in json.loads(p.answer)]
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest
