"""Deterministic image-benchmark manifests and transparent answer graders.

No external judge model is used. Benchmark normalization rules and click coordinate
scales must be explicit in the input manifest; these are reconstruction choices.
"""
from dataclasses import replace
import json
import re
from pathlib import Path
import random
from .evaluation import grade, last_boxed, math_equal

BENCHMARKS=('mathvista','mathverse','mmstar','hallusionbench','pope','screenspot-v2')


def final_text(text):
    if '</think>' not in text or text.rfind('</think>')<text.rfind('<think>'): return None
    final=text.rsplit('</think>',1)[-1]
    return re.sub(r'<\|[^>]+\|>','',final).strip()


def extract_stated(text):
    boxed=last_boxed(text)
    if boxed is not None: return boxed.strip()
    matches=list(re.finditer(r'(?:final\s+)?answer\s*(?:is|:|=)\s*(.+)',text,re.I))
    return (matches[-1].group(1) if matches else text).strip().strip('$').strip()


def image_correct(text,reference,kind):
    final=final_text(text)
    if final is None: return False
    stated=extract_stated(final)
    if kind=='image-math': return math_equal(stated,reference)
    if kind=='choice':
        match=re.fullmatch(r'\s*\(?([A-Z])\)?[.。]?\s*',stated.upper())
        return bool(match and match[1] in reference.strip().upper().split(','))
    if kind=='choice-set':
        normalized=re.sub(r'\s+and\s+',',',stated.upper(),flags=re.I)
        if not re.fullmatch(r'\s*[A-Z](?:[\s,;&]+[A-Z])+[.]?\s*',normalized): return False
        return set(re.findall('[A-Z]',normalized))==set(reference.split(','))
    if kind=='yesno':
        match=re.match(r'^(yes|no)\b',stated,re.I)
        return bool(match and match[1].lower()==reference.strip().lower())
    if kind=='point':
        target=json.loads(reference)
        bbox=target['bbox']; scale=target['coordinate_scale']
        number=r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)'
        points=re.findall(r'[\[(]\s*('+number+r')\s*,\s*('+number+r')\s*[\])]',stated)
        if len(points)!=1: return False
        x,y=map(float,points[0]);x/=scale[0];y/=scale[1]
        return bbox[0]<=x<=bbox[2] and bbox[1]<=y<=bbox[3]
    raise ValueError('Unknown image grading kind')


def grade_image(problem,sample,seed,text,tokens,limit,finish_reason):
    row=grade(problem.id,sample,seed,text,tokens,limit,problem.answer,
              finish_reason=finish_reason,comparator=lambda a,b:False)
    return replace(row,correct=image_correct(text,problem.answer,problem.grading))


def validate_image_problem(problem):
    if problem.benchmark not in BENCHMARKS: raise ValueError('Unknown image benchmark')
    if problem.grading not in {'image-math','choice','choice-set','yesno','point'}: raise ValueError('Invalid image grader')
    if not problem.image or not Path(problem.image).is_file(): raise ValueError('Image must be an existing local file')
    if problem.grading in {'choice','choice-set'} and not re.fullmatch('[A-Z](?:,[A-Z])*',problem.answer): raise ValueError('Choice reference must be an uppercase letter or comma-separated accepted letters')
    if problem.grading=='yesno' and problem.answer not in {'yes','no'}: raise ValueError('Binary reference must be yes or no')
    if problem.grading=='point':
        ref=json.loads(problem.answer);box=ref['bbox'];scale=ref['coordinate_scale']
        if len(box)!=4 or not (0<=box[0]<=box[2]<=1 and 0<=box[1]<=box[3]<=1): raise ValueError('Box must be normalized xyxy')
        if len(scale)!=2 or any(not isinstance(x,(int,float)) or not 0<x<1e9 for x in scale): raise ValueError('Invalid coordinate scale')


def prepare_images(sources,output,*,seed=0,draw_count=300,evaluate_count=100):
    from .data import write_problems,file_digest,digest_json
    if set(sources)!=set(BENCHMARKS): raise ValueError('Supply all six image benchmarks')
    if not 1<=evaluate_count<=draw_count: raise ValueError('Invalid image sample counts')
    root=Path(output)
    if root.exists() and any(root.iterdir()): raise ValueError('Output must be empty')
    selected=[];manifest={'seed':seed,'draw_count':draw_count,'evaluate_count':evaluate_count,'benchmarks':{}}
    for name in BENCHMARKS:
        rows=list(sources[name])
        if len(rows)<draw_count or len({p.id for p in rows})!=len(rows): raise ValueError('Insufficient or duplicate image examples')
        for problem in rows:
            validate_image_problem(problem)
            if problem.benchmark!=name: raise ValueError('Benchmark label mismatch')
        random.Random(seed).shuffle(rows)
        draw=rows[:draw_count];evaluated=draw[:evaluate_count]
        selected.extend(evaluated)
        manifest['benchmarks'][name]={'draw_ids':[p.id for p in draw],
            'evaluated_ids':[p.id for p in evaluated],
            'source_digest':digest_json([p.__dict__ for p in sources[name]]),
            'image_hashes':{p.id:file_digest(p.image) for p in evaluated}}
    if len({p.id for p in selected})!=len(selected): raise ValueError('IDs must be globally unique across image benchmarks')
    root.mkdir(parents=True,exist_ok=True)
    write_problems(root/'images.jsonl',selected)
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest
