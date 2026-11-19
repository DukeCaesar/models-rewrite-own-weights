"""Deterministic reconstruction of dataset splits with explicit provenance.

The original split IDs and semantic deduplication rules were not released.
Normalized exact matching here does not detect paraphrases. User-supplied manual
exclusion IDs can capture semantic overlaps identified by an external audit.
"""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import unicodedata
from .backend import Problem

EXCLUSION_ANSWER='__EXCLUSION_ONLY__'

BENCHMARKS={'math500','aime2024','aime2025','hmmt2025','amc2023','olympiadbench','gpqa-diamond'}


def question_key(question):
    return ' '.join(unicodedata.normalize('NFKC',question).split()).casefold()


def digest_json(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def file_digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def read_problems(path):
    problems=[]
    for line in Path(path).read_text().splitlines():
        if not line.strip(): continue
        row=json.loads(line)
        for field in ('id','question','answer'):
            if not isinstance(row.get(field),str) or not row[field]:
                raise ValueError(f'Problem {field} must be a nonempty string')
        problem=Problem(**{k:row[k] for k in ('id','question','answer','image','grading','benchmark') if k in row})
        if problem.image:
            image=Path(problem.image)
            if not image.is_absolute():
                from dataclasses import replace
                problem=replace(problem,image=str((Path(path).parent/image).resolve()))
            from .vision import validate_image_problem
            validate_image_problem(problem)
        elif problem.grading!='math': raise ValueError('Image grading requires an image')
        problems.append(problem)
    if not problems or len({p.id for p in problems})!=len(problems):
        raise ValueError('Problem file must contain nonempty, unique IDs')
    return problems


def write_problems(path,problems):
    Path(path).write_text(''.join(json.dumps(asdict(p),ensure_ascii=False)+'\n' for p in problems))


def require_disjoint(left,right):
    ids={p.id for p in left}
    questions={question_key(p.question) for p in left}
    overlaps=[p.id for p in right if p.id in ids or question_key(p.question) in questions]
    if overlaps: raise ValueError(f'Split overlap in IDs or normalized questions: {overlaps[:10]}')


def prepare(pool,exclusions,output,*,seed=0,practice_count=3000,heldout_count=1000,
            reserved_count=300,manual_exclusions=(),require_all=True):
    if require_all and set(exclusions)!=BENCHMARKS:
        raise ValueError('Provide all seven exclusion benchmarks: '+', '.join(sorted(BENCHMARKS)))
    if min(practice_count,heldout_count)<1 or not 0<=reserved_count<practice_count:
        raise ValueError('Invalid split sizes')
    if len({p.id for p in pool})!=len(pool): raise ValueError('Pool IDs must be unique')
    root=Path(output)
    if root.exists() and any(root.iterdir()): raise ValueError('Output directory must be empty')
    blocked={question_key(p.question) for group in exclusions.values() for p in group}
    manual=set(manual_exclusions)
    if manual-{p.id for p in pool}: raise ValueError('Unknown manual exclusion IDs')
    eligible=[]; removed=[]
    for p in pool:
        if question_key(p.question) in blocked or p.id in manual: removed.append(p.id)
        else: eligible.append(p)
    random.Random(seed).shuffle(eligible)
    if len(eligible)<practice_count+heldout_count: raise ValueError('Insufficient eligible problems')
    # Preserve within-pool duplicates, as in the paper, and audit them explicitly.
    practice=eligible[:practice_count]
    heldout=eligible[practice_count:practice_count+heldout_count]
    # Reserve from practice, never from the held-out split. Exclude text duplicates
    # of reserved items from editor-visible problems, as well as their identifiers.
    reserved=practice[-reserved_count:] if reserved_count else []
    reserved_keys={question_key(p.question) for p in reserved}
    visible=[p for p in (practice[:-reserved_count] if reserved_count else practice)
             if question_key(p.question) not in reserved_keys]
    practice_keys={question_key(p.question) for p in practice}
    seen=set(practice_keys)
    confirmatory=[]; rejected=[]
    for i,p in enumerate(heldout):
        key=question_key(p.question)
        if i>=500:
            if key in seen: rejected.append(p.id)
            else: confirmatory.append(p)
        seen.add(key)
    overlaps=[p.id for p in heldout if question_key(p.question) in practice_keys]
    manifest=dict(seed=seed,pool_digest=digest_json([asdict(p) for p in pool]),
        exclusions={name:digest_json([asdict(p) for p in rows]) for name,rows in exclusions.items()},
        manual_exclusions=sorted(manual),removed_ids=removed,
        practice_ids=[p.id for p in practice],heldout_ids=[p.id for p in heldout],
        reserved_ids=[p.id for p in reserved],heldout_practice_overlaps=overlaps,
        confirmatory_rejected_ids=rejected,confirmatory_count=len(confirmatory),
        reconstruction='Seeded shuffle; original IDs unavailable; normalized exact overlap only, not semantic deduplication.')
    root.mkdir(parents=True,exist_ok=True)
    for name,rows in [('practice',practice),('visible',visible),('reserved',reserved),
                      ('heldout',heldout),('initial',heldout[:200]),('confirmatory',confirmatory)]:
        write_problems(root/(name+'.jsonl'),rows)
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


def import_deepscaler(source,output):
    """Convert the public deepscaler.json list; retain source row IDs and hash."""
    rows=json.loads(Path(source).read_text())
    problems=[];rejected=[]
    for i,row in enumerate(rows):
        if not isinstance(row.get('problem'),str) or not isinstance(row.get('answer'),str):
            raise ValueError('Expected DeepScaleR problem and answer strings')
        if not row['problem'].strip() or not row['answer'].strip():
            rejected.append({'source_row':i,'reason':'Empty problem or reference answer'})
            continue
        problems.append(Problem(f'deepscaler-{i:06d}',row['problem'],row['answer']))
    if Path(output).exists(): raise ValueError('Output already exists')
    write_problems(output,problems)
    Path(str(output)+'.source.json').write_text(json.dumps(dict(source=str(source),sha256=file_digest(source),rows=len(problems),source_rows=len(rows),rejected=rejected),indent=2))
    return len(problems)
