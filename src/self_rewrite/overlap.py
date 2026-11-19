"""Reviewable lexical candidate retrieval, not a semantic decontamination oracle."""
from collections import Counter
import json
import re
from pathlib import Path
import unicodedata
import numpy as np
from scipy.sparse import csr_matrix
from .data import digest_json,question_key


def tokens(text):
    text=unicodedata.normalize('NFKC',text).casefold()
    text=re.sub(r'\\(?:left|right)\b','',text)
    text=text.replace('\\dfrac','\\frac').replace('\\tfrac','\\frac')
    return re.findall(r'\d+(?:\.\d+)?|[a-z]+|[+*/=<>^-]',text)


def features(text):
    words=tokens(text)
    return Counter(tuple(words[i:i+n]) for n in (2,3) for i in range(len(words)-n+1))


def number_signature(text):
    return sorted(re.findall(r'\d+(?:\.\d+)?',unicodedata.normalize('NFKC',text)))


def matrix(rows,vocabulary):
    values=[];indices=[];offsets=[0]
    for question in rows:
        for term,count in features(question).items():
            index=vocabulary.get(term)
            if index is not None: indices.append(index);values.append(1+np.log(count))
        offsets.append(len(values))
    return csr_matrix((values,indices,offsets),shape=(len(rows),len(vocabulary)),dtype=np.float64)


def normalize(matrix):
    norm=np.sqrt(np.asarray(matrix.multiply(matrix).sum(1)).ravel())
    return matrix.multiply((1/np.maximum(norm,1e-12))[:,None]).tocsr()


def overlap_candidates(pool,benchmarks,output,*,threshold=.5,top_k=3,batch_size=128):
    if not 0<threshold<=1 or top_k<1 or batch_size<1: raise ValueError('Invalid audit settings')
    target=Path(output)
    if target.exists(): raise ValueError('Audit output already exists')
    references=[(name,p) for name,rows in benchmarks.items() for p in rows]
    if not pool or not references: raise ValueError('Both pool and references are required')
    vocabulary={}
    for _,p in references:
        for term in features(p.question):
            if term not in vocabulary: vocabulary[term]=len(vocabulary)
    if not vocabulary: raise ValueError('Source questions have no indexable tokens')
    reference=matrix([p.question for _,p in references],vocabulary)
    df=np.asarray((reference>0).sum(0)).ravel()
    idf=np.log((1+len(references))/(1+df))+1
    reference=normalize(reference.multiply(idf)).T.tocsr()
    count=0
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_suffix('.tmp')
    with temporary.open('w') as handle:
        for start in range(0,len(pool),batch_size):
            batch=pool[start:start+batch_size]
            queries=normalize(matrix([p.question for p in batch],vocabulary).multiply(idf))
            scores=(queries@reference).tocsr()
            for i,p in enumerate(batch):
                row=scores.getrow(i)
                order=sorted(range(len(row.data)),key=lambda j:(-row.data[j],int(row.indices[j])))[:top_k]
                for j in order:
                    score=float(row.data[j])
                    if score<threshold: continue
                    name,other=references[int(row.indices[j])]
                    item={'pool_id':p.id,'benchmark':name,'benchmark_id':other.id,'cosine':score,
                          'normalized_exact':question_key(p.question)==question_key(other.question),
                          'same_numbers':number_signature(p.question)==number_signature(other.question),
                          'pool_question':p.question,'benchmark_question':other.question,
                          'decision':'unreviewed'}
                    handle.write(json.dumps(item,ensure_ascii=False)+'\n');count+=1
    temporary.replace(target)
    manifest={'method':'TF-IDF token bigram/trigram cosine; reference-fitted vocabulary',
              'threshold':threshold,'top_k_per_pool_item':top_k,'candidates':count,
              'pool_hash':digest_json([p.__dict__ for p in pool]),
              'source_hashes':{name:digest_json([p.__dict__ for p in rows]) for name,rows in benchmarks.items()},
              'warning':'Candidate retrieval only; low scores and absence of a candidate do not prove semantic disjointness.'}
    target.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


def reviewed_exclusions(audit_path,output):
    """Require a decision on every retrieved pair before producing exclusion IDs."""
    selected=set();count=0
    for line in Path(audit_path).read_text().splitlines():
        row=json.loads(line);decision=row.get('decision')
        if decision not in {'overlap','distinct'}: raise ValueError('Every candidate requires overlap/distinct review')
        if not row.get('review_note','').strip(): raise ValueError('Every review requires a rationale')
        if decision=='overlap': selected.add(row['pool_id'])
        count+=1
    if not count: raise ValueError('No review decisions found')
    target=Path(output)
    if target.exists(): raise ValueError('Output already exists')
    target.write_text(json.dumps(sorted(selected),indent=2))
    return sorted(selected)
