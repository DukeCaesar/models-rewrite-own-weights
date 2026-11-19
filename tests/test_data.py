import json
import pytest
from self_rewrite.backend import Problem
from self_rewrite.data import prepare,read_problems,require_disjoint,import_deepscaler


def test_preparation_reproducible_and_audited(tmp_path):
    pool=[Problem(str(i),f'Compute {i}+1',str(i+1)) for i in range(1100)]
    excluded=Problem('benchmark','  COMPUTE  1+1 ','2')
    kwargs=dict(practice_count=10,heldout_count=1000,reserved_count=3,require_all=False,seed=3)
    a=prepare(pool,{'test':[excluded]},tmp_path/'a',**kwargs)
    b=prepare(pool,{'test':[excluded]},tmp_path/'b',**kwargs)
    assert a==b and a['removed_ids']==['1']
    assert a['confirmatory_count']==500  # Never force the unpublished original count 490.
    visible=read_problems(tmp_path/'a'/'visible.jsonl')
    reserved=read_problems(tmp_path/'a'/'reserved.jsonl')
    require_disjoint(visible,reserved)
    assert len(visible)==7 and len(reserved)==3
    with pytest.raises(ValueError): require_disjoint(pool,[excluded])
    with pytest.raises(ValueError): prepare(pool,{},tmp_path/'bad')


def test_import_retains_source_identity(tmp_path):
    raw=tmp_path/'raw.json'; output=tmp_path/'pool.jsonl'
    raw.write_text(json.dumps([{'problem':'1+1?','answer':'2','solution':'ignored'}]))
    assert import_deepscaler(raw,output)==1
    assert read_problems(output)[0].id=='deepscaler-000000'
    assert json.loads((tmp_path/'pool.jsonl.source.json').read_text())['rows']==1


def test_empty_deepscaler_reference_is_audited_not_used(tmp_path):
    raw=tmp_path/'raw.json';out=tmp_path/'pool.jsonl'
    raw.write_text(json.dumps([{'problem':'p','answer':''},{'problem':'q','answer':'2'}]))
    assert import_deepscaler(raw,out)==1
    assert read_problems(out)[0].id=='deepscaler-000001'
    audit=json.loads((tmp_path/'pool.jsonl.source.json').read_text())
    assert audit['source_rows']==2 and audit['rejected'][0]['source_row']==0
