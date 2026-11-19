import math
import pytest
import torch
from self_rewrite.edits import EditStore, damage_rows, clip_outliers, remove_direction
from self_rewrite.evaluation import Answer, grade, paired_report, permutation_test, last_boxed, problem_seed


def test_projection_has_expected_action():
    w = torch.randn(7, 11)
    u = torch.randn(7); u /= u.norm()
    x = torch.randn(11)
    edited = remove_direction(w, u, .05)
    assert torch.allclose(u @ edited @ x, .95 * (u @ w @ x), atol=1e-5)
    v = torch.randn(7); v -= u * (u @ v)
    assert torch.allclose(v @ edited @ x, v @ w @ x, atol=1e-5)


def test_damage_and_clip():
    w = torch.randn(32, 100).bfloat16()
    d = damage_rows(w, 10, seed=4)
    assert torch.equal(d[:10], w[:10]) and torch.equal(d[18:], w[18:])
    assert torch.equal(d, damage_rows(w, 10, seed=4))
    assert d[10:18].float().std() > 100 * w.float().std()
    c = clip_outliers(d)
    assert c.abs().max() <= 15 * d.float().abs().median()


def test_edit_guards_and_roundtrip(tmp_path):
    base = {'model.layers.0.mlp.down_proj.weight': torch.randn(10, 10).bfloat16(),
            'model.norm.weight': torch.ones(10)}
    store = EditStore(tmp_path, base, base.__getitem__)
    name = next(iter(base))
    for value in [base[name] * 2, base[name], torch.ones(3), torch.full((10,10), float('nan'))]:
        with pytest.raises(ValueError): store.save_edit(name, value)
    with pytest.raises(ValueError): store.save_edit('model.norm.weight', torch.arange(10.))
    edit = base[name].clone(); edit[0] = 0
    store.save_edit(name, edit)
    assert torch.equal(store.load(name), edit)
    copy = store.load_original(name); copy.zero_()
    assert torch.count_nonzero(base[name])
    store.clear(); assert not store.list_edits()


def test_grading_requires_closed_thinking_and_final_box():
    eq = lambda a,b: a == b
    def g(text): return grade('a',0,1,text,100,100,'2',comparator=eq)
    assert not g('\\boxed{2}').correct
    assert not g('<think>\\boxed{2}</think>\\boxed{3}').correct
    assert g('reason</think>\\boxed{2}').correct
    assert not g('</think><think>\\boxed{2}').correct
    assert last_boxed(r'\boxed{\frac{2}{3}}') == r'\frac{2}{3}'
    assert g('repeat ' * 2000).looping


def test_statistics_cluster_by_problem():
    def row(pid,j,correct): return Answer(pid,j,42,'',10,False,correct,False)
    base = [row(p,j,False) for p in ['a','b'] for j in range(8)]
    edited = [row(p,j,p=='a') for p in ['a','b'] for j in range(8)]
    r = paired_report(base,edited)
    assert r['correct']['delta'] == .5
    assert r['correct']['se'] == .5  # not the much smaller sample-level SE
    assert r['wrong_to_right'] == 8
    assert .45 < permutation_test(base, edited, flips=10000) < .55
    with pytest.raises(ValueError): paired_report(base,edited[:-1])
    with pytest.raises(ValueError): paired_report(base+base,edited+edited)
    assert problem_seed('a') == problem_seed('a') != problem_seed('a',1)


def test_non_power_of_two_global_scales_rejected_after_bf16_rounding(tmp_path):
    torch.manual_seed(19)
    base=torch.randn(100,100).bfloat16();base[0,0]=0
    store=EditStore(tmp_path,['layers.0.weight'],lambda name:base)
    for scalar in [1.01,.37,7.3,-.91,0.]:
        for value in [base.float()*scalar,(base.float()*scalar).bfloat16()]:
            with pytest.raises(ValueError): store.save_edit('layers.0.weight',value)
    changed=(base.float()*1.01).bfloat16();changed[0,0]=1
    store.save_edit('layers.0.weight',changed)
    assert torch.equal(store.load('layers.0.weight'),changed)
