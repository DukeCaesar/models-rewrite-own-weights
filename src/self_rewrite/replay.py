"""Replay reported edit rules, distinct from model-driven discovery sessions."""
import re
import torch
from .edits import clip_outliers, remove_direction

DAMAGE_LOCATIONS={'2B':(11,1724),'4B':(15,1660),'9B':(15,3420),'27B':(30,3068)}


def repair_replay(checkpoint,edits,*,part='all'):
    if part not in {'all','named','early'}: raise ValueError('Unknown repair ablation')
    names=[]
    for name in checkpoint.files:
        if 'mtp' in name.split('.') or 'visual' in name.split('.'): continue
        match=re.search(r'(?:^|\.)layers\.(\d+)\.',name)
        if not match: continue
        layer=int(match[1])
        if layer==30 and name.endswith('mlp.down_proj.weight') and part!='early': names.append(name)
        # Figure 12's explicit MLP loop skips layer 4; attention covers 0..10.
        early_mlp=layer in {0,1,2,3,5,6,7,8,9,10} and name.endswith('mlp.down_proj.weight')
        early_attn=layer<=10 and name.endswith('linear_attn.out_proj.weight')
        if (early_mlp or early_attn) and part!='named': names.append(name)
    expected={'all':20,'named':1,'early':19}[part]
    if len(names)!=expected: raise ValueError('Reported 27B repair requires the expected architecture/tensor names')
    if edits.list_edits(): raise ValueError('Replay requires an empty edit store')
    for name in names:
        original=checkpoint.load(name)
        changed=clip_outliers(original,15).to(torch.bfloat16)
        if not torch.equal(original.to(torch.bfloat16),changed): edits.save_edit(name,changed)
    return edits.list_edits()


def direction_replay(checkpoint,edits,direction,*,layer=8,alpha=.05):
    """Direction must be measured from practice reads; original vectors unpublished."""
    names=checkpoint.output_matrices(layer)
    if len(names)!=2: raise ValueError('Expected two output matrices at chosen layer')
    if edits.list_edits(): raise ValueError('Replay requires an empty edit store')
    for name in names: edits.save_edit(name,remove_direction(checkpoint.load(name),direction,alpha))
    return edits.list_edits()


def time_bias_replay(checkpoint,edits,*,increment=.5,layers=None):
    """Protocol-2 session 8: +0.5 to all 48 linear-attention dt_bias tensors."""
    names=[]
    for name in checkpoint.files:
        if 'mtp' in name.split('.') or 'visual' in name.split('.'): continue
        match=re.search(r'(?:^|\.)layers\.(\d+)\.linear_attn\.dt_bias$',name)
        if match and (layers is None or int(match[1]) in layers): names.append(name)
    if layers is None and len(names)!=48: raise ValueError('Expected 48 linear-attention biases for the 27B replay')
    if not names or edits.list_edits(): raise ValueError('No target biases, or edit store not empty')
    for name in names: edits.save_edit(name,checkpoint.load(name).float()+increment)
    return edits.list_edits()
