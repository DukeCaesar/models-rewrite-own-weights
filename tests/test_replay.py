import torch
from safetensors.torch import save_file
from self_rewrite.checkpoint import Checkpoint
from self_rewrite.edits import EditStore
from self_rewrite.replay import repair_replay,time_bias_replay


def test_repair_uses_figure12_explicit_mlp_list(tmp_path):
    tensors={}
    for layer in range(64):
        w=torch.ones(8,8);w[0,0]=100
        tensors[f'model.layers.{layer}.mlp.down_proj.weight']=w
        if layer%4!=3:
            tensors[f'model.layers.{layer}.linear_attn.out_proj.weight']=w.clone()
            tensors[f'model.layers.{layer}.linear_attn.dt_bias']=torch.arange(8.)
    tensors['mtp.layers.0.mlp.down_proj.weight']=torch.ones(8,8)
    root=tmp_path/'base';root.mkdir();save_file(tensors,root/'model.safetensors')
    checkpoint=Checkpoint(root)
    assert len(checkpoint.output_matrices(0))==2
    all_edits=EditStore(tmp_path/'all',checkpoint.files,checkpoint.load)
    names=repair_replay(checkpoint,all_edits)
    assert len(names)==20
    assert 'model.layers.4.mlp.down_proj.weight' not in names
    assert 'model.layers.4.linear_attn.out_proj.weight' in names
    assert all_edits.load('model.layers.30.mlp.down_proj.weight')[0,0]==15
    named=EditStore(tmp_path/'named',checkpoint.files,checkpoint.load)
    early=EditStore(tmp_path/'early',checkpoint.files,checkpoint.load)
    assert set(repair_replay(checkpoint,named,part='named')) | set(repair_replay(checkpoint,early,part='early'))==set(names)
    biases=EditStore(tmp_path/'biases',checkpoint.files,checkpoint.load)
    assert len(time_bias_replay(checkpoint,biases))==48
    assert torch.equal(biases.load('model.layers.0.linear_attn.dt_bias'),(torch.arange(8.)+.5).bfloat16())
