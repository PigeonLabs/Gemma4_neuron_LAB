"""Synthetic CPU delivery checks only; these are not 12B experiments."""
import torch
from neuron_lab.round7_tools import AttentionEdges,RoundTripQuantizer


def test_attention_edge_mask_is_local_out_of_place_and_removed():
    class Attention(torch.nn.Module):
        def forward(self,hidden_states,attention_mask):return attention_mask.clone()
    class Layer:pass
    class Inv:pass
    inv=Inv();layer=Layer();layer.self_attn=Attention();inv.layers={0:layer}
    enc={'tokens':4,'regions':{'later':[2,3],'source':[0,1]}}
    mask=torch.zeros(1,1,4,4);mask[:,:,torch.triu(torch.ones(4,4,dtype=torch.bool),1)]=torch.finfo(torch.float32).min
    original=mask.clone();x=torch.zeros(1,4,2)
    base=layer.self_attn(x,attention_mask=mask)
    with AttentionEdges(inv,enc,[{'queries':'later','keys':'source'}]) as hooks:
        changed=layer.self_attn(x,attention_mask=mask)
        assert hooks.events[0]['removed_previously_allowed_edges']==4
    expected=original.clone();expected[:,:,2:,:2]=torch.finfo(torch.float32).min
    assert torch.equal(changed,expected) and torch.equal(mask,original)
    assert torch.equal(layer.self_attn(x,attention_mask=mask),base) and not layer.self_attn._forward_pre_hooks


def test_roundtrip_graph_edits_only_query_and_keeps_tensor_and_tuple_types():
    class Layer(torch.nn.Module):
        def forward(self,x):return x,None
    class Inv:layers={0:Layer()}
    x=torch.tensor([[[.10001,.20003],[.30007,.40009],[.50011,.60013]]]);enc={'regions':{'query':[2]}}
    with RoundTripQuantizer(Inv,enc,[0],mode='capture') as q:clean=Inv.layers[0](x)
    assert torch.equal(clean[0],x)
    with RoundTripQuantizer(Inv,enc,[0],mode='round') as q2:rounded=Inv.layers[0](x)
    assert rounded[1] is None and torch.equal(rounded[0][:,:2],x[:,:2])
    assert torch.equal(rounded[0][:,2:],x[:,2:].bfloat16().float())
    with RoundTripQuantizer(Inv,enc,[0],mode='inject',source_errors=q.errors):injected=Inv.layers[0](x)
    assert torch.equal(injected[0],rounded[0]) and not Inv.layers[0]._forward_hooks
