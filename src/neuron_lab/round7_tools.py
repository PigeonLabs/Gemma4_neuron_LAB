"""R7 generic forward journal and explicitly scoped attention-edge interventions.

No candidate or hypothesis selection occurs in this module. The model runtime
and original R6 probe remain unchanged for reproducibility of past studies.
"""
from __future__ import annotations
from contextlib import AbstractContextManager
import hashlib,json
import torch
from .round6_engine import Probe,encode,margin


def token_positions(tokenizer,text,begin,end):
    templ=tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    encoded=tokenizer(templ,add_special_tokens=False,return_offsets_mapping=True)
    a=templ.index(begin)+len(begin);b=templ.index(end,a)
    positions=[i for i,(x,y) in enumerate(encoded['offset_mapping']) if y>a and x<b]
    assert positions
    return positions


class AttentionEdges(AbstractContextManager):
    """Remove only registered query->key edges from actual additive HF masks."""
    def __init__(self,inv,encoded,edges):
        self.inv=inv;self.enc=encoded;self.edges=edges;self.handles=[];self.events=[]
    def __enter__(self):
        for l,layer in self.inv.layers.items():
            active=[e for e in self.edges if l in e.get('layers',range(48))]
            if not active:continue
            def before(module,args,kwargs,l=l,active=active):
                mask=kwargs.get('attention_mask')
                if mask is None or mask.ndim!=4:raise RuntimeError('Attention knockout requires observed full 4D additive mask')
                assert mask.shape[-2:]==(self.enc['tokens'],self.enc['tokens'])
                changed=mask.clone();count=0
                for edge in active:
                    qq=self.enc['regions'][edge['queries']];kk=self.enc['regions'][edge['keys']]
                    if not qq or not kk:raise ValueError('Empty knockout domain')
                    q=torch.tensor(qq,device=mask.device);k=torch.tensor(kk,device=mask.device)
                    old=changed[...,q[:,None],k[None,:]]
                    allowed=old==0;count+=int(allowed.sum().item())
                    changed[...,q[:,None],k[None,:]]=torch.finfo(changed.dtype).min
                assert torch.equal(mask,kwargs['attention_mask'])
                self.events.append({'layer':l,'removed_previously_allowed_edges':count,'out_of_place':changed.data_ptr()!=mask.data_ptr()})
                return args,{**kwargs,'attention_mask':changed}
            self.handles.append(layer.self_attn.register_forward_pre_hook(before,with_kwargs=True))
        return self
    def __exit__(self,*args):
        for h in self.handles:h.remove()
        self.handles=[]


class RoundTripQuantizer(AbstractContextManager):
    """Explicit BF16 activation-roundtrip graph in an FP32 model, not BF16 GEMM."""
    def __init__(self,inv,encoded,layers=(),mode='round',source_errors=None):
        self.inv=inv;self.enc=encoded;self.layers=list(layers);self.mode=mode;self.source_errors=source_errors or {};self.handles=[];self.errors={};self.events=[]
    def __enter__(self):
        for l in self.layers:
            def after(module,args,out,l=l):
                x=out[0] if isinstance(out,tuple) else out
                if x.dtype!=torch.float32:raise RuntimeError('Roundtrip graph requires FP32 state')
                ix=self.enc['regions']['query'];local=x[:,ix];error=local.to(torch.bfloat16).float()-local
                self.errors[l]=error.detach().cpu()
                if self.mode=='capture':return out
                if self.mode=='round':delta=error
                elif self.mode=='inject':delta=self.source_errors[l].to(x.device)
                elif self.mode=='subtract':delta=-self.source_errors[l].to(x.device)
                else:raise ValueError(self.mode)
                if delta.shape!=local.shape:raise ValueError('Unaligned roundtrip residual')
                y=x.clone();y[:,ix]=local+delta
                self.events.append({'layer':l,'mode':self.mode,'delta_norm':float(delta.norm().item()),'scope':'query'})
                return (y,*out[1:]) if isinstance(out,tuple) else y
            self.handles.append(self.inv.layers[l].register_forward_hook(after))
        return self
    def __exit__(self,*args):
        for h in self.handles:h.remove()
        self.handles=[]


class Journal:
    def __init__(self,probe,cfg):
        self.p=probe;self.cfg=cfg;self.rows=[];self.encodings={};self.file=(probe.run.path/'rows.jsonl').open('x')
        for l,layer in probe.inv.layers.items():probe.inv.sites[f'resid:{l}']=(layer,False)
    def enc(self,case):
        text=case['text']
        if text not in self.encodings:
            enc=encode(self.p.runtime.tokenizer,text,self.cfg['max_input_tokens']);enc['regions']['last']=[enc['tokens']-1]
            for name,(begin,end) in case.get('sections',{}).items():
                enc['regions'][name]=token_positions(self.p.runtime.tokenizer,text,begin,end)
                enc['regions']['after_'+name]=list(range(max(enc['regions'][name])+1,enc['tokens']))
            if 'table_tail' in enc['regions']:enc['regions']['after_tables']=list(range(max(enc['regions']['table_tail'])+1,enc['tokens']))
            self.encodings[text]=enc
        return self.encodings[text]
    def call(self,case,arm='clean',specs=(),capture=(),edges=(),quantizer=None,phase='P0',regions=('query',)):
        enc=self.enc(case)
        quantizer=quantizer or {}
        with AttentionEdges(self.p.inv,enc,edges) as edge_hooks,RoundTripQuantizer(self.p.inv,enc,**quantizer) as quant:
            row,h,logits=self.p.forward(enc,specs=specs,capture=capture,regions=regions)
        expected=case['expected'];assert expected in 'ABCD'
        unique=case.get('has_unique_target_answer',True)
        row={**row,'id':case['id'],'episode_id':case['episode_id'],'study':case['study'],'phase':phase,'arm':arm,
             'expected':expected,'has_unique_target_answer':unique,'correct':row['prediction']==expected if unique else None,
             'correct_margin':margin(row,expected) if unique else None,'reference_margin':margin(row,expected),
             'text_sha256':enc['text_sha256'],'factors':case.get('factors',{}),'oracle':case.get('oracle',{}),
             'edge_delivery':edge_hooks.events,'roundtrip_delivery':quant.events}
        self.rows.append(row);self.file.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');self.file.flush()
        return row,h,quant
    def close(self):self.file.close()
