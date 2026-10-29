"""Counted, out-of-place forward-only interventions for the R6 experiments."""
from __future__ import annotations
from contextlib import AbstractContextManager
import hashlib
import json
import re
import time
import numpy as np
import torch
from .runtime import ROOT,save_json
from .round5_execution import encode as query_encode


def encode(tokenizer,text,maximum=1000):
    out=query_encode(tokenizer,text,maximum)
    templ=tokenizer.apply_chat_template([{"role":"user","content":text}],tokenize=False,
        add_generation_prompt=True,enable_thinking=False)
    offsets=tokenizer(templ,add_special_tokens=False,return_offsets_mapping=True)["offset_mapping"]
    intervals={"cue":(templ.index("CUE_BEGIN:\n")+len("CUE_BEGIN:\n"),templ.index("\nFACTS_BEGIN:")),
        "facts":(templ.index("FACTS_BEGIN:\n")+len("FACTS_BEGIN:\n"),templ.index("\nEND_FACTS"))}
    out["regions"]={k:[i for i,(a,b) in enumerate(offsets) if b>start and a<end] for k,(start,end) in intervals.items()}
    out["regions"]["query"]=list(range(out["query_start"],out["tokens"]))
    out["regions"]["all"]=list(range(out["tokens"]))
    if any(not v for v in out["regions"].values()):raise ValueError("Empty semantic token scope")
    return out


class Inventory:
    def __init__(self,runtime):
        self.runtime=runtime;allmods=dict(runtime.model.named_modules());self.sites={};self.layers={};self.norms={}
        for l,name in runtime.module_names.items():
            prefix=name[:-len(".mlp.down_proj")];layer=allmods[prefix];self.layers[l]=layer
            for k,m,pre in (("mlp",layer.mlp,False),("attn",layer.self_attn,False),
                            ("native_mlp",layer.mlp.down_proj,True),("native_attn",layer.self_attn.o_proj,True),
                            ("post_mlp",layer.post_feedforward_layernorm,False),("post_attn",layer.post_attention_layernorm,False)):
                self.sites[f"{k}:{l}"]=(m,pre)
        text_prefix=runtime.module_names[0].split(".layers.")[0]
        for name,m in allmods.items():
            if name.startswith(text_prefix) and type(m).__name__=="Gemma4UnifiedRMSNorm":
                match=re.search(r"\.layers\.(\d+)\.",name)
                self.norms[name]=(m,int(match.group(1)) if match else 999)

    def projection(self,key):
        return self.sites[key][0]


def key_for(kind,layer):return f"{kind}:{layer}"


class Hooks(AbstractContextManager):
    def __init__(self,inventory,encoded,specs=(),capture=(),regions=("query",),norm_capture_layer=None,norm_reference=None):
        self.inv=inventory;self.enc=encoded;self.specs=list(specs);self.capture=set(capture);self.regions=regions
        self.norm_capture_layer=norm_capture_layer;self.norm_reference=norm_reference
        self.cache={};self.denominators={};self.events=[];self.delivery=[];self.handles=[]

    def __enter__(self):
        keys=self.capture|{s["key"] for s in self.specs}
        for key in sorted(keys):
            module,pre=self.inv.sites[key]
            def change(x,key=key):
                self.events.append(key)
                if key in self.capture:
                    for region in self.regions:
                        ix=self.enc["regions"][region]
                        self.cache[(key,region)]=x[:,ix,:].float().cpu().clone()
                relevant=[s for s in self.specs if s["key"]==key]
                if not relevant:return x
                y=x.clone()
                for s in relevant:
                    ix=self.enc["regions"][s.get("scope","query")]
                    coordinates=s.get("coordinates")
                    cols=list(range(x.shape[-1])) if coordinates is None else coordinates
                    if not cols or min(cols)<0 or max(cols)>=x.shape[-1]:raise ValueError("Invalid native coordinate set")
                    amount=s.get("attenuation",1.)
                    if not 0<=amount<=1:raise ValueError("Invalid attenuation")
                    original=y[:,ix,:][:,:,cols]
                    if s.get("mode","patch")=="zero":target=torch.zeros_like(original)
                    elif s.get("mode")=="identity":target=original
                    else:
                        source=s["source"][(key,s.get("source_scope",s.get("scope","query")))]
                        if source.shape[1]!=len(ix):raise ValueError(f"Unaligned source region {key}:{len(ix)}/{source.shape[1]}")
                        target=source[:,:,cols].to(x.device,dtype=x.dtype)
                    # Advanced-index temporary slices do not mutate the source.
                    vals=original+amount*(target-original)
                    y[:,torch.tensor(ix,device=x.device)[:,None],torch.tensor(cols,device=x.device)[None,:]]=vals
                    if s.get("mode")=="zero" and amount==1 and torch.count_nonzero(y[:,ix,:][:,:,cols]).item():
                        raise RuntimeError("Zero intervention was not delivered")
                    if s.get("scope","query")=="query" and not torch.equal(y[:,:self.enc["query_start"]],x[:,:self.enc["query_start"]]):
                        # A preceding persistent lesion may have already changed
                        # the prefix in a joint arm; compare only new local edits.
                        if len(relevant)==1:raise RuntimeError("Query intervention changed prefix")
                    self.delivery.append({"key":key,"mode":s.get("mode","patch"),"scope":s.get("scope","query"),
                        "positions":len(ix),"coordinates":len(cols),"delta_norm":float((vals-original).float().norm().item())})
                return y
            if pre:
                def ph(module,args,change=change):return (change(args[0]),*args[1:])
                self.handles.append(module.register_forward_pre_hook(ph))
            else:
                def oh(module,args,out,change=change):
                    if isinstance(out,tuple):return (change(out[0]),*out[1:])
                    return change(out)
                self.handles.append(module.register_forward_hook(oh))
        if self.norm_capture_layer is not None or self.norm_reference is not None:
            for name,(m,layer) in self.inv.norms.items():
                if self.norm_capture_layer is not None and layer<self.norm_capture_layer:continue
                if self.norm_reference is not None and name not in self.norm_reference:continue
                def nh(module,args,out,name=name):
                    x=args[0];ix=self.enc["regions"]["query"]
                    if x.shape[1]!=self.enc["tokens"]:raise RuntimeError("Unexpected norm sequence axis")
                    v=x[:,ix].float();inv=torch.pow(v.pow(2).mean(-1,keepdim=True)+module.eps,-.5)
                    if self.norm_capture_layer is not None:self.denominators[name]=inv.cpu().clone()
                    if self.norm_reference is None:return out
                    ref=self.norm_reference[name].to(x.device)
                    if ref.shape!=inv.shape:raise RuntimeError("Mismatched norm-replay reference")
                    z=v*ref
                    if module.with_scale:z=z*module.weight.float()
                    y=out.clone();y[:,ix]=z.to(out.dtype)
                    return y
                self.handles.append(m.register_forward_hook(nh))
        return self

    def __exit__(self,*args):
        for h in self.handles:h.remove()
        self.handles=[]


class Probe:
    def __init__(self,runtime,run,cfg,cap=None,global_state=None):
        self.runtime=runtime;self.run=run;self.cfg=cfg;self.inv=Inventory(runtime)
        self.calls=0;self.completed=0;self.cap=cap or cfg["max_actual_forward_calls"]
        self.started=time.monotonic();self.armed=False;self.raw=None;self.global_state=global_state
        self.ids=[runtime.tokenizer.encode(c,add_special_tokens=False) for c in "ABCD"]
        assert all(len(i)==1 for i in self.ids);self.ids=[i[0] for i in self.ids]
        self.handle=runtime.model.register_forward_pre_hook(self._before,with_kwargs=True)
        self.head_handle=runtime.model.get_output_embeddings().register_forward_hook(self._head)
        self.versions={n:p._version for n,p in runtime.model.named_parameters()}
        self.first50=None;self.local_evaluations=0

    def _head(self,module,args,out):self.raw=out[:,-1,:].detach()

    def _before(self,module,args,kwargs):
        if not self.armed or kwargs.get("use_cache") is not False:raise RuntimeError("Uncounted or cached forward")
        if self.calls>=self.cap:raise RuntimeError("STUDY_FORWARD_BUDGET")
        if self.global_state:
            if self.global_state["calls"]+self.calls>=self.cfg["max_actual_forward_calls"]:raise RuntimeError("GLOBAL_FORWARD_BUDGET")
            if time.time()-self.global_state["started_unix"]>self.cfg["max_seconds"]:raise TimeoutError("GLOBAL_TIME_BUDGET")
        self.run.check_budget();self.calls+=1

    def check_frozen(self):
        if any(p.requires_grad for p in self.runtime.model.parameters()):raise RuntimeError("Unfrozen parameter")
        if self.versions!={n:p._version for n,p in self.runtime.model.named_parameters()}:raise RuntimeError("Parameter changed")
        self.runtime.verify_sources()

    @torch.inference_mode()
    def forward(self,enc,specs=(),capture=(),regions=("query",),norm_capture_layer=None,norm_reference=None,full=False):
        self.armed=True;self.raw=None
        try:
            tensors={k:torch.tensor([enc[k]],device=self.runtime.cfg["device"]) for k in ("input_ids","attention_mask")}
            with Hooks(self.inv,enc,specs,capture,regions,norm_capture_layer,norm_reference) as hooks:
                logits=self.runtime.model(**tensors,use_cache=False,logits_to_keep=1).logits[:,-1,:]
            if self.raw is None or not torch.isfinite(logits).all():raise RuntimeError("Missing or nonfinite readout")
            soft=self.runtime.model.config.get_text_config().final_logit_softcapping
            if not torch.equal(logits,torch.tanh(self.raw/soft)*soft):raise RuntimeError("Tied-head/softcap parity failed")
            four=logits[0,self.ids].float();raw=self.raw[0,self.ids].float()
            top=int(logits[0].argmax().item())
            row={"prediction":"ABCD"[int(four.argmax())],"post_softcap_logits":four.cpu().tolist(),
                "raw_head_logits":raw.cpu().tolist(),"full_top1_id":top,"full_top1_text":self.runtime.tokenizer.decode([top]),
                "full_top1_code":"ABCD"[self.ids.index(top)] if top in self.ids else None,
                "tokens":enc["tokens"],"query_start":enc["query_start"],"delivery":hooks.delivery}
            self.completed+=1
            prior=self.global_state["calls"] if self.global_state else 0
            if self.first50 is None and (self.completed==50 or prior+self.completed==50):
                torch.cuda.synchronize();self.first50={"completed_calls_in_run":self.completed,"completed_including_prior_runs":prior+self.completed,
                    "seconds_in_run":time.monotonic()-self.started,"last_tokens":enc["tokens"]}
                save_json(self.run.path/"first50.json",self.first50)
                print(json.dumps({"event":"first_50_completed",**self.first50}),flush=True)
            return row,hooks,logits.detach().clone() if full else None
        finally:self.armed=False;self.raw=None

    def close(self):self.handle.remove();self.head_handle.remove()


def margin(row,code):
    x=np.asarray(row["post_softcap_logits"],dtype=float);i="ABCD".index(code);o=np.delete(x,i);m=o.max()
    return float(x[i]-m-np.log(np.exp(o-m).sum()))


def pair_score(row,code1,code2):
    x=row["post_softcap_logits"];return float(x["ABCD".index(code1)]-x["ABCD".index(code2)])


def bootstrap(values,cfg):
    x=np.asarray(values,dtype=float);assert len(x) and np.isfinite(x).all()
    rng=np.random.default_rng(cfg["bootstrap_seed"])
    means=x[rng.integers(0,len(x),(cfg["bootstrap_resamples"],len(x)))].mean(1)
    signs=rng.choice([-1.,1.],size=(cfg["bootstrap_resamples"],len(x)))
    p=float((1+np.sum((signs*x).mean(1)>=x.mean()))/(len(signs)+1))
    return {"point":float(x.mean()),"ci95":np.quantile(means,[.025,.975]).tolist(),
            "episodes":len(x),"one_sided_signflip_p":p,"unit":"episode"}


def projection_sizes(inv,kind,layer):
    key=f"native_{kind}:{layer}";module=inv.projection(key)
    width=module.in_features
    if kind=="mlp":return width,[128,512]
    heads=inv.layers[layer].self_attn.config.num_attention_heads
    return width,[width//heads,2*(width//heads)]


def native_candidates(inv,kind,layer,absolute,coherent):
    key=f"native_{kind}:{layer}";m=inv.projection(key)
    weight=m.weight.float().norm(dim=0).cpu().numpy()
    scores={"absolute":np.asarray(absolute)*weight,"coherent":np.abs(np.asarray(coherent))*weight}
    width,sizes=projection_sizes(inv,kind,layer);groups=[]
    for rank,score in scores.items():
        if kind=="attn":
            hd=width//inv.layers[layer].self_attn.config.num_attention_heads
            order=np.lexsort((np.arange(width//hd),-score.reshape(-1,hd).sum(1)))
        else:order=np.lexsort((np.arange(width),-score))
        for size in sizes:
            coords=(np.concatenate([np.arange(h*hd,(h+1)*hd) for h in order[:size//hd]]).tolist() if kind=="attn" else order[:size].tolist())
            groups.append({"id":f"{kind}_L{layer}_K{size}_{rank}","kind":kind,"layer":layer,"key":key,
                "coordinates":coords,"scope":"query","unit":"native_MLP_down_proj_input" if kind=="mlp" else "attention_o_proj_input_head_slices"})
    return groups


def random_panel(group,inv,cfg):
    width=inv.projection(group["key"]).in_features;forbidden=set(group["coordinates"]);available=[i for i in range(width) if i not in forbidden]
    panel=[]
    for i,seed in enumerate(cfg["random_seeds"]):
        rng=np.random.default_rng(seed+group["layer"])
        if group["kind"]=="attn":
            hd=width//inv.layers[group["layer"]].self_attn.config.num_attention_heads
            heads=[h for h in range(width//hd) if not forbidden.intersection(range(h*hd,(h+1)*hd))]
            chosen=rng.choice(heads,len(group["coordinates"])//hd,replace=False)
            coords=np.concatenate([np.arange(h*hd,(h+1)*hd) for h in chosen]).tolist()
        else:coords=rng.choice(available,len(group["coordinates"]),replace=False).tolist()
        panel.append({**group,"id":f"random{i+1}","coordinates":coords,"attenuation":1.})
    return panel
