"""Out-of-place native down-projection-input interventions for R5."""
from __future__ import annotations
from contextlib import AbstractContextManager
import torch


class R5Hooks(AbstractContextManager):
    def __init__(self, modules, query_start, groups=(), capture_layers=(), restore=False):
        self.modules=modules
        self.query_start=query_start
        self.groups=list(groups)
        self.capture_layers=set(capture_layers)
        self.restore=restore
        self.handles=[]
        self.means={}
        self.events=[]
        self.delivery=[]

    def __enter__(self):
        layers=self.capture_layers | {g["layer"] for g in self.groups}
        for layer in sorted(layers):
            def hook(module,args,layer=layer):
                x=args[0]
                self.events.append(layer)
                if layer in self.capture_layers:
                    self.means[layer]=x[:,self.query_start:,:].abs().float().mean(dim=(0,1)).cpu()
                local=[g for g in self.groups if g["layer"]==layer]
                if not local:
                    return None
                y=x.clone()
                for g in local:
                    start=0 if g.get("scope","all")=="all" else self.query_start
                    if g.get("scope","all") not in {"all","query"}:
                        raise ValueError("Unknown intervention scope")
                    ix=g["coordinates"]
                    amount=g.get("attenuation",1.)
                    if not 0<=amount<=1:
                        raise ValueError("Attenuation outside [0,1]")
                    y[:,start:,ix]=y[:,start:,ix]*(1-amount)
                    self.delivery.append({"layer":layer,"scope":g.get("scope","all"),
                        "positions":x.shape[1]-start,"coordinates":len(ix),
                        "zero_delivered":bool(torch.count_nonzero(y[:,start:,ix]).item()==0) if amount==1 else None,
                        "prefix_unchanged":bool(torch.equal(x[:,:start,:],y[:,:start,:])) if start else True})
                if self.restore:
                    # Test-only: restore in this very forward, not a new clean pass.
                    y=x.clone()
                return (y,*args[1:])
            self.handles.append(self.modules[layer].register_forward_pre_hook(hook))
        return self

    def __exit__(self,*exc):
        for handle in self.handles:
            handle.remove()
        self.handles=[]
