"""Three predeclared causal models; mechanisms may coexist, no recovery authority."""
import torch
from torch import nn
from .sequence_models import RelativeAttentionBlock
from .multifault_20261026_v1 import GLOBAL_WIDTH,CELLS,CELL_WIDTH,BASE_WIDTH

ARMS=('global_temporal','mechanism_temporal','spatial_temporal')
class MultiFaultModel(nn.Module):
    def __init__(self,arm):
        super().__init__();assert arm in ARMS;self.arm=arm
        self.spatial=nn.Sequential(nn.Linear(CELL_WIDTH,16),nn.GELU(),nn.Linear(16,16),nn.GELU()) if arm=='spatial_temporal' else None
        self.projection=nn.Linear(GLOBAL_WIDTH+(32 if self.spatial is not None else CELLS*CELL_WIDTH),32)
        self.blocks=nn.ModuleList([RelativeAttentionBlock(32,4,64) for _ in range(2)])
        self.cause=nn.Linear(32,8);self.transition=nn.Linear(32,2);self.parameter=nn.Linear(32,2)
    def forward(self,g,c,state=None):
        if self.arm=='global_temporal':
            g=torch.cat([g[...,:BASE_WIDTH],torch.zeros_like(g[...,BASE_WIDTH:])],-1);sp=torch.zeros((*g.shape[:-1],CELLS*CELL_WIDTH),device=g.device,dtype=g.dtype)
        elif self.spatial is None:sp=c.flatten(-2)
        else:
            z=self.spatial(c);sp=torch.cat([z.amax(-2),z.mean(-2)],-1)
        z=self.projection(torch.cat([g,sp],-1));prior=state if state is not None else [None,None];new=[]
        for block,past in zip(self.blocks,prior):z,cache=block(z,past);new.append(cache)
        return (self.cause(z),self.transition(z),self.parameter(z)),new

def predict(model,g,c,chunk=64):
    model.eval();state=None;outputs=[[],[],[]]
    with torch.no_grad():
        for i in range(0,g.shape[1],chunk):
            y,state=model(g[:,i:i+chunk],c[:,i:i+chunk],state)
            for target,v in zip(outputs,y):target.append(v)
    return [torch.cat(v,1) for v in outputs]
