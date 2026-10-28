"""Small causal sequence models; recurrent state is reset only at bag boundaries.

Memory attention is Transformer-XL-inspired, not a full reproduction: cached
layer inputs + detached segment recurrence + learned relative-distance bias.
"""
import math
import torch
from torch import nn

class RelativeAttentionBlock(nn.Module):
    def __init__(self,width,heads,memory):
        super().__init__();self.width=width;self.heads=heads;self.memory=memory;self.dk=width//heads
        self.norm1=nn.LayerNorm(width);self.norm2=nn.LayerNorm(width)
        self.q=nn.Linear(width,width);self.k=nn.Linear(width,width);self.v=nn.Linear(width,width);self.o=nn.Linear(width,width)
        self.distance=nn.Embedding(129,heads)
        self.ff=nn.Sequential(nn.Linear(width,width*2),nn.GELU(),nn.Linear(width*2,width))
    def forward(self,x,past=None):
        # x: batch,current-time,width; past stores pre-layer activations only.
        m=0 if past is None else past.shape[1]
        history=x if past is None else torch.cat((past,x),1)
        current=self.norm1(x);all_values=self.norm1(history);b,t,d=x.shape;s=history.shape[1]
        q=self.q(current).reshape(b,t,self.heads,self.dk).transpose(1,2)
        k=self.k(all_values).reshape(b,s,self.heads,self.dk).transpose(1,2)
        v=self.v(all_values).reshape(b,s,self.heads,self.dk).transpose(1,2)
        distance=torch.arange(m,m+t,device=x.device)[:,None]-torch.arange(s,device=x.device)[None,:]
        allowed=distance>=0
        if self.memory:allowed=allowed&(distance<=self.memory)
        bias=self.distance(distance.clamp(0,128)).permute(2,0,1)
        logits=(q@k.transpose(-1,-2))/math.sqrt(self.dk)+bias[None]
        weights=logits.masked_fill(~allowed[None,None],float('-inf')).softmax(-1)
        z=(weights@v).transpose(1,2).reshape(b,t,d);y=x+self.o(z);y=y+self.ff(self.norm2(y))
        cache=history[:,-self.memory:].detach() if self.memory else None
        return y,cache

class BagSequenceModel(nn.Module):
    def __init__(self,kind,config):
        super().__init__();self.kind=kind;width=config['hidden'];self.projection=nn.Linear(34,width)
        if kind=='gru_sequence':self.encoder=nn.GRU(width,width,batch_first=True)
        else:
            memory=config['memory_events'] if kind=='memory_transformer' else 0
            self.blocks=nn.ModuleList([RelativeAttentionBlock(width,config['heads'],memory) for _ in range(config['layers'])])
        self.head=nn.Linear(width,1)
    def forward(self,x,state=None):
        z=self.projection(x)
        if self.kind=='gru_sequence':z,state=self.encoder(z,state);state=state.detach()
        else:
            prior=state if state is not None else [None]*len(self.blocks);state=[]
            for block,past in zip(self.blocks,prior):z,cache=block(z,past);state.append(cache)
        return self.head(z)[:,:,0],state

def predict(model,x,chunk=64):
    state=None;values=[]
    with torch.no_grad():
        for start in range(0,len(x),chunk):
            score,state=model(x[None,start:start+chunk],state);values.append(score[0])
    return torch.cat(values).sigmoid().cpu().numpy()
