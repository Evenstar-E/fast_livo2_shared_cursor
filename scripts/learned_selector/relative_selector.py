"""Task-unit features and bounded causal context; no dataset-fit normalization."""
from collections import deque
import numpy as np
import torch
from torch import nn
from .telemetry_features import FEATURES

NAMES=('scale_evidence','log_sigma','fit_gain','held_gain','coverage','information',
       'nominal_shift','position_update','rotation_update','velocity_update',
       'gyro_rate','gyro_variation','acc_variation')


def physical_event(event, units):
    x=dict(zip(FEATURES,event['x'])); m=dict(zip(FEATURES,event['mask']))
    values=[]; valid=[]
    def add(names,fn):
        ok=all(m[n] and np.isfinite(x[n]) for n in names)
        v=float(fn()) if ok else 0.
        ok=ok and np.isfinite(v)
        values.append(float(np.clip(v,-units['clip'],units['clip'])) if ok else 0.)
        valid.append(bool(ok))
    s=units['sigma_q_floor']; r=units['residual_unit_m']
    add(('q_raw','sigma'),lambda:np.arcsinh((x['q_raw']-1)/max(s,x['sigma'])))
    add(('sigma',),lambda:np.log1p(max(0.,x['sigma'])/s))
    add(('fit','null_fit'),lambda:np.log((max(0.,x['null_fit'])+r)/(max(0.,x['fit'])+r)))
    add(('held','null_held'),lambda:np.log((max(0.,x['null_held'])+r)/(max(0.,x['held'])+r)))
    for name in ('coverage','information'):add((name,),lambda n=name:x[n])
    for name,unit in [('nominal_fit_position_shift_m','position_unit_m'),('native_position_update_m','position_unit_m'),
                      ('native_rotation_update_rad','rotation_unit_rad'),('native_velocity_update_mps','velocity_unit_mps'),
                      ('gyro_mean_norm','gyro_unit_radps'),('gyro_std_norm','gyro_unit_radps'),('acc_std_norm','acc_std_unit_mps2')]:
        add((name,),lambda n=name,u=unit:np.log1p(max(0.,x[n])/units[u]))
    return np.array(values,dtype=np.float32),np.array(valid,dtype=bool)


def stream(events,config):
    """Transforms t using only actual input at <=t; context excludes current event."""
    history=deque(maxlen=config['past_context_events']);prev=-np.inf
    for e in events:
        if e['available_time']<=prev:raise ValueError('event order')
        prev=e['available_time'];v,m=physical_event(e,config['physical_transforms'])
        baseline=np.zeros(len(NAMES),dtype=np.float32); past=np.zeros(len(NAMES),dtype=bool)
        if history:
            h=np.array([x[0] for x in history]);hm=np.array([x[1] for x in history])
            for j in range(len(NAMES)):
                observed=h[hm[:,j],j]
                if len(observed)>=config['minimum_context_valid']:
                    baseline[j]=np.median(observed);past[j]=True
        delta=np.where(m&past,v-baseline,0)
        row=np.concatenate((v,delta,m.astype(np.float32),past.astype(np.float32),[1.])).astype(np.float32)
        # Preserve prior all-source-fields eligibility, add explicit context warm-up.
        eligible=bool(all(e['mask']) and m.all() and past.all())
        yield row,eligible
        history.append((v,m))


def windows(events,config):
    n=config['window'];h=deque(maxlen=n);result=[];eligible=[]
    for row,ok in stream(events,config):
        h.append(row); w=np.zeros((n,len(row)),dtype=np.float32);w[-len(h):]=h
        result.append(w);eligible.append(ok)
    return np.array(result),np.array(eligible,dtype=bool)


def training_pairs(clean,fault):
    """Only use same-bag experiment pairs, exact raw-header alignment; no interpolation."""
    lookup={e['index']:i for i,e in enumerate(clean['events'])}
    pairs=[]
    for i,(e,label) in enumerate(zip(fault['events'],fault['labels'])):
        if not label['sensor_fault_active']:continue
        j=lookup.get(e['index'])
        if j is None:raise ValueError('unmatched pair')
        if abs(e['header_time']-clean['events'][j]['header_time'])>1e-9:raise ValueError('pair time mismatch')
        if clean['labels'][j]['sensor_fault_active']:raise ValueError('not clean pair')
        pairs.append((j,i))
    return pairs


class RelativeSelector(nn.Module):
    def __init__(self,kind,hidden=16):
        super().__init__();self.kind=kind;d=4*len(NAMES)+1
        if kind=='relative_linear':self.head=nn.Linear(d+2*len(NAMES),1)
        else:self.encoder=nn.GRU(d,hidden,batch_first=True);self.head=nn.Linear(hidden,1)

    def forward(self,x):
        if self.kind=='relative_linear':
            h=torch.cat((x[:,-1],x[:,-8:,:2*len(NAMES)].mean(1)),dim=1)
        else:
            # Bounded-window inference resets hidden; do not persist indefinitely online.
            _,h=self.encoder(x);h=h[0]
        return self.head(h)[:,0]
