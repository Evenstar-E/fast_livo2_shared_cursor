"""Explicit synthetic LiDAR mechanisms; no labels enter the feature extractor."""
import copy
from collections import deque
import math
import numpy as np

KINDS=('scale','bias','noise','sector','beam','echo_loss','spurious','extrinsic')
LEVELS={'scale':[.03,.06,.10],'bias':[.05,.15,.30],'noise':[.02,.05,.10],
        'sector':[20,45,90],'beam':[2,4,8],'echo_loss':[.2,.4,.6],'spurious':[.01,.03,.06],'extrinsic':[1,3,5]}
BASE_WIDTH=24
GLOBAL_WIDTH=24+32+1+24+9
CELL_WIDTH=10
CELLS=48
FORMATS={1:'i1',2:'u1',3:'i2',4:'u2',5:'i4',6:'u4',7:'f4',8:'f8'}
EXT=np.array([[0,-1,0],[-1,0,0],[0,0,-1]],float)

def decode(msg):
    f={x.name:x for x in msg.fields};names=['x','y','z','intensity','ring']
    if any(n not in f or f[n].count!=1 for n in names):raise ValueError('XYZ/intensity/ring scalar fields required')
    dtype=np.dtype(dict(names=names,formats=[('>' if msg.is_bigendian else '<')+FORMATS[f[n].datatype] for n in names],offsets=[f[n].offset for n in names],itemsize=msg.point_step))
    a=np.ndarray((msg.height,msg.width),dtype=dtype,buffer=msg.data,strides=(msg.row_step,msg.point_step))
    return np.column_stack([a[n].reshape(-1) for n in 'xyz']).astype(np.float32),a['intensity'].reshape(-1).astype(np.float32),a['ring'].reshape(-1).astype(np.int64)

def corrupt(xyz,ring,kind,level,seed,sign=1,azimuth=0.):
    p=np.array(xyz,dtype=np.float32,copy=True);ids=np.arange(len(p));rng=np.random.default_rng(seed)
    if kind=='clean' or level==0:return p,ids,dict(changed=0,dropped=0,clamped=0)
    if kind not in KINDS:raise ValueError(kind)
    r=np.linalg.norm(p.astype(float),axis=1);valid=np.isfinite(p).all(1)&(r>0)
    nr=r.copy();keep=np.ones(len(p),bool);clamp=0
    if kind=='scale':p[valid]=(p[valid].astype(float)*(1+sign*level)).astype(np.float32)
    elif kind in ('bias','noise','spurious'):
        if kind=='bias':nr[valid]+=sign*level
        elif kind=='noise':nr[valid]+=rng.normal(0,level,valid.sum())
        else:
            chosen=np.flatnonzero(valid);chosen=chosen[rng.random(len(chosen))<level];nr[chosen]*=rng.uniform(.15,.85,len(chosen))
        clamp=int(np.sum(valid&(nr<.05)));nr=np.maximum(.05,nr)
        p[valid]=(p[valid].astype(float)*(nr[valid]/r[valid])[:,None]).astype(np.float32)
    elif kind=='sector':
        angle=np.arctan2(p[:,1],p[:,0]);distance=np.arctan2(np.sin(angle-azimuth),np.cos(angle-azimuth))
        keep &= ~(valid&(np.abs(distance)<=np.deg2rad(level)/2))
    elif kind=='beam':
        bad=rng.choice(np.arange(32),int(level),replace=False);keep &= ~np.isin(ring,bad)
    elif kind=='echo_loss':keep &= ~(valid&(rng.random(len(p))<level))
    elif kind=='extrinsic':
        a=np.deg2rad(sign*level);rotation=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1.]])
        shift=sign*{1:.02,3:.05,5:.10}[int(level)]
        p[valid]=(p[valid].astype(float)@rotation.T+np.array([shift,0,0])).astype(np.float32)
    changed=int(np.sum(~keep))+int(np.sum(keep&valid&np.any(p!=xyz,axis=1)))
    return p[keep],ids[keep],dict(changed=changed,dropped=int(np.sum(~keep)),clamped=clamp)

def corrupt_message(msg,kind,level,seed,sign=1,azimuth=0.):
    """Same transform, preserving complete retained records and message fields."""
    xyz,_,ring=decode(msg);p,ids,audit=corrupt(xyz,ring,kind,level,seed,sign,azimuth)
    if not audit['changed']:return copy.copy(msg),audit
    offsets={f.name:f.offset for f in msg.fields};data=np.frombuffer(msg.data,dtype=np.uint8)
    if any(f.datatype!=7 for f in msg.fields if f.name in ('x','y','z')):raise ValueError('message adapter requires float32 XYZ')
    rows=np.concatenate([data[y*msg.row_step:y*msg.row_step+msg.width*msg.point_step].reshape(msg.width,msg.point_step) for y in range(msg.height)])
    payload=bytearray(rows[ids].copy().tobytes());endian='>' if msg.is_bigendian else '<'
    for j,n in enumerate('xyz') if len(ids) else []:
        a=np.ndarray((len(ids),),dtype=endian+'f4',buffer=payload,offset=offsets[n],strides=(msg.point_step,));a[:]=p[:,j]
    m=copy.copy(msg);m.height=1;m.width=len(ids);m.row_step=len(ids)*m.point_step;m.data=bytes(payload);m.is_dense=bool(np.isfinite(p).all())
    return m,audit

def schedules(seq_seed):
    out={'clean':[]}
    for k,kind in enumerate(KINDS):
        for version in range(2):
            rng=np.random.default_rng(seq_seed+100*k+version);episodes=[]
            for j,center in enumerate((22,47,77,107)):
                start=center+float(rng.uniform(-3,3));duration=float(rng.uniform(5,8));level=LEVELS[kind][int(rng.integers(3))]
                episodes.append(dict(start=start,end=start+duration,kind=kind,level=level,sign=int(rng.choice([-1,1])),azimuth=float(rng.uniform(-np.pi,np.pi)),seed=seq_seed+10000*k+100*version+j,episode=j))
            out[kind+'_'+str(version)]=episodes
    # Composition challenge is excluded from training and normalization.
    out['mixed']=[dict(start=35.,end=43.,kind=k,level=v,sign=-1,azimuth=.7,seed=seq_seed+99000+j,episode=0) for j,(k,v) in enumerate([('scale',.06),('sector',45)])]
    return out

def rotate(p,omega,dt):
    v=-np.asarray(omega)*dt;angle=np.linalg.norm(v)
    if angle<1e-12:return p
    axis=v/angle;k=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    return p@(np.eye(3)+np.sin(angle)*k+(1-np.cos(angle))*(k@k)).T

def cells(p,intensity,total):
    r=np.linalg.norm(p.astype(float),axis=1);az=np.arctan2(p[:,1],p[:,0]);el=np.arcsin(np.clip(p[:,2]/r,-1,1))
    ia=np.minimum(11,((az+np.pi)/(2*np.pi)*12).astype(int));ie=np.searchsorted([-.14,0,.14],el);bins=ie*12+ia
    table=np.zeros((48,10),np.float32)
    for i in range(48):
        angle=-np.pi+(i%12+.5)*np.pi/6;elevation=(-.25,-.07,.07,.25)[i//12];q=bins==i
        table[i,6:9]=[np.cos(angle),np.sin(angle),np.sin(elevation)]
        if not q.any():continue
        table[i,:6]=[np.log1p(q.sum()*total/max(1,len(p))),*np.quantile(r[q],[.25,.5,.75]),np.median(intensity[q]),np.std(r[q])];table[i,9]=1
    return table

class Features:
    """No schedule, label, clean counterpart, absolute time or sequence input."""
    def __init__(self):self.previous=None;self.history=deque(maxlen=20)
    def update(self,xyz,intensity,ring,gyro,accel,dt,imu_valid):
        valid=np.isfinite(xyz).all(1)&(np.linalg.norm(xyz.astype(float),axis=1)>.05)&np.isfinite(intensity)
        p=xyz[valid];it=intensity[valid];rg=ring[valid];n=len(p)
        if n:
            ix=np.linspace(0,n-1,min(4096,n),dtype=int);p=p[ix].astype(float);it=it[ix];rg=rg[ix]
        if n<8:
            base=np.zeros(24);table=np.zeros((48,10),np.float32);ring_hist=np.zeros(32)
        else:
            ranges=np.linalg.norm(p,axis=1);eigs=np.linalg.eigvalsh(np.cov(p.T));eigs=eigs/max(float(eigs.sum()),1e-12)
            base=np.array([np.log1p(n),*np.quantile(ranges,[.05,.25,.5,.75,.95]),*np.quantile(it,[.1,.5,.9]),*eigs,*gyro,*accel,dt,float(imu_valid),*p.mean(0),ranges.std()])
            table=cells(p,it,n);ring_hist=np.bincount(np.clip(rg,0,31),minlength=32)/len(rg)
        gap=not 0<dt<=.2
        if gap:self.previous=None;self.history.clear()
        delta=base-np.median(self.history,axis=0) if self.history else np.zeros(24)
        motion=np.zeros(9)
        if self.previous is not None and n>=8 and imu_valid and not gap:
            oldp,oldit,oldn=self.previous;prior=cells(rotate(oldp,gyro,dt),oldit,oldn)
            common=(table[:,9]>0)&(prior[:,9]>0);cnt=int(common.sum());motion[7]=cnt/48
            if cnt>=8:
                v=table[common];o=prior[common];u=np.column_stack([o[:,6]*np.sqrt(1-o[:,8]**2),o[:,7]*np.sqrt(1-o[:,8]**2),o[:,8]])
                design=np.column_stack([o[:,2],np.ones(cnt),u]);target=v[:,2]-o[:,2]
                fit,_,rank,s=np.linalg.lstsq(design,target,rcond=1e-4)
                motion[:5]=fit;motion[5]=np.sqrt(np.mean((design@fit-target)**2));motion[6]=s[-1]/max(s[0],1e-12);motion[8]=float(rank==5)
        if n>=8:self.previous=(p.copy(),it.copy(),n)
        else:self.previous=None
        self.history.append(base.copy())
        g=np.concatenate([base,ring_hist,[float(n>=8)],delta,motion]).astype(np.float32)
        assert g.shape==(GLOBAL_WIDTH,) and np.isfinite(g).all() and np.isfinite(table).all()
        return g,table,n
