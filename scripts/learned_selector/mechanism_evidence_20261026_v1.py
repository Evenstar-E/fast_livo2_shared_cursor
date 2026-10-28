"""Saved-observation diagnostics. No fault labels, schedules, future or clean input."""
import numpy as np

def scalar_stats(values):
    x=np.asarray(values,float);x=x[np.isfinite(x)]
    if not len(x):return dict(n=0,median=None,p10=None,p90=None)
    return dict(n=len(x),median=float(np.median(x)),p10=float(np.quantile(x,.1)),p90=float(np.quantile(x,.9)))

def local_surface(xyz,max_reference=4096,max_queries=512):
    p=np.asarray(xyz,float);p=p[np.isfinite(p).all(1)&(np.linalg.norm(p,axis=1)>.05)]
    n=len(p)
    if n<9:return dict(points=n,queries=0,supported=0,coverage=0.,abs_residual=scalar_stats([]),thickness=scalar_stats([]),nearest_distance=scalar_stats([]))
    ref=p[np.linspace(0,n-1,min(n,max_reference),dtype=int)];qid=np.linspace(0,len(ref)-1,min(max_queries,len(ref)),dtype=int);query=ref[qid]
    residual=[];thickness=[];nearest=[];support=0
    for begin in range(0,len(query),64):
        q=query[begin:begin+64];dist=np.maximum(0,(q*q).sum(1)[:,None]+(ref*ref).sum(1)[None,:]-2*q@ref.T)
        dist[np.arange(len(q)),qid[begin:begin+len(q)]]=np.inf
        k=min(16,len(ref)-1);ids=np.argpartition(dist,k-1,axis=1)[:,:k]
        for a,b,c in zip(q,ids,dist):
            near=c[b];nearest.append(float(np.sqrt(np.min(near))));selected=ref[b[near<=.5**2]]
            if len(selected)<8:continue
            centered=selected-selected.mean(0);values,vectors=np.linalg.eigh(centered.T@centered/len(selected))
            if values[1]<=1e-12:continue
            support+=1;residual.append(float((a-selected.mean(0))@vectors[:,0]));thickness.append(float(max(0,values[0])/max(values.sum(),1e-12)))
    az=np.arctan2(p[:,1],p[:,0]);el=np.arcsin(np.clip(p[:,2]/np.linalg.norm(p,axis=1),-1,1));bins=np.minimum(11,((az+np.pi)*12/(2*np.pi)).astype(int))+12*np.searchsorted([-.14,0,.14],el)
    counts=np.bincount(bins,minlength=48);r=np.asarray(residual)
    return dict(points=n,queries=len(query),supported=support,coverage=support/len(query),abs_residual=scalar_stats(np.abs(r)),signed_mad=float(1.4826*np.median(np.abs(r-np.median(r)))) if len(r) else None,thickness=scalar_stats(thickness),nearest_distance=scalar_stats(nearest),occupied_cells=int(np.sum(counts>0)),cell_fraction=(counts/n).tolist())

def residual_scale(r,s,H):
    """OLS scale after projecting away all six pose directions."""
    r=np.asarray(r,float);s=np.asarray(s,float);H=np.asarray(H,float)
    if len(r)<8:return dict(scale=None,information=None,pose=None)
    hs=np.linalg.lstsq(H,s,rcond=1e-10)[0];sr=s-H@hs;den=float(sr@sr);total=float(s@s)
    info=max(0.,den/total) if total>1e-20 else 0.
    if info<=1e-10:return dict(scale=None,information=info,pose=None)
    scale=-float(sr@r)/den;pose=np.linalg.lstsq(H,-r-s*scale,rcond=1e-10)[0]
    return dict(scale=scale,information=info,pose=pose.tolist())

def fit_model(r,s,H,variance,P,mode,include_scale):
    X=np.column_stack([H,s]) if include_scale else H
    if len(r)<8:return None
    if mode=='free':
        if include_scale:
            answer=residual_scale(r,s,H)
            if answer['scale'] is None:return None
            return np.array(answer['pose']+[answer['scale']])
        return np.linalg.lstsq(X,-r,rcond=1e-10)[0]
    if mode!='prior':raise ValueError(mode)
    if P.shape!=(6,6) or not np.isfinite(P).all():raise ValueError('pose covariance')
    ev,U=np.linalg.eigh((P+P.T)/2)
    if ev.min()<=0:raise ValueError('nonpositive pose covariance')
    # P is a nuisance prior. Its projected variance is not added again to W.
    w=1/(np.maximum(variance,0)+.01**2);A=X.T@(w[:,None]*X);b=-X.T@(w*r)
    A[:6,:6]+=(U*(1/ev))@U.T
    return np.linalg.lstsq(A,b,rcond=1e-12)[0]

def score_group(source_ids,matched,r,s,H,variance,P,all_ids):
    source_ids=np.asarray(source_ids,int);matched=np.asarray(matched,bool);valid=matched&np.isfinite(r)&np.isfinite(s)&np.isfinite(H).all(1)
    train_ids=set(np.asarray(sorted(set(all_ids)))[::2].tolist());train=valid&np.array([i in train_ids for i in source_ids]);held=valid&~np.array([i in train_ids for i in source_ids])
    nheld=sum(i not in train_ids for i in source_ids);rr=np.asarray(r);ss=np.asarray(s);HH=np.asarray(H);vv=np.asarray(variance)
    scale_only=-float(ss[valid]@rr[valid])/float(ss[valid]@ss[valid]) if np.sum(valid)>0 and ss[valid]@ss[valid]>1e-20 else None
    output=dict(total=len(valid),matched=int(valid.sum()),coverage=float(valid.mean()),train=int(train.sum()),held=int(held.sum()),held_total=nheld,scale_only=scale_only,free_information=residual_scale(rr[train],ss[train],HH[train])['information'])
    for mode in ('free','prior'):
        for plus in (False,True):
            name=mode+('_scale_pose' if plus else '_pose_only')
            try:beta=fit_model(rr[train],ss[train],HH[train],vv[train],P,mode,plus)
            except ValueError as exc:
                output[name]=dict(scale=None,held_rmse=None,held_penalized_rmse=None,pose=None,reason=str(exc));continue
            if beta is None or not np.sum(held):output[name]=dict(scale=None,held_rmse=None,held_penalized_rmse=None,pose=None);continue
            X=np.column_stack([HH[held],ss[held]]) if plus else HH[held];error=rr[held]+X@beta
            output[name]=dict(scale=float(beta[-1]) if plus else None,pose=beta[:6].tolist(),held_rmse=float(np.sqrt(np.mean(error**2))),held_penalized_rmse=float(np.sqrt((error@error+(nheld-held.sum())*.5**2)/max(nheld,1))))
    return output
