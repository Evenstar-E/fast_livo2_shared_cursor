"""Causal pose/IMU direction consistency and working-map reliability proxies.

No independent map-health claim: a self-consistent biased map can pass these
proxies. Pose is the incumbent IMU/body pose; gyro/accel are in that same body
frame. Acceleration is specific force, NOT gravity-removed acceleration.
"""
from collections import deque
import numpy as np
from .telemetry_features import FEATURES

NAMES = tuple('body_rotation_rate_'+a for a in 'xyz') + tuple('body_gyro_'+a for a in 'xyz') + tuple('rotation_gyro_residual_'+a for a in 'xyz') + (
    'rotation_gyro_cosine','rotation_gyro_relative_error','rotation_gyro_residual_norm',
) + tuple('body_displacement_direction_'+a for a in 'xyz') + (
    'displacement_speed','consecutive_displacement_cosine',
) + tuple('body_specific_force_direction_'+a for a in 'xyz') + (
    'held_train_log_gap','nominal_held_train_log_gap','fit_log_gain','held_log_gain',
    'fit_held_gain_disagreement','working_map_support_fraction','log_approx_supported_points',
    'working_map_information','coverage_minus_past_median','held_log_ratio_to_past_median',
    'q_minus_past_median','scale_direction_agreement_with_past',
)
WIDTH = 76  # Existing 186-input projection stays byte-for-byte the same shape.
NAMES = NAMES + tuple('unavailable_reserved_'+str(i) for i in range(WIDTH-len(NAMES)))
ACTIVE = WIDTH-sum(n.startswith('unavailable_reserved_') for n in NAMES)


def rotation(q):
    q=np.asarray(q,dtype=float)
    if q.shape!=(4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>1e-5:
        raise ValueError('invalid xyzw quaternion')
    x,y,z,w=q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def log_rotation(r):
    v=np.array([r[2,1]-r[1,2],r[0,2]-r[2,0],r[1,0]-r[0,1]])/2
    angle=np.arccos(np.clip((np.trace(r)-1)/2,-1.,1.))
    if angle<1e-8:return v
    if np.pi-angle<1e-5:raise ValueError('ambiguous near-pi consecutive rotation')
    return v*angle/np.sin(angle)


def extract(events,poses,gyro,accel,imu_count):
    """Each row uses only arrived event, its posterior pose, and past rows.

    Input poses are [endpoint,x,y,z,qx,qy,qz,qw]. No interpolation, labels,
    schedules, absolute coordinates, dataset id or wall clock enter outputs.
    """
    n=len(events);poses=np.asarray(poses,float);gyro=np.asarray(gyro,float);accel=np.asarray(accel,float)
    if poses.shape!=(n,8) or gyro.shape!=(n,3) or accel.shape!=(n,3) or len(imu_count)!=n:
        raise ValueError('feature cardinality')
    if not np.isfinite(poses).all() or np.any(np.diff(poses[:,0])<=0):raise ValueError('pose order/finite')
    values=np.zeros((n,WIDTH));mask=np.zeros((n,WIDTH),bool);history=deque(maxlen=20)
    previous_r=previous_p=previous_t=previous_displacement=None
    for i,e in enumerate(events):
        if e['index']!=i+1:raise ValueError('event identity')
        t=poses[i,0];r=rotation(poses[i,4:]);p=poses[i,1:4]
        def put(name,v,valid=True):
            j=NAMES.index(name)
            if valid and np.isfinite(v):values[i,j]=v;mask[i,j]=True
        def vector(prefix,v):
            for a,x in zip('xyz',v):put(prefix+a,x)
        def cosine(name,a,b):
            den=np.linalg.norm(a)*np.linalg.norm(b)
            put(name,np.dot(a,b)/den if den>1e-10 else 0,den>1e-10)
        g,acc=gyro[i],accel[i];imu_ok=imu_count[i]>0 and np.isfinite(g).all() and np.isfinite(acc).all()
        if imu_ok:
            vector('body_gyro_',g)
            length=np.linalg.norm(acc)
            if length>1e-9:vector('body_specific_force_direction_',acc/length)
        if previous_r is not None:
            dt=t-previous_t
            if not 0<dt<=.5:raise ValueError('unsupported pose gap; no interpolation')
            rate=log_rotation(previous_r.T@r)/dt
            vector('body_rotation_rate_',rate)
            displacement=p-previous_p;body=previous_r.T@displacement;distance=np.linalg.norm(body)
            if distance>1e-9:vector('body_displacement_direction_',body/distance)
            put('displacement_speed',distance/dt)
            if previous_displacement is not None:cosine('consecutive_displacement_cosine',displacement,previous_displacement)
            if imu_ok:
                # Mean gyro is in current body axes; express it in previous body
                # axes before comparing the endpoint rotation increment.
                expected=previous_r.T@r@g;residual=rate-expected
                vector('rotation_gyro_residual_',residual)
                cosine('rotation_gyro_cosine',rate,expected)
                put('rotation_gyro_relative_error',np.linalg.norm(residual)/(np.linalg.norm(rate)+np.linalg.norm(expected)+1e-4))
                put('rotation_gyro_residual_norm',np.linalg.norm(residual))
            previous_displacement=displacement
        x=dict(zip(FEATURES,e['x']));m=dict(zip(FEATURES,e['mask']))
        def observed(*keys):return all(m[k] and np.isfinite(x[k]) for k in keys)
        def lg(a,b):return np.log((max(0.,a)+.005)/(max(0.,b)+.005))
        if observed('held','fit'):put('held_train_log_gap',lg(x['held'],x['fit']))
        if observed('null_held','null_fit'):put('nominal_held_train_log_gap',lg(x['null_held'],x['null_fit']))
        if observed('null_fit','fit'):put('fit_log_gain',lg(x['null_fit'],x['fit']))
        if observed('null_held','held'):put('held_log_gain',lg(x['null_held'],x['held']))
        if observed('null_fit','fit','null_held','held'):
            put('fit_held_gain_disagreement',lg(x['null_fit'],x['fit'])-lg(x['null_held'],x['held']))
        if observed('coverage'):put('working_map_support_fraction',x['coverage'])
        if observed('coverage','sample_points'):put('log_approx_supported_points',np.log1p(max(0.,x['coverage']*x['sample_points'])))
        if observed('information'):put('working_map_information',x['information'])
        for key,name,fn in (
            ('coverage','coverage_minus_past_median',lambda v,b:v-b),
            ('held','held_log_ratio_to_past_median',lg),
            ('q_raw','q_minus_past_median',lambda v,b:v-b)):
            past=[h[key] for h in history if key in h]
            if observed(key) and len(past)>=5:put(name,fn(x[key],np.median(past)))
        past=[h['q_raw']-1 for h in history if 'q_raw' in h]
        if observed('q_raw') and len(past)>=5:
            # A consistency descriptor, not a hard fault rule.
            signs=np.sign(np.where(np.abs(past)>1e-4,past,0))
            put('scale_direction_agreement_with_past',np.mean(signs==np.sign(x['q_raw']-1)))
        history.append({k:x[k] for k in ('coverage','held','q_raw') if observed(k)})
        previous_r,previous_p,previous_t=r,p,t
    if not np.isfinite(values).all():raise ValueError('nonfinite output')
    return values,mask
