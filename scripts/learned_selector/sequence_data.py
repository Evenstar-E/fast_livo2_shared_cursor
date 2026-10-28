"""Multi-episode labels/evaluation remain separate from runtime model inputs."""
import math
import numpy as np
from .l2_data import alarm_indices
from .telemetry_features import FEATURES

UNITS=dict(sigma_q_floor=.001,residual_unit_m=.005,position_unit_m=.01,rotation_unit_rad=.01,
           velocity_unit_mps=.1,gyro_unit_radps=.1,acc_std_unit_mps2=.1,clip=10.)

def validate_schedule(rows):
    end=-1
    for a,b,d,k in rows:
        if not all(math.isfinite(v) for v in (a,b,d)) or a<10 or b<=a or a<end or not 0<abs(d)<=.1 or k not in ('ramp','step','triangle'):
            raise ValueError('invalid injection schedule')
        end=b

def scale_at(rows,t):
    for i,(a,b,d,k) in enumerate(rows,1):
        if a<=t<b:
            v=(t-a)/(b-a)
            if k=='step':v=1.
            elif k=='triangle':v=1-abs(2*v-1)
            return 1+d*v,i
    return 1.,0

def labels_for_schedule(events,rows):
    validate_schedule(rows);labels=[]
    for e in events:
        t=e['header_time'];q,episode=scale_at(rows,t);active=abs(q-1)>1e-12
        exposed=any(t>a or (t==a and k=='step') for a,b,d,k in rows)
        labels.append(dict(index=e['index'],sensor_fault_active=int(active),episode=episode if active else 0,
            onset_sec=rows[episode-1][0] if active else None,history_exposed=exposed,
            phase='active_injection' if active else 'sensor_nominal_history_exposed' if exposed else 'no_injection_exposure',
            map_pollution='UNKNOWN',utility_target=None))
    return labels

def inputs(events):
    """17 fixed-unit channels + masks. No bag time, injection state, or labels."""
    from .relative_selector import physical_event
    output=[];eligibility=[];previous=-math.inf
    for e in events:
        if e['available_time']<=previous:raise ValueError('noncausal ordering')
        previous=e['available_time'];v,m=physical_event(e,UNITS);v=list(v);m=list(m)
        for name in ('fit','held','null_fit','null_held'):
            j=FEATURES.index(name);ok=bool(e['mask'][j]) and math.isfinite(e['x'][j]) and e['x'][j]>=0
            v.append(min(10.,math.log1p(e['x'][j]/.005)) if ok else 0.);m.append(ok)
        output.append(v+[float(z) for z in m]);eligibility.append(bool(all(e['mask']) and all(m)))
    return np.asarray(output,dtype=np.float32),np.asarray(eligibility,dtype=bool)

def evaluate_sequence(case,score,eligible,config):
    events,labels,episodes=case['events'],case['labels'],case['schedule']
    found=[];alarms=[]
    for i,start in alarm_indices(score,eligible,config['threshold'],config['confirm_events'],config['rearm_events']):
        e=events[i];label=labels[i]
        phase='clean' if not episodes else 'active_injection' if label['sensor_fault_active'] else 'history_exposed' if label['history_exposed'] else 'pre_fault'
        row=dict(index=e['index'],available_time=e['available_time'],onset_proxy_sec=events[start]['header_time'],
                 phase=phase,episode=label['episode'],score=float(score[i]))
        alarms.append(row)
    for k,(a,b,d,shape) in enumerate(episodes,1):
        hits=[r for r in alarms if r['episode']==k and a<=r['available_time']<b]
        first=hits[0] if hits else None
        horizon=min(b+config['episode_horizon_sec'],episodes[k][0] if k<len(episodes) else math.inf)
        late=[r for r in alarms if b<=r['available_time']<=horizon]
        found.append(dict(episode=k,start=a,end=b,full_end_observed=events[-1]['available_time']>=b,
            detected_inside=bool(first),first_delay_sec=first['available_time']-a if first else None,
            onset_proxy_error_sec=abs(first['onset_proxy_sec']-a) if first else None,
            bounded_post_alarm_count=len(late),post_alarm_attribution='ambiguous_history_not_accepted_detection'))
    truth=np.asarray([v['sensor_fault_active'] for v in labels]);positive=(score>=config['threshold'])&eligible
    return dict(events=len(events),eligible=int(eligible.sum()),unknown=int((~eligible).sum()),
        positive_events=int(truth.sum()),tp=int((positive&(truth==1)).sum()),fp=int((positive&(truth==0)).sum()),
        fn_including_unknown=int((~positive&(truth==1)).sum()),
        alarms=alarms,alarms_by_phase={k:sum(r['phase']==k for r in alarms) for k in ['clean','pre_fault','active_injection','history_exposed']},
        episodes=found,episode_denominator=len(episodes),episodes_detected=sum(r['detected_inside'] for r in found),
        score_calibrated=False,onset_interval='UNKNOWN',cause='UNRESOLVED',map_action='NONE',paper_ready=False)
