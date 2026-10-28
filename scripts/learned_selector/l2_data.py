"""L2 train-only transforms and causal alarm evaluation (labels stay separate)."""
import json
import numpy as np
from .telemetry_features import FEATURES, iter_windows


def read_case(root, name):
    events = [json.loads(x) for x in (root/name/'events.jsonl').read_text(encoding='utf8').splitlines()]
    labels = [json.loads(x) for x in (root/name/'labels.jsonl').read_text(encoding='utf8').splitlines()]
    assert len(events) == len(labels)
    assert all(e['index'] == y['index'] for e, y in zip(events, labels))
    return dict(name=name, events=events, labels=labels)


def fit_normalizer(cases):
    x = np.array([e['x'] for c in cases for e in c['events']], dtype=np.float64)
    m = np.array([e['mask'] for c in cases for e in c['events']], dtype=bool)
    center, spread = [], []
    for j in range(len(FEATURES)):
        v = x[m[:, j], j]
        if not len(v):
            center.append(0.); spread.append(1.); continue
        q = np.quantile(v, [.25, .5, .75])
        center.append(float(q[1])); spread.append(float(max(q[2]-q[0], 1e-6)))
    return dict(center=center, spread=spread, features=list(FEATURES))


def tensors(case, norm, length=64):
    windows = list(iter_windows(case['events'], length))
    x = np.asarray([w['x'] for w in windows], dtype=np.float32)
    m = np.asarray([w['mask'] for w in windows], dtype=bool)
    em = np.asarray([w['event_mask'] for w in windows], dtype=bool)
    z = np.clip((x-np.asarray(norm['center']))/np.asarray(norm['spread']), -10, 10)
    z = np.where(m, z, 0).astype(np.float32)
    channels = np.concatenate((z, m.astype(np.float32), em[:, :, None]), axis=2)
    # Same 64-event information, compressed rather than flattened for trees.
    count = m.sum(axis=1)
    mean = (z*m).sum(axis=1)/np.maximum(count, 1)
    var = (((z-mean[:, None, :])**2)*m).sum(axis=1)/np.maximum(count, 1)
    lo = np.min(np.where(m, z, np.inf), axis=1)
    hi = np.max(np.where(m, z, -np.inf), axis=1)
    lo[count == 0] = 0; hi[count == 0] = 0
    tree = np.concatenate((z[:, -1], m[:, -1], mean, np.sqrt(var), lo, hi,
                           count/length, z[:, -1]-z[:, 0], em.mean(axis=1)[:, None]), axis=1).astype(np.float32)
    y = np.array([a['sensor_fault_active'] for a in case['labels']], dtype=np.float32)
    onset = np.full(len(y), length+1, dtype=np.int64)
    for i, label in enumerate(case['labels']):
        if not y[i]: continue
        start = max(0, i-length+1)
        times = np.array([e['header_time'] for e in case['events'][start:i+1]])
        t0 = label['onset_sec']
        if t0 < times[0]: onset[i] = length
        else: onset[i] = length-len(times)+int(np.searchsorted(times, t0, side='right')-1)
    eligible = m[:, -1].all(axis=1)
    return channels.transpose(0, 2, 1), tree, y, onset, eligible


def alarm_indices(probability, eligible, threshold=.5, confirm=3, rearm=3):
    """One event per positive run; UNKNOWN breaks confirmation, cannot rearm."""
    high = low = 0; armed = True; result = []
    for i, (p, ok) in enumerate(zip(probability, eligible)):
        if not ok:
            high = low = 0
            continue
        if p >= threshold:
            high += 1; low = 0
            if armed and high >= confirm:
                result.append((i, i-confirm+1)); armed = False
        else:
            high = 0; low += 1
            if low >= rearm: armed = True
    return result


def onset_value(case, i, category, length=64):
    if category >= length: return None
    start = max(0, i-length+1); pad = max(0, length-i-1)
    if category < pad: return None
    return case['events'][start+category-pad]['header_time']


def evaluate(case, probability, eligible, config, onset_probs=None):
    alarms = []
    for i, start in alarm_indices(probability, eligible, config['threshold'], config['confirm_events'], config['rearm_events']):
        e = case['events'][i]
        t0 = case['events'][start]['header_time']
        bounds = None; confidence = None; in_window_mass = None
        if onset_probs is not None:
            op = onset_probs[i]; cat = int(np.argmax(op)); confidence = float(op[cat])
            t0 = onset_value(case, i, cat)
            # The special classes are NOT ordinal time bins. Report a conditional
            # interval only when >=90% mass is inside the historical window.
            in_window_mass = float(op[:64].sum())
            if in_window_mass >= .9:
                cats = np.searchsorted(np.cumsum(op[:64])/in_window_mass, [.05, .95])
                bounds = [onset_value(case, i, int(k)) for k in cats]
        alarms.append(dict(index=e['index'], available_time=e['available_time'], onset=t0,
                           onset_interval_conditional_uncalibrated=bounds, onset_mode_mass=confidence,
                           onset_in_window_mass=in_window_mass,
                           score=float(probability[i])))
    return summarize_alarms(case, alarms, int((~eligible).sum()))


def summarize_alarms(case, alarms, unknown=None):
    injected = any(x['sensor_fault_active'] for x in case['labels'])
    first = case['events'][0]['available_time']; last = case['events'][-1]['available_time']
    phases = dict(clean=[], pre_fault=[], within_fault=[], post_fault_observed=[])
    for a in alarms:
        t = a['available_time']
        phase = 'clean' if not injected else 'pre_fault' if t <= 50 else 'within_fault' if t < 60 else 'post_fault_observed'
        phases[phase].append(a)
    valid = phases['within_fault']
    a = valid[0] if valid else None
    return dict(events=len(case['events']), observed_endpoint=[first, last],
                unknown_events=unknown, fault_episode_denominator=int(injected),
                within_fault_detected=bool(a), fault_end_observed=last >= 60 if injected else None,
                first_detection_delay_sec=a['available_time']-50 if a else None,
                first_onset_abs_error_sec=abs(a['onset']-50) if a and a['onset'] is not None else None,
                alarm_count_by_phase={k:len(v) for k,v in phases.items()}, alarms=alarms,
                recovery_eligibility='UNMEASURED', public_switches='NOT_RUN', recovery_utility='UNKNOWN')
