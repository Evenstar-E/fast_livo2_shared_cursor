"""Label-only logic and group split guards; never imported by feature code."""
from __future__ import annotations


def labels_for(events, *, injected, onset=50.0, end=60.0, scale_end=1.08):
    if not onset < end or scale_end == 1.0:
        raise ValueError('invalid injection specification')
    result = []
    for event in events:
        header = event['header_time']
        # The audited feeder uses scan header time, not per-point/end time.
        active = bool(injected and onset < header < end)
        exposed = bool(injected and header > onset)
        phase = ('active_injection' if active else 'post_injection_history_exposed'
                 if exposed else 'no_injected_fault')
        result.append(dict(index=event['index'], sensor_fault_active=int(active),
                           onset_sec=onset if active else None, phase=phase,
                           history_exposed=exposed, map_pollution='UNKNOWN',
                           recovery_benefit='UNKNOWN', utility_target=None))
    return result


def validate_split(records):
    """No bag or declared scene group can occur in two data partitions."""
    seen = {}
    for row in records:
        for key in ('bag_sha256', 'scene_group'):
            value = row.get(key)
            if not value:
                raise ValueError('missing group identity')
            identity = (key, value)
            split = row['partition']
            if split not in ('development', 'train', 'validation', 'calibration', 'test'):
                raise ValueError('invalid partition')
            if identity in seen and seen[identity] != split:
                raise ValueError('group leakage: '+key)
            seen[identity] = split
