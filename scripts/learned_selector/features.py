"""Causal adapters for the audited 20261016 native prequential CSV format.

Times are sensor-time logical availability, NOT measured execution latency.
The legacy score stream switches reference branches. Mask those rows rather
than letting a detector depend on a proposal which it is meant to create.
"""
from __future__ import annotations

import math
from collections import deque

GEOMETRY = ('q_raw', 'sigma', 'fit', 'held', 'null_held', 'coverage', 'information')
FEATURES = ('dt_sec', 'scan_duration_sec', 'imu_count', 'sampled_point_count') + GEOMETRY
CANDIDATE_FEATURES = ('q', 'sigma', 'fit', 'held', 'null_held', 'coverage', 'information')
TIME_TOLERANCE = 1e-6  # CSV/absolute-epoch rounding, not an algorithm gate.


def finite(value):
    number = float(value)
    return number if math.isfinite(number) else None


def indexed(rows):
    result = {}
    for row in rows:
        key = int(row['index'])
        if key in result:
            raise ValueError('duplicate event index')
        result[key] = row
    return result


def candidate_records(rows):
    """Keep emission order and actual availability; never use header as arrival."""
    result = []
    previous = -math.inf
    for row in rows:
        available = finite(row['available_time'])
        header = finite(row['header_time'])
        if available is None or header is None or available < header or available < previous:
            raise ValueError('invalid candidate availability/order')
        previous = available
        values = [finite(row[name]) for name in CANDIDATE_FEATURES]
        result.append(dict(proposal=int(row['proposal']), index=int(row['index']),
                           header_time=header, available_time=available, route=row['route'],
                           x=[v if v is not None else 0.0 for v in values],
                           mask=[v is not None for v in values]))
    return result


def candidates_asof(records, time):
    return [r for r in records if r['available_time'] <= time]


def build_events(inputs, scores, ends, lifecycle, epoch):
    """One event for EVERY input frame, including initialization and missing fits.

    Lifecycle events at an exact decision time are conservatively masked both
    on freeze and release: release can occur either before or after scoring.
    The implementation consumes no future lifecycle event to mark earlier data.
    """
    if not math.isfinite(epoch):
        raise ValueError('invalid epoch')
    score_by, end_by = indexed(scores), indexed(ends)
    input_by = indexed(inputs)
    if (set(score_by) | set(end_by)) - set(input_by):
        raise ValueError('orphan score event')
    if set(score_by) != set(end_by):
        raise ValueError('score/end ledger mismatch')
    last_lifecycle = -math.inf
    for row in lifecycle:
        t = finite(row['available_time'])
        if t is None or t < last_lifecycle or row['action'] not in ('freeze', 'release'):
            raise ValueError('invalid lifecycle order/action')
        last_lifecycle = t
    live, pos, previous_header, previous_time, previous_index = set(), 0, None, -math.inf, 0
    events = []
    for row in inputs:
        index = int(row['index'])
        header = finite(row['header_relative_sec'])
        absolute = finite(row['endpoint_absolute_sec'])
        if header is None or absolute is None:
            raise ValueError('nonfinite input timestamp')
        time = absolute - epoch
        if index != previous_index + 1 or time <= previous_time or time < header:
            raise ValueError('input chronology/index gap')
        if previous_header is not None and header <= previous_header:
            raise ValueError('input header order')
        end, score = end_by.get(index), score_by.get(index)
        if end:
            logical = float(end['available_time'])
            if abs(logical-time) > TIME_TOLERANCE or abs(float(end['header_time'])-header) > TIME_TOLERANCE:
                raise ValueError('endpoint join mismatch')
            time = logical
        if score and abs(float(score['time'])-header) > TIME_TOLERANCE:
            raise ValueError('score header mismatch')
        if score and float(score['q_applied']) != 1.0:
            raise ValueError('requires no-repair incumbent input')
        touched = False
        while pos < len(lifecycle) and float(lifecycle[pos]['available_time']) <= time:
            event = lifecycle[pos]
            at, proposal = float(event['available_time']), int(event['proposal'])
            touched |= abs(at-time) <= TIME_TOLERANCE
            if event['action'] == 'freeze':
                if proposal in live:
                    raise ValueError('duplicate live proposal')
                live.add(proposal)
            else:
                if proposal not in live:
                    raise ValueError('release without freeze')
                live.remove(proposal)
            pos += 1
        branch_dependent = bool(live) or touched
        values = [None if previous_header is None else header-previous_header,
                  time-header, finite(row['imu_count']),
                  finite(end['source_point_count']) if end else None]
        reason = 'candidate_branch' if branch_dependent else 'missing_score'
        geometry = [None] * len(GEOMETRY)
        if score and not branch_dependent:
            # The audited Fit computes these only at its terminal evaluation;
            # coverage>=.3 is the producer's existing validity rule, not tuning.
            if finite(score['coverage']) is not None and float(score['coverage']) >= .3:
                geometry = [finite(score[k]) for k in GEOMETRY]
                reason = 'incumbent_fit' if all(v is not None for v in geometry) else 'nonfinite_fit'
            else:
                reason = 'invalid_or_unfinished_fit'
        values += geometry
        events.append(dict(index=index, header_time=header, available_time=time,
                           x=[v if v is not None else 0.0 for v in values],
                           mask=[v is not None for v in values], geometry_status=reason))
        previous_header, previous_time, previous_index = header, time, index
    return events


class CausalWindow:
    """64-event windows, no resampling, no interpolation and no normalization.

    Feature order is the allowlist above. IDs and absolute/relative clock are
    metadata only; export_model_input never exposes them to a learner.
    """
    def __init__(self, length=64):
        if length < 1:
            raise ValueError('invalid window length')
        self.length = length
        self.buffer = deque(maxlen=length)
        self.last = -math.inf

    def push(self, event):
        if event['available_time'] <= self.last:
            raise ValueError('non-increasing stream')
        if len(event['x']) != len(FEATURES) or len(event['mask']) != len(FEATURES):
            raise ValueError('invalid feature width')
        if not all(math.isfinite(v) for v in event['x']):
            raise ValueError('nonfinite tensor')
        self.buffer.append(event)
        self.last = event['available_time']
        pad = self.length - len(self.buffer)
        return dict(x=[[0.0]*len(FEATURES) for _ in range(pad)] + [list(e['x']) for e in self.buffer],
                    mask=[[False]*len(FEATURES) for _ in range(pad)] + [list(e['mask']) for e in self.buffer],
                    event_mask=[False]*pad + [True]*len(self.buffer))


def windows(events, length=64):
    stream = CausalWindow(length)
    return [stream.push(event) for event in events]
