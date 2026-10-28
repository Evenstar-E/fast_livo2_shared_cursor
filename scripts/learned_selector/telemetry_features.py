"""Version-2 continuous incumbent adapter; does not alter sealed v1 adapter."""
from __future__ import annotations

import math
from collections import deque

COMMON=('imu_count','gyro_mean_norm','gyro_std_norm','acc_mean_norm','acc_std_norm','bias_g_norm','bias_a_norm')
SCALE=('q_raw','sigma','fit','held','coverage','information')
NOMINAL=('null_fit','null_held','nominal_fit_position_shift_m')
MOTION=('sample_points','native_position_update_m','native_rotation_update_rad','native_velocity_update_mps')
FEATURES=('dt_sec','scan_duration_sec')+COMMON+SCALE+NOMINAL+MOTION


def convert(rows, inputs, epoch):
    if len(rows)!=len(inputs):raise ValueError('missing telemetry event')
    result=[];previous_header=None;previous_end=-math.inf;previous_wall=-math.inf
    for position,(row,source) in enumerate(zip(rows,inputs),1):
        if int(row['index'])!=position or int(source['index'])!=position:
            raise ValueError('index mismatch')
        header,available=float(row['header_time']),float(row['available_time'])
        if not math.isfinite(header) or not math.isfinite(available):raise ValueError('invalid timestamp')
        if abs(header-float(source['header_relative_sec']))>1e-6 or abs(available-(float(source['endpoint_absolute_sec'])-epoch))>1e-6:
            raise ValueError('timestamp association')
        if available<=previous_end or available<header or (previous_header is not None and header<=previous_header):
            raise ValueError('time order')
        wall=float(row['emitted_wall_since_start_sec']);cost=float(row['process_wall_sec'])
        if not math.isfinite(wall) or not math.isfinite(cost) or wall<=previous_wall or cost<0:
            raise ValueError('wall timing invalid')
        if row['route']!='incumbent_working_map':raise ValueError('wrong feature branch')
        for name in ('geometry_evaluated','nominal_valid','scale_valid'):
            if row[name] not in ('0','1'):raise ValueError('invalid validity marker')
        geometry=row['geometry_evaluated']=='1'
        if not geometry and (row['nominal_valid']=='1' or row['scale_valid']=='1'):
            raise ValueError('valid fit without geometry')
        if int(row['imu_count'])!=int(source['imu_count']):raise ValueError('IMU association')
        values=[None if previous_header is None else header-previous_header,available-header]
        for group,valid in ((COMMON,int(row['imu_count'])>0),(SCALE,geometry and row['scale_valid']=='1'),
                            (NOMINAL,geometry and row['nominal_valid']=='1'),(MOTION,geometry)):
            for name in group:
                value=float(row[name])
                values.append(value if valid and math.isfinite(value) else None)
        result.append(dict(index=position,header_time=header,available_time=available,
                           x=[v if v is not None else 0.0 for v in values],mask=[v is not None for v in values]))
        previous_header,previous_end,previous_wall=header,available,wall
    return result


def iter_windows(events,length=64):
    buffer=deque(maxlen=length)
    if length<1:raise ValueError('window length')
    last=-math.inf
    for event in events:
        if event['available_time']<=last:raise ValueError('window order')
        if len(event['x'])!=len(FEATURES) or len(event['mask'])!=len(FEATURES):raise ValueError('feature width')
        buffer.append(event);last=event['available_time'];pad=length-len(buffer)
        yield dict(x=[[0.0]*len(FEATURES) for _ in range(pad)]+[list(x['x']) for x in buffer],
                   mask=[[False]*len(FEATURES) for _ in range(pad)]+[list(x['mask']) for x in buffer],
                   event_mask=[False]*pad+[True]*len(buffer))
