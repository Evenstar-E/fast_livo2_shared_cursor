"""Private, bounded parameter heads. Injection parameters are NOT action utility.

The existing ActionTransformer (including both classification heads) is frozen.
No schedule, label, absolute time, GT, map or action result enters forward().
Outputs describe current/past sensor scale; they grant no replay authority.
"""
import math
import numpy as np
import torch
from torch import nn
from .sequence_data import scale_at, validate_schedule

LAGS_SEC = (0., .5, 1., 2.)
ONSET_BIN_SEC = .25
ONSET_HORIZON_SEC = 6.
ONSET_BINS = 24
ONSET_CLASSES = 26  # nominal=0; ages [0,6)=1..24; older active=25
LOG_BOUND = -math.log(.9)


def validate_groups(groups):
    expected = {'exp04': 'train', 'exp05': 'train',
                'exp11': 'validation', 'exp16': 'validation'}
    bags, scenes, cases = {}, {}, set()
    for row in groups:
        name, part = row['case'], row['partition']
        if name in cases or expected.get(name[:5]) != part:
            raise ValueError('unexpected case/group partition')
        cases.add(name)
        for mapping, key in ((bags, row['bag_sha256']), (scenes, row['scene_group'])):
            if not key or (key in mapping and mapping[key] != part):
                raise ValueError('source bag/scene leakage')
            mapping[key] = part
    return True


def parameter_targets(events, schedule):
    """Label-only path: past/current injected scale, never future curve targets.

    Age is measured at the current scan header (not decision availability).
    Nominal sensor scale does not certify checkpoint/map health. Earlier than
    observed bag start is masked, not padded with a fictitious healthy label.
    """
    validate_schedule(schedule)
    if not events:
        raise ValueError('empty sequence')
    previous = -math.inf
    rows = []
    origin = float(events[0]['header_time'])
    for event in events:
        t, available = float(event['header_time']), float(event['available_time'])
        if not math.isfinite(t) or not math.isfinite(available) or t > available or available <= previous:
            raise ValueError('noncausal event ordering')
        previous = available
        scale, episode = scale_at(schedule, t)
        active = abs(scale - 1.) > 1e-12
        age = t - schedule[episode-1][0] if active else None
        onset = min(ONSET_BINS + 1, 1 + int(age / ONSET_BIN_SEC)) if active else 0
        means, masks = [], []
        for lag in LAGS_SEC:
            valid = t - lag >= origin
            means.append(math.log(scale_at(schedule, t-lag)[0]) if valid else 0.)
            masks.append(valid)
        rows.append(dict(index=event['index'], onset_class=onset,
                         log_scale=means, scale_mask=masks,
                         sensor_fault_active=active,
                         history_exposed=any(t > a or (t == a and k == 'step') for a,b,d,k in schedule),
                         benefit_label='UNKNOWN', map_health='UNKNOWN'))
    return rows


class ParameterHead(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.onset = nn.Linear(hidden, ONSET_CLASSES)
        self.scale = nn.Linear(hidden, 2*len(LAGS_SEC))

    def forward(self, features):
        raw = self.scale(features)
        mean = LOG_BOUND * torch.tanh(raw[..., :len(LAGS_SEC)])
        # Bounded positive standard deviation, not calibrated confidence.
        sigma = .001 + .199 * torch.sigmoid(raw[..., len(LAGS_SEC):])
        return dict(onset_logits=self.onset(features), log_scale_mean=mean,
                    log_scale_sigma=sigma, correction=torch.exp(-mean))


class FrozenRecoveryParameters(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone.eval()
        self.backbone.requires_grad_(False)
        self.parameters_head = ParameterHead(backbone.hidden).to(
            dtype=next(backbone.parameters()).dtype)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def encode(self, x, state=None):
        captured = []
        hook = self.backbone.scale_head.register_forward_pre_hook(
            lambda module, args: captured.append(args[0]))
        try:
            with torch.no_grad():
                old_logits, caches = self.backbone(x, state)
        finally:
            hook.remove()
        if len(captured) != 1:
            raise ValueError('backbone feature tap contract')
        return captured[0].detach(), old_logits, caches

    def forward(self, x, state=None):
        z, old, state = self.encode(x, state)
        result = self.parameters_head(z)
        result['old_classifier_logits'] = old
        return result, state


def parameter_loss(output, onset, log_scale, eligible, scale_mask):
    """No benefit labels or post-action outputs are consumed."""
    from torch.nn import functional as F
    mask = scale_mask & eligible[..., None]
    if not eligible.any() or not mask.any():
        return None
    ce = F.cross_entropy(output['onset_logits'][eligible], onset[eligible])
    mean, sigma = output['log_scale_mean'][mask], output['log_scale_sigma'][mask]
    residual = (log_scale[mask] - mean) / sigma
    return ce + (.5*residual.square() + sigma.log()).mean()
