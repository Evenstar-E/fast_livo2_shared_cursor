"""Continuous causal relative-attention backbone with two independent heads.

Each layer attends to at most 128 previous events plus the current event.
Two layers have a theoretical raw-input receptive field of 256 previous events;
the cache is not a 128-event limit on that composed receptive field. Detaching
layer-input caches truncates training gradients, not inference history. This is
the bounded-memory variant, NOT the older cache-free, 64-event-reset comparator.
No map, mux, or robot-control authority is granted by model scores.
"""
import torch
from torch import nn
from .sequence_models import RelativeAttentionBlock


class ActionTransformer(nn.Module):
    def __init__(self, extra_width, hidden=32, heads=4, layers=2, memory=128):
        super().__init__()
        if hidden <= 0 or heads <= 0 or hidden % heads or layers <= 0:
            raise ValueError('invalid attention dimensions')
        if memory <= 0 or memory > 128 or extra_width < 0:
            raise ValueError('memory must be 1..128; extra_width nonnegative')
        self.hidden, self.heads, self.layers, self.memory = hidden, heads, layers, memory
        self.projection = nn.Linear(34 + extra_width, hidden)
        self.blocks = nn.ModuleList([RelativeAttentionBlock(hidden, heads, memory) for _ in range(layers)])
        self.scale_head = nn.Linear(hidden, 1)
        self.quality_head = nn.Linear(hidden, 1)

    def forward(self, x, state=None):
        if x.ndim != 3 or x.shape[1] == 0:
            raise ValueError('expected nonempty batch,time,features')
        if state is not None and len(state) != self.layers:
            raise ValueError('one cache per layer required')
        z = self.projection(x)
        previous = state if state is not None else [None] * self.layers
        caches = []
        for block, cache in zip(self.blocks, previous):
            if cache is not None and (cache.ndim != 3 or cache.shape[1] > self.memory
                                      or cache.shape[0] != x.shape[0] or cache.shape[2] != self.hidden):
                raise ValueError('invalid cache dimensions')
            z, cache = block(z, cache)
            caches.append(cache)
        return torch.cat((self.scale_head(z), self.quality_head(z)), -1), caches

    def metadata(self):
        return dict(backbone='causal_transformer_bounded_relative_cache', hidden=self.hidden,
                    heads=self.heads, layers=self.layers, memory_events_per_layer=self.memory,
                    maximum_raw_input_history_events=self.layers * self.memory,
                    parameter_count=sum(p.numel() for p in self.parameters()),
                    logical_cache_values_per_stream=self.layers * self.memory * self.hidden,
                    chunk_reset=False, gradient_history='detached at training chunk boundaries',
                    quality_target_is_proxy=True, public_output=False, paper_ready=False)


def predict(model, x, chunk=64):
    if chunk <= 0 or len(x) == 0:
        raise ValueError('nonempty input and positive chunk required')
    state, values = None, []
    with torch.no_grad():
        for start in range(0, len(x), chunk):
            output, state = model(x[None, start:start + chunk], state)
            values.append(output[0].sigmoid())
    return torch.cat(values).cpu().numpy()
