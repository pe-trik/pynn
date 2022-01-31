# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import random
from turtle import forward

import torch
import torch.nn as nn
import torch.nn.functional as F

def _weight_drop(module, weights, dropout):
    for name_w in weights:
        w = getattr(module, name_w)
        del module._parameters[name_w]
        module.register_parameter(name_w + '_raw', nn.Parameter(w.data))

    original_forward = module.forward

    def forward(*args, **kwargs):
        for name_w in weights:
            raw_w = getattr(module, name_w + '_raw')
            w = F.dropout(raw_w, p=dropout, training=module.training)
            setattr(module, name_w, w)
        out = original_forward(*args, **kwargs)
        for name_w in weights:
            delattr(module, name_w)
        return out

    setattr(module, 'forward', forward)

class LSTM(torch.nn.LSTM):
    def __init__(self, *args, dropconnect=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        weights = ['weight_hh_l' + str(i) for i in range(self.num_layers)]
        _weight_drop(self, weights, dropconnect)

    def flatten_parameters(*args, **kwargs):
        # Learn from https://github.com/salesforce/awd-lstm-lm/blob/master/weight_drop.py
        # Replace flatten_parameters with nothing
        return

class Adapter(nn.Module):
    def __init__(self, d_model, d_adapter=64, bias=True):
        super().__init__()
        self.down_projection = nn.Linear(d_model, d_adapter, bias=bias)
        self.up_projection = nn.Linear(d_adapter, d_model, bias=bias)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        d = torch.relu(self.down_projection(x))
        x = x + self.up_projection(d)
        return self.norm(x)

class LSTWithAdapters(nn.Module):
    def __init__(self,  *args, hidden_size = 1024, dropconnect=0.0, num_layers=1, dropout=0.0, d_adapter=64, adapter_names=[], **kwargs):
        super().__init__()
        self.layers = []
        self.adapters = []
        for _ in range(num_layers):
            self.layers.append(LSTM(*args,hidden_size=hidden_size, dropconnect=dropconnect, **kwargs))
            adapters = {}
            for a in adapter_names:
                adapters[a] = Adapter(hidden_size, d_adapter)
            self.adapters.append(adapters)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adapter_name, mask=None, hid=None):
        was_packed = isinstance(x, nn.utils.rnn.PackedSequence)

        for idx, l, la in enumerate(zip(self.layers, self.adapters)):
            if was_packed and not isinstance(x, nn.utils.rnn.PackedSequence):
                lengths = mask.sum(-1); #lengths[0] = mask.size(1)
                x = torch._pack_padded_sequence(x, lengths.cpu(), batch_first=True)    
                x, hid = l(x, hid)
                x = torch._pad_packed_sequence(x, batch_first=True)[0]
            else:
                x, hid = l(x, hid)
            if idx + 1 < len(self.layers): #drop out except for the last layer
                x = self.dropout(x)
            a = la[adapter_name]
            x = a(x)
        if was_packed:
            lengths = mask.sum(-1); #lengths[0] = mask.size(1)
            x = torch._pack_padded_sequence(x, lengths.cpu(), batch_first=True)    
        return x, hid