# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import random
from turtle import forward

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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

    def forward(self, x):
        d = torch.relu(self.down_projection(x))
        return x + self.up_projection(d)

class LSTMWithAdapters(nn.Module):
    def __init__(self,  *args, input_size=1024, hidden_size = 1024, dropconnect=0.0, num_layers=1, dropout=0.0, d_adapter=64, adapter_names=[], **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        self.adapters = nn.ModuleList()
        self.normalization = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(LSTM(*args,input_size=input_size, hidden_size=hidden_size, dropconnect=dropconnect, **kwargs))

            input_size = 2 * hidden_size if kwargs['bidirectional'] else hidden_size
            adapters = nn.ModuleDict()
            for a in adapter_names:
                adapters[a] = Adapter(input_size, d_adapter)
            self.adapters.append(adapters)
            self.normalization.append(nn.LayerNorm(input_size))
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, adapter_name, mask=None, hid=None, enforce_sorted=True):
        was_packed = isinstance(x, nn.utils.rnn.PackedSequence)
        lengths = None
        for idx, (lstm, adapters, normalization) in enumerate(zip(self.layers, self.adapters, self.normalization)):
            if was_packed:
                x, hid = lstm(x, hid)
                x, lengths = pad_packed_sequence(x, batch_first=True)
            else:
                x, hid = lstm(x, hid)
            if idx + 1 < len(self.layers): #drop out except for the last layer
                x = self.dropout(x)
            a = adapters[adapter_name]
            x = a(x)
            x = normalization(x)
            if was_packed:
                #lengths = mask.sum(-1); #lengths[0] = mask.size(1)
                x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=enforce_sorted)    
        return x, hid

class ParameterGenerator(nn.Module):
    def __init__(self, d_model, d_adapter, num_layers, langs, emb_dim):
        super().__init__()
        self.embedding = nn.Embedding(len(langs), embedding_dim=emb_dim)
        self.layer_size = d_model * d_adapter * 2 #+ d_adapter + d_model
        self.d_model = d_model
        self.d_adapter = d_adapter
        self.num_layers = num_layers
        self.lang2id = dict((lang, idx) for idx, lang in enumerate(langs))
        w = torch.zeros(self.layer_size * num_layers, emb_dim)
        w = nn.init.xavier_uniform_(w)
        self.w = nn.parameter.Parameter(w)
        self.w_size = d_model * d_adapter

    def forward(self, lang):
        lang = self.lang2id[lang]
        lang = torch.LongTensor((lang,)).to(self.w.device)
        x = self.embedding(lang).view(-1)
        return torch.matmul(self.w, x)

    def forward_adapter_layer(self, x, layer, params):
        params = params[layer * self.layer_size: (layer + 1) * self.layer_size]       
        w_down = params[:self.w_size].view(self.d_adapter, self.d_model)
        w_up = params[self.w_size:2*self.w_size].view(self.d_model, self.d_adapter)
        #bias_down = params[2*self.w_size:2*self.w_size + self.d_adapter]
        #bias_up = params[2*self.w_size + self.d_adapter:]

        x = F.linear(x, w_down)#, bias_down)
        x = torch.relu(x)
        return F.linear(x, w_up)#, bias_up)


class LSTMWithAdaptersWithPrameterGenerator(nn.Module):
    def __init__(self,  *args, input_size=1024, hidden_size = 1024, dropconnect=0.0, num_layers=1, dropout=0.0, d_adapter=64, adapter_names=[], **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        self.adapters = ParameterGenerator(2 * hidden_size if kwargs['bidirectional'] else hidden_size, d_adapter, num_layers, adapter_names, 32)
        self.normalization = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(LSTM(*args,input_size=input_size, hidden_size=hidden_size, dropconnect=dropconnect, **kwargs))

            input_size = 2 * hidden_size if kwargs['bidirectional'] else hidden_size
            self.normalization.append(nn.LayerNorm(input_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adapter_name, mask=None, hid=None, enforce_sorted=True):
        was_packed = isinstance(x, nn.utils.rnn.PackedSequence)
        lengths = None
        adapter_params = self.adapters(adapter_name)
        for idx, (lstm, normalization) in enumerate(zip(self.layers, self.normalization)):
            if was_packed:
                x, hid = lstm(x, hid)
                x, lengths = pad_packed_sequence(x, batch_first=True)
            else:
                x, hid = lstm(x, hid)
            if idx + 1 < len(self.layers): #drop out except for the last layer
                x = self.dropout(x)
            x = self.adapters.forward_adapter_layer(x, idx, adapter_params)
            x = normalization(x)
            if was_packed:
                #lengths = mask.sum(-1); #lengths[0] = mask.size(1)
                x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=enforce_sorted)    
        return x, hid