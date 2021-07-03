import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR
from . import few_shot
from collections.abc import Mapping
from flatten_dict import flatten
import pickle as pkl
from filelock import FileLock
import pandas as pd
import scipy.stats
from pathlib import Path
import yaml

_log_path = None
_log_name = None

def set_log_path(path):
    global _log_path
    _log_path = path


def set_log_name(name):
    global _log_name
    _log_name = name


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        if _log_name is not None:
            filename = _log_name
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


# def set_gpu(gpu):
#     print('set gpu:', gpu)
#     os.environ['CUDA_VISIBLE_DEVICES'] = gpu


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')):
            shutil.rmtree(path)
            os.makedirs(path)
        else:
            raise ValueError(f'The save path already exists: {path}')
    else:
        os.makedirs(path)


def time_str(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    if t >= 60:
        return '{:.1f}m'.format(t / 60)
    return '{:.1f}s'.format(t)


def compute_logits(feat, proto, metric='dot', temp=1.0):
    assert feat.dim() == proto.dim()

    if feat.dim() == 2:
        if metric == 'dot':
            logits = torch.mm(feat, proto.t())
        elif metric == 'cos':
            logits = torch.mm(F.normalize(feat, dim=-1),
                              F.normalize(proto, dim=-1).t())
        elif metric == 'sqr':
            logits = -(feat.unsqueeze(1) -
                       proto.unsqueeze(0)).pow(2).sum(dim=-1)

    elif feat.dim() == 3:
        if metric == 'dot':
            logits = torch.bmm(feat, proto.permute(0, 2, 1))
        elif metric == 'cos':
            logits = torch.bmm(F.normalize(feat, dim=-1),
                               F.normalize(proto, dim=-1).permute(0, 2, 1))
        elif metric == 'sqr':
            logits = -(feat.unsqueeze(2) -
                       proto.unsqueeze(1)).pow(2).sum(dim=-1)

    return logits * temp


def distance(proto, x_query, method, temp):
    if method == 'cos':
        proto = F.normalize(proto, dim=-1)
        x_query = F.normalize(x_query, dim=-1)
        metric = 'dot'
    elif method == 'sqr':
        metric = 'sqr'
    logits = compute_logits(
        x_query, proto, metric=metric, temp=temp)
    return logits


def compute_acc(logits, label, reduction='mean'):
    ret = (torch.argmax(logits, dim=1) == label).float()
    if reduction == 'none':
        return ret.detach()
    elif reduction == 'mean':
        return ret.mean().item()


def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot

def compute_n_trainable_params(model):
    tot = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if tot >= 1e6:
        return '{:.1f}M'.format(tot / 1e6)
    elif tot >= 1e3:
        return '{:.1f}K'.format(tot / 1e3)
    else:
        return '{}'.format(tot)


def make_optimizer(params, name, lr, weight_decay=None, milestones=None):
    if weight_decay is None:
        weight_decay = 0.
    if name == 'sgd':
        optimizer = SGD(params, lr, momentum=0.9, weight_decay=weight_decay)
    elif name == 'adam':
        optimizer = Adam(params, lr, weight_decay=weight_decay)
    if milestones:
        lr_scheduler = MultiStepLR(optimizer, milestones)
    else:
        lr_scheduler = None
    return optimizer, lr_scheduler


def visualize_dataset(dataset, name, writer, n_samples=16):
    demo = []
    for i in np.random.choice(len(dataset), n_samples):
        demo.append(dataset.convert_raw(dataset[i][0]))
    writer.add_images('visualize_' + name, np.stack(demo))
    writer.flush()


def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return h

def update_dictionary(original, patch):
    for k, v in patch.items():
        if isinstance(v, Mapping):
            original[k] = update_dictionary(original.get(k, {}), v)
        else:
            original[k] = v
    return original


# def update_dictionary_entry(original, new_key, new_value):
#
def update_dictionary_entry(original, key, new_value):
    updated = original.copy()
    for k, v in original.items():
        if isinstance(v, Mapping):
            updated[k] = update_dictionary_entry(v, key, new_value)
        else:
            if k == key:
                updated[k] = new_value
    return updated


def update_model_args(config, submodel):
    submodel_name = submodel["model"]
    submodel_args = submodel["model_args"]
    config["model_args"][submodel_name] = submodel_name
    config["model_args"][''.join([submodel_name,"_args"])] = submodel_args
    return config


def clear_args(inp):
    all_keys = list(inp.keys())
    for k in all_keys:
        if isinstance(inp[k], dict):
            inp[k] = clear_args(inp[k])
        if 'args' in k:
            inp[k]['name'] = inp[k.replace('_args', '')]
            inp[k.replace('_args', '')] = inp[k]
            del inp[k]
    return inp


