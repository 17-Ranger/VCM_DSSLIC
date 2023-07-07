import numpy as np
import torch

# _min = -23.1728
# _max = 20.3891

# _scale = 23.4838
# _min = -23.1728

_scale = 22.4838
_min = -26.43

_scale_dsslic = 4
_min_dsslic = -140

_scale_resid = 2.4
_min_resid = -200

def quant_fix(features):
    for name, pyramid in features.items():
        pyramid_q = (pyramid-_min) * _scale
        features[name] = pyramid_q
    return features

def dequant_fix(x):
    return x.type(torch.float32)/_scale + _min

def quant_fix_dsslic(features):
    for name, pyramid in features.items():
        pyramid_q = (pyramid - _min_dsslic) * _scale_dsslic
        features[name] = pyramid_q
    return features

def dequant_fix_dsslic(x):
    return x.type(torch.float32) / _scale_dsslic + _min_dsslic

def quant_fix_resid(features):
    for name, pyramid in features.items():
        pyramid_q = (pyramid - _min_resid) * _scale_resid
        features[name] = pyramid_q
    return features

def dequant_fix_resid(x):
    return x.type(torch.float32) / _scale_resid + _min_resid