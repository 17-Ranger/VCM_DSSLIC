#Copyright (c) Facebook, Inc. and its affiliates.
import os
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import move_device_like
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
from detectron2.layers.batch_norm import FrozenBatchNorm2d, get_norm

from ..backbone import Backbone, build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY

from torch.utils.tensorboard import SummaryWriter
import random
writer = SummaryWriter()

import functools
from torch.autograd import Variable
from PIL import Image
import math
import cv2
import torch.nn.functional as F

__all__ = ["GeneralizedRCNN", "ProposalNetwork"]
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#############################################################################################
# Functions
#############################################################################################
_scale = 22.4838
_min = -26.43

_scale_dsslic = 4
_min_dsslic = -140

_scale_resid = 2.4
_min_resid = -200

# _scale = 22.4838
# _min = -26.43
#
# _scale_dsslic = 5
# _min_dsslic = -120
#
# _scale_resid = 2.8
# _min_resid = -160

def quant_fix_resid(features):
    for name, pyramid in features.items():
        pyramid_q = (pyramid - _min_resid) * _scale_resid
        features[name] = pyramid_q
    return features

def dequant_fix_resid(x):
    return x.type(torch.float32) / _scale_resid + _min_resid

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

def feature_slice(image, shape):
    height = image.shape[0]
    width = image.shape[1]

    blk_height = shape[0]
    blk_width = shape[1]
    blk = []

    for y in range(height // blk_height):
        for x in range(width // blk_width):
            y_lower = y * blk_height
            y_upper = (y + 1) * blk_height
            x_lower = x * blk_width
            x_upper = (x + 1) * blk_width
            blk.append(image[y_lower:y_upper, x_lower:x_upper])
    feature = torch.from_numpy(np.array(blk)).cuda().float()
    return feature

def feat2feat(fname):
    pyramid = {}

    png = cv2.imread(fname, -1).astype(np.float32)
    vectors_height = png.shape[0]
    v2_h = int(vectors_height / 85 * 64)
    v3_h = int(vectors_height / 85 * 80)
    v4_h = int(vectors_height / 85 * 84)

    v2_blk = png[:v2_h, :]
    v3_blk = png[v2_h:v3_h, :]
    v4_blk = png[v3_h:v4_h, :]
    v5_blk = png[v4_h:vectors_height, :]

    pyramid["p2"] = feature_slice(v2_blk, [v2_blk.shape[0] // 16, v2_blk.shape[1] // 16 ])
    pyramid["p3"] = feature_slice(v3_blk, [v3_blk.shape[0] // 8 , v3_blk.shape[1] // 32 ])
    pyramid["p4"] = feature_slice(v4_blk, [v4_blk.shape[0] // 4 , v4_blk.shape[1] // 64 ])
    pyramid["p5"] = feature_slice(v5_blk, [v5_blk.shape[0] // 2 , v5_blk.shape[1] // 128])

    pyramid["p2"] = dequant_fix(pyramid["p2"])
    pyramid["p3"] = dequant_fix(pyramid["p3"])
    pyramid["p4"] = dequant_fix(pyramid["p4"])
    pyramid["p5"] = dequant_fix(pyramid["p5"])

    pyramid["p2"] = torch.unsqueeze(pyramid["p2"], 0)
    pyramid["p3"] = torch.unsqueeze(pyramid["p3"], 0)
    pyramid["p4"] = torch.unsqueeze(pyramid["p4"], 0)
    pyramid["p5"] = torch.unsqueeze(pyramid["p5"], 0)

    pyramid["p6"] = F.max_pool2d(pyramid["p5"], kernel_size=1, stride=2, padding=0)

    #加了下面这几句弄到cuda
    pyramid["p2"] = pyramid["p2"].cuda()
    pyramid["p3"] = pyramid["p3"].cuda()
    pyramid["p4"] = pyramid["p4"].cuda()
    pyramid["p5"] = pyramid["p5"].cuda()
    pyramid["p6"] = pyramid["p6"].cuda()

    return pyramid

def feat2feat_p345(fname):
    pyramid = {}
    png = cv2.imread(fname, -1).astype(np.float32)
    vectors_height = png.shape[0]
    v3_h = int(vectors_height / 21 * 16)
    v4_h = int(vectors_height / 21 * 20)

    v3_blk = png[:v3_h, :]
    v4_blk = png[v3_h:v4_h, :]
    v5_blk = png[v4_h:vectors_height, :]

    pyramid["p3"] = feature_slice(v3_blk, [v3_blk.shape[0] // 8 , v3_blk.shape[1] // 32 ])
    pyramid["p4"] = feature_slice(v4_blk, [v4_blk.shape[0] // 4 , v4_blk.shape[1] // 64 ])
    pyramid["p5"] = feature_slice(v5_blk, [v5_blk.shape[0] // 2 , v5_blk.shape[1] // 128])

    pyramid["p3"] = dequant_fix(pyramid["p3"])
    pyramid["p4"] = dequant_fix(pyramid["p4"])
    pyramid["p5"] = dequant_fix(pyramid["p5"])

    pyramid["p3"] = torch.unsqueeze(pyramid["p3"], 0)
    pyramid["p4"] = torch.unsqueeze(pyramid["p4"], 0)
    pyramid["p5"] = torch.unsqueeze(pyramid["p5"], 0)
    pyramid["p6"] = F.max_pool2d(pyramid["p5"], kernel_size=1, stride=2, padding=0)

    #加了下面这几句弄到cuda
    pyramid["p3"] = pyramid["p3"].cuda()
    pyramid["p4"] = pyramid["p4"].cuda()
    pyramid["p5"] = pyramid["p5"].cuda()
    pyramid["p6"] = pyramid["p6"].cuda()
    return pyramid

def feat2feat_onlyp2_resid(fname):
    pyramid = {}
    png = cv2.imread(fname, -1).astype(np.float32)
    pyramid["p2"] = feature_slice(png, [png.shape[0] // 16, png.shape[1] // 16])
    pyramid["p2"] = dequant_fix_resid(pyramid["p2"])
    pyramid["p2"] = torch.unsqueeze(pyramid["p2"], 0)
    #加了下面这几句弄到cuda
    pyramid["p2"] = pyramid["p2"].cuda()
    return pyramid

def feat2feat_onlyp2_8channels(fname):
    pyramid = {}
    png = cv2.imread(fname, -1).astype(np.float32)
    pyramid["p2"] = feature_slice(png, [png.shape[0] // 4, png.shape[1] // 2])
    pyramid["p2"] = dequant_fix_dsslic(pyramid["p2"])
    pyramid["p2"] = torch.unsqueeze(pyramid["p2"], 0)
    # 加了下面这几句弄到cuda
    pyramid["p2"] = pyramid["p2"].cuda()
    return pyramid

def feat2feat_onlyp3_8channels(fname):
    pyramid = {}
    png = cv2.imread(fname, -1).astype(np.float32)
    pyramid["p3"] = feature_slice(png, [png.shape[0] // 4, png.shape[1] // 2])
    pyramid["p3"] = dequant_fix_dsslic(pyramid["p3"])
    pyramid["p3"] = torch.unsqueeze(pyramid["p3"], 0)
    # 加了下面这几句弄到cuda
    pyramid["p3"] = pyramid["p3"].cuda()
    return pyramid

def feat2feat_p45(fname):
    pyramid = {}
    png = cv2.imread(fname, -1).astype(np.float32)
    vectors_height = png.shape[0]
    v4_h = int(vectors_height / 5 * 4)

    v4_blk = png[:v4_h, :]
    v5_blk = png[v4_h:vectors_height, :]

    pyramid["p4"] = feature_slice(v4_blk, [v4_blk.shape[0] // 16 , v4_blk.shape[1] // 16 ])
    pyramid["p5"] = feature_slice(v5_blk, [v5_blk.shape[0] // 8 , v5_blk.shape[1] // 32])

    pyramid["p4"] = dequant_fix(pyramid["p4"])
    pyramid["p5"] = dequant_fix(pyramid["p5"])

    pyramid["p4"] = torch.unsqueeze(pyramid["p4"], 0)
    pyramid["p5"] = torch.unsqueeze(pyramid["p5"], 0)
    pyramid["p6"] = F.max_pool2d(pyramid["p5"], kernel_size=1, stride=2, padding=0)

    #加了下面这几句弄到cuda
    pyramid["p4"] = pyramid["p4"].cuda()
    pyramid["p5"] = pyramid["p5"].cuda()
    pyramid["p6"] = pyramid["p6"].cuda()
    return pyramid




def get_psnr(img1, img2):
   mse = np.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def mkdirs(path):
    # if not os.path.exists(path):
    #     os.makedirs(path)
    os.makedirs(path, exist_ok=True)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_compG(input_nc, output_nc, ngf, n_downsample_global=3, norm='instance', gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    netcompG = CompGenerator(input_nc, output_nc, ngf, n_downsample_global, norm_layer)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        #netcompG.cuda(gpu_ids[1])
        #netcompG.cuda('cuda:1')
        #netcompG.cuda(1)
        netcompG.to(device)

    netcompG.apply(weights_init)
    return netcompG

def define_G(input_nc, output_nc, ngf, netG, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1,
             n_blocks_local=3, norm='instance', gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == 'global':
        netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
    elif netG == 'local':
        netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global,
                                  n_local_enhancers, n_blocks_local, norm_layer)
    else:
        raise('generator not implemented!')
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        #netG.cuda(gpu_ids[0])
        #netG.cuda(1)
        netG.to(device)
    netG.apply(weights_init)
    return netG

def define_G_8to256(input_nc, output_nc, ngf, netG, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1,
             n_blocks_local=3, norm='instance', gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == 'global':
        netG = GlobalGenerator_8to256(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
    elif netG == 'local':
        netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global,
                                  n_local_enhancers, n_blocks_local, norm_layer)
    else:
        raise('generator not implemented!')
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        #netG.cuda(gpu_ids[0])
        #netG.cuda(1)
        netG.to(device)
    netG.apply(weights_init)
    return netG

def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        #netD.cuda(gpu_ids[0])
        netD.to(device)
    netD.apply(weights_init)
    return netD

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)

class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9,
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers

        ###### global generator model #####
        ngf_global = ngf * (2 ** n_local_enhancers)
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global,
                                       norm_layer).model
        model_global = [model_global[i] for i in
                        range(len(model_global) - 3)]  # get rid of final convolution layers
        self.model = nn.Sequential(*model_global)

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers + 1):
            ### downsample
            ngf_global = ngf * (2 ** (n_local_enhancers - n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0),
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1),
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            model_upsample += [
                nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1),
                norm_layer(ngf_global), nn.ReLU(True)]

            ### final convolution
            if n == n_local_enhancers:
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]

            setattr(self, 'model' + str(n) + '_1', nn.Sequential(*model_downsample))
            setattr(self, 'model' + str(n) + '_2', nn.Sequential(*model_upsample))

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input):
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        output_prev = self.model(input_downsampled[-1])
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers + 1):
            model_downsample = getattr(self, 'model' + str(n_local_enhancers) + '_1')
            model_upsample = getattr(self, 'model' + str(n_local_enhancers) + '_2')
            input_i = input_downsampled[self.n_local_enhancers - n_local_enhancers]
            output_prev = model_upsample(model_downsample(input_i) + output_prev)
        return output_prev


class CompGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=3, norm_layer=nn.BatchNorm2d):
        super(CompGenerator, self).__init__()
        self.output_nc = output_nc

        #model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), nn.ReLU(True)]
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, input_nc, kernel_size=7, padding=0), norm_layer(input_nc),
                 nn.ReLU(True)]

        ####################################################################ccr added
        for i in range(5):
            model += [ResnetBlock(input_nc, padding_type='reflect', activation=nn.ReLU(True), norm_layer=norm_layer)]
        ####################################################################ccr added

        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            list1=[]
            list1.append(mult)
            model += [nn.Conv2d(int(input_nc / mult), int(input_nc / mult), kernel_size=3, stride=2, padding=1),
                      norm_layer(int(input_nc / mult)), nn.ReLU(True)]
            model += [nn.Conv2d(int(input_nc / mult), int(input_nc / (mult * 2)), kernel_size=3, stride=1, padding=1),
                      norm_layer(int(input_nc / (mult * 2))), nn.ReLU(True)]

        ####################################################################ccr added
        # model += [ResnetBlock(input_nc * mult * 2, padding_type='reflect', activation=nn.ReLU(True), norm_layer=norm_layer)]
        ####################################################################ccr added
        # model += [nn.ReflectionPad2d(3), nn.Conv2d(input_nc * mult * 2, input_nc, kernel_size=7, padding=0)] #nn.Tanh()
        # model += [ResnetBlock(input_nc,  padding_type='reflect', activation=nn.ReLU(True), norm_layer=norm_layer)]
        # model += [nn.ReflectionPad2d(3), nn.Conv2d(int(input_nc / (mult * 2)), ngf, kernel_size=7, padding=0)]  # nn.Tanh()
        for i in range(5):
            model += [ResnetBlock(int(input_nc / (mult * 2)), padding_type='reflect', activation=nn.ReLU(True), norm_layer=norm_layer)]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(int(input_nc / (mult * 2)), output_nc, kernel_size=7, padding=0)]  # nn.Tanh()
        ####################################################################ccr added
        # for i in range(2):
        #     model += [ResnetBlock(output_nc, padding_type='reflect', activation=nn.ReLU(True), norm_layer=norm_layer)]
        ####################################################################ccr added

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(GlobalGenerator, self).__init__()
        activation = nn.ReLU(True)

        # n_downsampling=0
        # input: 1x3xwxh
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        # output: 1x64xwxh
        # model = [nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]

        ### downsample / NIMA: instead of DS, we feed the downsampled_bic image (1/4)
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

            # after 4 downsampling
        # output: 1x128x240x248
        # output: 1x256x120x124
        # output: 1x256x60x62
        # output: 1x1024x30x31

        ### resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        # n_downsampling=1

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                      norm_layer(int(ngf * mult / 2)), activation]

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)] #nn.Tanh()
        # model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        # for i in range(1):
        #     model += [ResnetBlock(output_nc, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

class GlobalGenerator_8to256(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(GlobalGenerator_8to256, self).__init__()
        activation = nn.ReLU(True)

        # n_downsampling=0
        # input: 1x3xwxh
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        # output: 1x64xwxh
        # model = [nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]

        ### resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)] #nn.Tanh()
        # model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        # for i in range(1):
        #     model += [ResnetBlock(output_nc, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        # print(x.size())
        out = x + self.conv_block(x)
        return out

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        # tt
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False,
                 getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)
                #############################################################################################

@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """

        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

        ############################################################################20220519
        self.gpu_ids = [0,1,2]
        self.compG = define_compG(256, 8, 64, 2, norm='instance', gpu_ids=self.gpu_ids)
        #self.netG = define_G(8, 8, 64, 'global', 0, 9, 1, 3, 'instance', gpu_ids=self.gpu_ids)
        self.netG_256 = define_G(256, 256, 64, 'global', 0, 9, 1, 3, 'instance', gpu_ids=self.gpu_ids)
        self.netG_8to256 = define_G_8to256(8, 256, 64, 'global', 0, 6, 1, 3, 'instance', gpu_ids=self.gpu_ids)
        #self.compG_image = define_compG(3, 3, 64, 1, norm='instance', gpu_ids=self.gpu_ids)
        #self.netG_image = define_G(3, 3, 64, 'global', 3, 9, 1, 3, 'instance', gpu_ids=self.gpu_ids)
        #self.netD = define_D(3, 64, 3, 'instance', 'store_true', 2, getIntermFeat=False, gpu_ids=self.gpu_ids)
        #self.compG = define_compG(netG_input_nc, opt.output_nc, opt.ncf, opt.n_downsample_comp, norm=opt.norm, gpu_ids=self.gpu_ids)
        #self.netG = define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG, opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers, opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)
        #self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
        ############################################################################20220519
        #####ccr added


        for p in self.backbone.parameters():
            p.requires_grad = False
            FrozenBatchNorm2d.convert_frozen_batchnorm(self.backbone)

        for p in self.proposal_generator.parameters():
            p.requires_grad = False
            FrozenBatchNorm2d.convert_frozen_batchnorm(self.proposal_generator)

        for p in self.roi_heads.parameters():
            p.requires_grad = False
            FrozenBatchNorm2d.convert_frozen_batchnorm(self.roi_heads)
        ############################
        self.i_step_count = 0
        compressai_logdir = '/media/data/ccr/VCM/tensorboard/'
        mkdirs(compressai_logdir)
        self.belle_writer = SummaryWriter(log_dir=compressai_logdir)
        self.belle_savetensorboardfreq = 10

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)
        
        images = self.preprocess_image(batched_inputs)
        
        #batch_size = len(batched_inputs)
        #for i in range(batch_size):
        #    slice_image = images.tensor[i:i+1,:,:,:].clone()
        #    images_down = self.compG_image.forward(slice_image)
        #    upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        #    upimage = upsample(images_down)
        #    inputflabel = None
        #    inputfconcat = upimage
        #    res_image = self.netG_image.forward(inputfconcat)
        #    fake_image_f = res_image + upimage
        #    images.tensor[i:i+1,:,:,:] = fake_image_f


        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        
        features = self.backbone(images.tensor)

        #######################################################################################
        ##############################################
        maxP2 = float(features['p2'].max())
        minP2 = float(features['p2'].min())
        meanP2 = float(features['p2'].mean())
        ##############################################

        # ###############################################################################################################20220519
        for p in features:
            if p=='p2' or p=='p3':
                fake_image_f_GT = features[p]
                compG_input = features[p]
                comp_image = self.compG.forward(compG_input)

                comp_image_256 = self.netG_8to256.forward(comp_image)
                upsample = torch.nn.Upsample(scale_factor=4, mode='bilinear')
                up_image = upsample(comp_image_256)
                # fake_image_f = up_image

                input_fconcat = up_image
                res = self.netG_256.forward(input_fconcat)
                fake_image_f = res + up_image
                features[p] = fake_image_f
        ################################## 20220516 ccr added end

        ###########################################################################
        maxcomp_image = float(comp_image.max())
        mincomp_image = float(comp_image.min())
        meancomp_image = float(comp_image.mean())
        # maxcomp_image_256 = float(comp_image_256.max())
        # mincomp_image_256 = float(comp_image_256.min())
        # meancomp_image_256 = float(comp_image_256.mean())
        maxfake_image_f = float(fake_image_f.max())
        minfake_image_f = float(fake_image_f.min())
        meanfake_image_f = float(fake_image_f.mean())

        PSNR1 = fake_image_f_GT[0:1, :, :, :]
        PSNR2 = fake_image_f[0:1, :, :, :]
        PSNR11 = PSNR1.squeeze().cpu().detach().numpy()
        PSNR22 = PSNR2.squeeze().cpu().detach().numpy()
        PSNR = get_psnr(PSNR11, PSNR22)

        def to01(tensor,channel):
            return (tensor[0, channel:(channel + 1), :, :]-tensor[0, channel:(channel + 1), :, :].min())/(tensor[0, channel:(channel + 1), :, :].max()-tensor[0, channel:(channel + 1), :, :].min())
        # writer.add_scalar("oriP2", 'maxP2': maxP2)
        if (self.i_step_count % self.belle_savetensorboardfreq == 0):  # and (channel_idx == 0):
            i_select_channel = random.randint(0, 255)
            i_select_channel_8 = random.randint(0, 7)
            self.belle_writer.add_scalars("Max", {'max_P2': maxP2, 'max_compnet': maxcomp_image, 'max_finenet': maxfake_image_f}, global_step=self.i_step_count)
            self.belle_writer.add_scalars("Min", {'minP2': minP2, 'min_compnet': mincomp_image, 'min_finenet': minfake_image_f}, global_step=self.i_step_count)
            self.belle_writer.add_scalars("Mean", {'meanP2': meanP2, 'mean_compnet': meancomp_image,
                                                    'mean_finenet': meanfake_image_f}, global_step=self.i_step_count)


        self.i_step_count = self.i_step_count + 1

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        ######MSE P2 loss
        #line 2+1 -1-2
        l_l2 = torch.nn.MSELoss().to(device)
        loss_l2 = l_l2(up_image, fake_image_f_GT) / 50

        # loss_l2 = l_l2(fake_image_f, fake_image_f_GT) / 10000
        loss_l2_dict = {}
        loss_l2_dict['loss_l2'] = loss_l2
        ###print(loss_l2.size(), '-----------------loss_ls size')
        #print("train_loss:%8.4f, max/min_P2(GT): %8.4f/%8.4f max/min_P2(CompNet output): %8.4f/%8.4f, max/min_P2(FineNet output): %8.4f/%8.4f" % (loss_l2, torch.max(fake_image_f_GT), torch.min(fake_image_f_GT),torch.max(comp_image), torch.min(comp_image), torch.max(fake_image_f), torch.min(fake_image_f)))
        #print('max/min_P2(GT): %8.4f/%8.4f max/min_P2(output): %8.4f/%8.4f' %(torch.max(fake_image_f_GT), torch.min(fake_image_f_GT), torch.max(fake_image_f), torch.min(fake_image_f)))

        #print(detector_losses)
        #print(proposal_losses)

        ###print(loss_l2_dict)
        ##print(detector_losses.size(), '-------------------detector_losses size')
        ##print(proposal_losses.size(), '-------------------proposal_losses size')
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        losses.update(loss_l2_dict)
        print(losses)
        return losses

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)


        #########################################################################################################
        # qp = 41
        # # orig_yuv_fname = batched_inputs[0]['file_name']
        # # orig_yuv_fname = orig_yuv_fname.replace('datasets/coco/val2017/', '')
        # # orig_yuv_fname = orig_yuv_fname.replace('jpg', 'png')
        #
        # # orig_yuv_fname = '000000027620.png'
        # orig_yuv_fname = '0ac51477636a6933.png'
        # # png_dir = f'/media/data/ccr/VCM/feature_COCOPA/{qp}_rec'
        # png_dir = f'/media/data/ccr/VCM/feature_OIPA/{qp}_rec'
        # png_fname = orig_yuv_fname
        # fname = os.path.join(png_dir, png_fname)
        # if os.path.exists(fname):
        #     features = feat2feat(fname)
        #########################################################################################################
        # qp = 35
        # # orig_yuv_fname = batched_inputs[0]['file_name']
        # # orig_yuv_fname = orig_yuv_fname.replace('datasets/coco/val2017/', '')
        # # orig_yuv_fname = orig_yuv_fname.replace('jpg', 'png')
        #
        # orig_yuv_fname = '86463a5a7dcb1a69.png'
        #
        # # png_dir = f'/media/data/ccr/VCM/feature_416/feature/{qp}_rec'
        # # png_dir_35 = f'/media/data/ccr/VCM/feature_416/feature/35_rec'
        # png_dir = f'/media/data/ccr/VCM/feature_seg/{qp}_rec'
        # png_fname = orig_yuv_fname
        # fname = os.path.join(png_dir, png_fname)
        #
        # fname_ds_rec = fname.replace('rec', 'ds_rec')
        # fname_resid_rec = fname.replace('rec', 'resid_rec')
        # fname_ori = fname.replace('rec', 'ori')
        # fname_p3_ds_rec = fname.replace('rec', 'p3_ds_rec')
        # fname_p3_resid_rec = fname.replace('rec', 'p3_resid_rec')
        #
        # print(fname)
        # if os.path.exists(fname):
        #     features = feat2feat_p45(fname)
        # else:
        #     features = self.backbone(images.tensor)
        # features_ds = feat2feat_onlyp2_8channels(fname_ds_rec)
        # features['p2'] = features_ds['p2']
        # features_p3_ds = feat2feat_onlyp3_8channels(fname_p3_ds_rec)
        # features['p3'] = features_p3_ds['p3']
        #
        # upsample = torch.nn.Upsample(scale_factor=4, mode='bilinear')
        #
        # for p in features:
        #     if p == 'p2':
        #         comp_image = self.netG_8to256.forward(features[p])
        #         up_image = upsample(comp_image)
        #         input_fconcat = up_image
        #         res = self.netG_256.forward(input_fconcat)
        #         fake_image_f = res + up_image
        #         # fake_image_f = up_image
        #         features[p] = fake_image_f
        #     if p == 'p3':
        #         comp_image_p3 = self.netG_8to256.forward(features[p])
        #         up_image_p3 = upsample(comp_image_p3)
        #         input_fconcat_p3 = up_image_p3
        #         res_p3 = self.netG_256.forward(input_fconcat_p3)
        #         fake_image_f_p3 = res_p3 + up_image_p3
        #         # fake_image_f_p3 = up_image_p3
        #         features[p] = fake_image_f_p3
        #
        # features_resid_rec = feat2feat_onlyp2_resid(fname_resid_rec)
        # features['p2'] = features['p2'] + features_resid_rec['p2']
        # features_resid_rec_p3 = feat2feat_onlyp2_resid(fname_p3_resid_rec)
        # features['p3'] = features['p3'] + features_resid_rec_p3['p2']
        #########################################################################################################

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results


@META_ARCH_REGISTRY.register()
class ProposalNetwork(nn.Module):
    """
    A meta architecture that only predicts object proposals.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)

    def forward(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)
        print(features['p5'].size(), '----------------------------------------------------------------------ProposalNetwork features')

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results
