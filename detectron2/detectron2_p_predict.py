# Copyright (c) 2020-2021 Nokia Corporation and/or its subsidiary(-ies). All rights reserved.
# This file is covered by the license agreement found in the file “license.txt” in the root of this project.

import struct
import numpy as np
import torch, torchvision
import torch.nn.functional as F
import detectron2.data.transforms as T
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
import detectron2
import os

from detectron2.utils.logger import setup_logger
import struct

setup_logger()

import os, json, cv2, random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling.meta_arch import GeneralizedRCNN

# import sys
import oid_mask_encoding

# import matplotlib.pyplot as plt

import argparse
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

_scale = 22.4838
#_min = -27.43
_min = -26.43

_scale_dsslic = 4
_min_dsslic = -140

_scale_resid = 2.4
_min_resid = -200


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


class PLayerPredictor():
    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image, features, do_postprocess: bool = True, ):
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}

            batched_inputs = [inputs]
            images = self.model.preprocess_image(batched_inputs)

            # x = torch.as_tensor([features]).cuda().float()
            if self.model.proposal_generator is not None:
                proposals, _ = self.model.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.model.device) for x in batched_inputs]

            results, _ = self.model.roi_heads(images, features, proposals, None)

            if do_postprocess:
                assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
                return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
            else:
                return results

qp = 43
parser = argparse.ArgumentParser()
parser.add_argument('--task_id', type=str, default='qp43', \
                    help='task id')
parser.add_argument('--image_dir', type=str, default='/media/data/ccr/OI5000_seg/', \
                    help='image directory') ####/media/data/ccr/OI5000_seg/                /media/data/liutie/VCM/OpenImageV6-5K/
parser.add_argument('--task', type=str, default='segmentation', \
                    help='task: detection or segmentation')
parser.add_argument('--input_file', type=str, default='/media/data/ccr/VCM/dataset/annotations_5k/segmentation_validation_input_5k.lst', \
                    help='input file that contains a list of image file names')        ###'/media/data/ccr/VCM/dataset/annotations_5k/segmentation_validation_input_5k_2.lst'
parser.add_argument('--yuv_dir', type=str, default='/media/data/ccr/VCM/feature_seg/', \
                    help='directory that containes (reconstructed) stem feature maps')     ##/media/data/ccr/VCM/dataset/recon_p_yuv/segmentation/qp43/yuv/'
parser.add_argument('--output_file', type=str, default=f'output_{qp}_coco.txt', \
                    help='prediction output file in OpenImages format')
args = parser.parse_args()

if args.task == 'detection':
    # model_cfg_name = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    # model_cfg_name = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
    model_cfg_name = "../configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
elif args.task == 'segmentation':
    model_cfg_name = "../configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x_vcm.yaml"
else:
    assert False, print("Unrecognized task:", args.task)

# construct detectron2 model
print('constructing detectron model ...')

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
# cfg.merge_from_file(model_zoo.get_config_file(model_cfg_name))
cfg.merge_from_file(model_cfg_name)
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_cfg_name)
p_predictor = PLayerPredictor(cfg)

# prediciton output
# output_fname = args.output_file
output_fname = f'output_{qp}_coco.txt'

coco_classes_fname = './data/coco_classes.txt'
with open(coco_classes_fname, 'r') as f:
    coco_classes = f.read().splitlines()

# of = open(output_fname, 'w')

# write header
# if args.task == 'detection':
#     of.write('ImageID,LabelName,Score,XMin,XMax,YMin,YMax\n')
# else:
#     of.write('ImageID,LabelName,Score,XMin,XMax,YMin,YMax,ImageWidth,ImageHeight,Mask\n')

################################################3333
# The same min/max values are obtained from the partially different image sets.
if args.task=="segmentation":
    global_max = 28.397470474243164
    global_min = -26.426828384399414
elif args.task=="detection":
    global_max = 28.397470474243164
    global_min = -26.426828384399414

# iterate all (reconstructed) stem files
def run_eval(qp):
    output_fname = f'output_{qp}_coco.txt'
    of = open(output_fname, 'w')
    of.write('ImageID,LabelName,Score,XMin,XMax,YMin,YMax,ImageWidth,ImageHeight,Mask\n')
    with open(args.input_file, 'r') as f:
        # with open(f"yuvinfo_{args.task}.txt", 'r') as f_yuvinfo:
        #     for idx, yuv_info in enumerate(f_yuvinfo.readlines()):
        #         # 1. Load YUV / Dequantisation / unpacking
        #         # 1-1 Load YUV
        #         orig_yuv_fname, width, height = yuv_info.split(',')
        #         width = int(width)
        #         height = int(height)
        #         flattened_plane = []
        #
        #
        #         if args.task_id=="UNCOMPRESSED":
        #             yuv_fname = orig_yuv_fname.replace('_UNCOMPRESSED','')
        #         else:
        #             yuv_fname = os.path.splitext(orig_yuv_fname)[0] + "_{}".format(args.task_id.replace("qp", "q")) + '.yuv'
        #
        #         yuv_full_fname = os.path.join(args.yuv_dir, yuv_fname)
        #         print(f'processing {idx}:{yuv_full_fname}...')

                #####################################################################################################
                # with open(yuv_full_fname, "rb") as f_yuv:
                #     bytes = f_yuv.read(2)
                #     while bytes:
                #         val = int.from_bytes(bytes, byteorder='little')
                #         flattened_plane.append(val)
                #         bytes = f_yuv.read(2)
                #     flattened_plane = np.array(flattened_plane)
                #     q_plane = flattened_plane.reshape(height, width)
                #
                # print("DEBUG1")
                # # 1-2 Dequantisation
                # bits = 10
                # steps = np.power(2, bits) - 1
                # scale = steps / (global_max - global_min)
                # dq_plane = q_plane / scale + global_min
                #
                # print("DEBUG2")
                # # 1-3 Unpacking
                # pyramid = {}
                # v2_h = int(height / 85 * 64)
                # v3_h = int(height / 85 * 80)
                # v4_h = int(height / 85 * 84)
                #
                # v2_blk = dq_plane[:v2_h, :]
                # v3_blk = dq_plane[v2_h:v3_h, :]
                # v4_blk = dq_plane[v3_h:v4_h, :]
                # v5_blk = dq_plane[v4_h:height, :]
                #
                # pyramid["p2"] = feature_slice(v2_blk, [v2_blk.shape[0] // 16, v2_blk.shape[1] // 16])
                # pyramid["p3"] = feature_slice(v3_blk, [v3_blk.shape[0] // 8, v3_blk.shape[1] // 32])
                # pyramid["p4"] = feature_slice(v4_blk, [v4_blk.shape[0] // 4, v4_blk.shape[1] // 64])
                # pyramid["p5"] = feature_slice(v5_blk, [v5_blk.shape[0] // 2, v5_blk.shape[1] // 128])
                #
                # pyramid["p2"] = torch.unsqueeze(pyramid["p2"], 0)
                # pyramid["p3"] = torch.unsqueeze(pyramid["p3"], 0)
                # pyramid["p4"] = torch.unsqueeze(pyramid["p4"], 0)
                # pyramid["p5"] = torch.unsqueeze(pyramid["p5"], 0)
                #
                # # print(pyramid["p2"])
                # # print(pyramid["p3"])
                # # print(pyramid["p4"])
                # # print(pyramid["p5"])
                #
                # pyramid["p6"] = F.max_pool2d(pyramid["p5"], kernel_size=1, stride=2, padding=0)
                #####################################################################################################
                #####################################################################################################
        with open(f"/media/data/ccr/VCM/dataset/annotations_5k/segmentation_validation_input_5k.lst", 'r') as f_yuvinfo:
            n=0
            ori = 0

            ################################################################################
            for idx in enumerate(f_yuvinfo.readlines()):
                orig_yuv_fname = idx[1].replace('\n', '')
                orig_yuv_fname = orig_yuv_fname.replace('jpg', 'png')
                png_dir = f'/media/data/ccr/VCM/feature_seg/{qp}_rec'
                png_fname = orig_yuv_fname
                fname = os.path.join(png_dir, png_fname)

                fname_ds_ori = fname.replace('rec', 'ds')
                fname_ds_rec = fname.replace('rec', 'ds_rec')
                fname_resid_rec = fname.replace('rec', 'resid_rec')
                fname_ori = fname.replace('rec', 'ori')
                fname_resid_ori = fname.replace('rec', 'resid')
                fname_p3_ds_rec = fname.replace('rec', 'p3_ds_rec')
                fname_p3_resid_rec = fname.replace('rec', 'p3_resid_rec')

                fname_ori_43 = fname_ori.replace('45', '43')
                if os.path.exists(fname):
                    features = feat2feat_p45(fname)
                elif os.path.exists(fname_ori):
                    features = feat2feat_p45(fname_ori)
                    ori+=1
                    print('1 ori added,ori=============================', ori)
                elif os.path.exists(fname_ori_43):
                    print(fname, 'missing!!!!!!!')
                    features = feat2feat_p45(fname_ori_43)
                else:
                    img_fname = os.path.join(args.image_dir, os.path.splitext(orig_yuv_fname)[0] + '.jpg')
                    img = cv2.imread(img_fname)
                    images = p_predictor.model.preprocess_image(img)
                    features = p_predictor.model.backbone(images.tensor)

                features_ds = feat2feat_onlyp2_8channels(fname_ds_rec)
                features['p2'] = features_ds['p2']
                features_p3_ds = feat2feat_onlyp3_8channels(fname_p3_ds_rec)
                features['p3'] = features_p3_ds['p3']

                upsample = torch.nn.Upsample(scale_factor=4, mode='bilinear')

                for p in features:
                    if p == 'p2':
                        comp_image = p_predictor.model.netG_8to256.forward(features[p])
                        up_image = upsample(comp_image)
                        input_fconcat = up_image
                        res = p_predictor.model.netG_256.forward(input_fconcat)
                        fake_image_f = res + up_image
                        features[p] = fake_image_f
                    if p == 'p3':
                        comp_image_p3 = p_predictor.model.netG_8to256.forward(features[p])
                        up_image_p3 = upsample(comp_image_p3)
                        input_fconcat_p3 = up_image_p3
                        res_p3 = p_predictor.model.netG_256.forward(input_fconcat_p3)
                        fake_image_f_p3 = res_p3 + up_image_p3
                        features[p] = fake_image_f_p3

                features_resid_rec = feat2feat_onlyp2_resid(fname_resid_rec)
                features['p2'] = features['p2'] + features_resid_rec['p2']
                features_resid_rec_p3 = feat2feat_onlyp2_resid(fname_p3_resid_rec)
                features['p3'] = features['p3'] + features_resid_rec_p3['p2']

                pyramid = features
                #####################################################################################################
                n+=1

                print("DEBUG3-------", n)
                # 2. Task performance evaluation
                img_fname = os.path.join(args.image_dir, os.path.splitext(orig_yuv_fname)[0] + '.jpg')
                # print(img_fname)
                img = cv2.imread(img_fname)
                if img is None:
                    img = cv2.imread(os.path.splitext(img_fname)[0] + '.png')
                assert img is not None, print(f'Image file not found: {img_fname}')
                # print(f'processing {img_fname}...')

                # print("DEBUG4")
                outputs = p_predictor(img, pyramid)[0]

                stemId = os.path.splitext(os.path.basename(orig_yuv_fname))[0]
                classes = outputs['instances'].pred_classes.to('cpu').numpy()
                scores = outputs['instances'].scores.to('cpu').numpy()
                bboxes = outputs['instances'].pred_boxes.tensor.to('cpu').numpy()
                H, W = outputs['instances'].image_size
                # convert bboxes to 0-1
                # detectron: x1, y1, x2, y2 in pixels
                bboxes = bboxes / [W, H, W, H]
                # OpenImage output x1, x2, y1, y2 in percentage
                bboxes = bboxes[:, [0, 2, 1, 3]]

                if args.task == 'segmentation':
                    masks = outputs['instances'].pred_masks.to('cpu').numpy()



                # print("DEBUG5")
                for ii in range(len(classes)):
                    coco_cnt_id = classes[ii]
                    class_name = coco_classes[coco_cnt_id]

                    if args.task == 'segmentation':
                        assert (masks[ii].shape[1] == W) and (masks[ii].shape[0] == H), \
                            print('Detected result does not match the input image size: ', stemId)

                    rslt = [stemId, class_name, scores[ii]] + \
                           bboxes[ii].tolist()

                    if args.task == 'segmentation':
                        rslt += \
                            [masks[ii].shape[1], masks[ii].shape[0], \
                             oid_mask_encoding.encode_binary_mask(masks[ii]).decode('ascii')]

                    o_line = ','.join(map(str, rslt))

                    of.write(o_line + '\n')
                print("DEBUG6")

            of.close()

for qp in [45, 43, 41, 39, 37, 35]:
    run_eval(qp)





