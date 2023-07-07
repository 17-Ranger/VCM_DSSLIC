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


class StemPredictor():
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

    def __call__(self, original_image, stem, do_postprocess: bool = True, ):
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
            # print(batched_inputs['file_name'])
            # VisImage(input_images_cpu[0][:, :, permute]).save("{}/input/{}.png".format(self.cfg.OUTPUT_DIR,
            #                                                                            os.path.splitext(
            #                                                                              os.path.basename(
            #                                                                                batched_inputs[0][
            #                                                                                  'file_name']))[0]))

            #######################START: STEM-TO-RESNET OUTPUT###########################
            outputs = {}
            x = torch.as_tensor([stem]).cuda().float()
            if "stem" in self.model.backbone.bottom_up._out_features:
                outputs["stem"] = x
            for name, stage in zip(self.model.backbone.bottom_up.stage_names, self.model.backbone.bottom_up.stages):
                x = stage(x)
                if name in self.model.backbone.bottom_up._out_features:
                    outputs[name] = x
            if self.model.backbone.bottom_up.num_classes is not None:
                x = self.model.backbone.bottom_up.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.model.backbone.bottom_up.linear(x)
                if "linear" in self.model.backbone.bottom_up._out_features:
                    outputs["linear"] = x
            bottom_up_features = outputs
            #######################END: STEM-TO-RESNET OUTPUT###########################

            #######################START: REST OF FPN#########################
            results = []
            prev_features = self.model.backbone.lateral_convs[0](
                bottom_up_features[self.model.backbone.in_features[-1]])
            results.append(self.model.backbone.output_convs[0](prev_features))

            # Reverse feature maps into top-down order (from low to high resolution)
            for idx, (lateral_conv, output_conv) in enumerate(
                    zip(self.model.backbone.lateral_convs, self.model.backbone.output_convs)
            ):
                # Slicing of ModuleList is not supported https://github.com/pytorch/pytorch/issues/47336
                # Therefore we loop over all modules but skip the first one
                if idx > 0:
                    features = self.model.backbone.in_features[-idx - 1]
                    features = bottom_up_features[features]
                    top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode="nearest")
                    lateral_features = lateral_conv(features)
                    prev_features = lateral_features + top_down_features
                    if self.model.backbone._fuse_type == "avg":
                        prev_features /= 2
                    results.insert(0, output_conv(prev_features))

            if self.model.backbone.top_block is not None:
                if self.model.backbone.top_block.in_feature in bottom_up_features:
                    top_block_in_feature = bottom_up_features[self.model.backbone.top_block.in_feature]
                else:
                    top_block_in_feature = results[
                        self.model.backbone._out_features.index(self.model.backbone.top_block.in_feature)]
                results.extend(self.model.backbone.top_block(top_block_in_feature))
            assert len(self.model.backbone._out_features) == len(results)
            features = {f: res for f, res in zip(self.model.backbone._out_features, results)}
            #######################END: REST OF FPN#########################

            # features = self.backbone(stem)
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


parser = argparse.ArgumentParser()
parser.add_argument('--task_id', type=str, default='0', \
                    help='task id')
parser.add_argument('--image_dir', type=str, default='./', \
                    help='image directory')
parser.add_argument('--task', type=str, default='detection', \
                    help='task: detection or segmentation')
parser.add_argument('--input_file', type=str, default='input.lst', \
                    help='input file that contains a list of image file names')
parser.add_argument('--qp_bin_dir', type=str, default='.', \
                    help='directory that containes compressed bin files')
parser.add_argument('--output_bpp_file', type=str, default='output_bpp.txt', \
                    help='prediction output file in OpenImages format')
args = parser.parse_args()

# bpp output
output_bpp_fname = args.output_bpp_file
of = open(output_bpp_fname, 'w')

# iterate all (reconstructed) stem files

with open(f"yuvinfo_{args.task}.txt", 'r') as f_yuvinfo:
    total_pxls = 0
    total_bits = 0

    for idx, yuv_info in enumerate(f_yuvinfo.readlines()):
        orig_yuv_fname, _, _ = yuv_info.split(',')
        bin_fname = os.path.splitext(orig_yuv_fname)[0] + "_{}".format(args.task_id.replace("qp", "q")) + '.bin'
        bin_full_fname = os.path.join(args.qp_bin_dir, bin_fname)
        print(f'processing {idx}:{bin_full_fname}...')
        file_size_in_bits = os.path.getsize(bin_full_fname) * 8
        total_bits += file_size_in_bits


        img_fname = os.path.join(args.image_dir, os.path.splitext(orig_yuv_fname)[0] + '.jpg')
        # print(img_fname)
        img = cv2.imread(img_fname)
        if img is None:
            img = cv2.imread(os.path.splitext(img_fname)[0] + '.png')
        assert img is not None, print(f'Image file not found: {img_fname}')
        # print(img.shape)
        h, w, _ = img.shape
        size = h * w
        total_pxls += size


bpp = total_bits / total_pxls
print('total_bits: {}'.format(total_bits))
print('total_pxls: {}'.format(total_pxls))
print('bpp: {}'.format(bpp))
of.write('total_bits: {}\n'.format(total_bits))
of.write('total_pxls: {}\n'.format(total_pxls))
of.write('bpp: {}\n'.format(bpp))
of.close()














