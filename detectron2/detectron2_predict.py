# Copyright (c) 2020-2021 Nokia Corporation and/or its subsidiary(-ies). All rights reserved.
# This file is covered by the license agreement found in the file “license.txt” in the root of this project.

import torch, torchvision
import detectron2

from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import pandas as pd
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

#import sys
import oid_mask_encoding

import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', type=str, default='./', \
  help='image directory')
parser.add_argument('--task', type=str, default='detection', \
  help='task: detection or segmentation')
parser.add_argument('--input_file', type=str, default='input.lst', \
  help='input file that contains a list of image file names')
parser.add_argument('--output_file', type=str, default='output_coco.txt', \
  help='prediction output file in OpenImages format')

args = parser.parse_args()

if args.task == 'detection':
  #model_cfg_name = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
  model_cfg_name = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
elif args.task == 'segmentation':
  model_cfg_name = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
else:
  assert False, print("Unrecognized task:", args.task)

# construct detectron2 model
print('constructing detectron model ...')


cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file(model_cfg_name))
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_cfg_name)
predictor = DefaultPredictor(cfg)

# prediciton output
output_fname = args.output_file

##outut ID mapping file
#with open('class_id_coco_name_mapping.txt','w') as of:
#  of.write('\n'.join(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes))

coco_classes_fname = './data/coco_classes.txt'
with open(coco_classes_fname, 'r') as f:
  coco_classes = f.read().splitlines()

of = open(output_fname, 'w')

#write header
if args.task == 'detection':
    of.write('ImageID,LabelName,Score,XMin,XMax,YMin,YMax\n')
else:
    of.write('ImageID,LabelName,Score,XMin,XMax,YMin,YMax,ImageWidth,ImageHeight,Mask\n')


# iterate all input files
with open(args.input_file, 'r') as f:
  for img_fname in f.readlines():
    img_fname = os.path.join(args.image_dir, img_fname.strip())

    img = cv2.imread(img_fname)
    assert img is not None, print(f'Image file not found: {img_fname}')
    print(f'processing {img_fname}...')

    # generate ouptut
    outputs = predictor(img)
    imageId = os.path.splitext(os.path.basename(img_fname))[0]
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

    for ii in range(len(classes)):
      coco_cnt_id = classes[ii]
      class_name = coco_classes[coco_cnt_id]

      if args.task == 'segmentation':
        assert (masks[ii].shape[1]==W) and (masks[ii].shape[0]==H), \
          print('Detected result does not match the input image size: ', imageId)
      
      rslt = [imageId, class_name, scores[ii]] + \
        bboxes[ii].tolist()

      if args.task == 'segmentation':
        rslt += \
          [masks[ii].shape[1], masks[ii].shape[0], \
          oid_mask_encoding.encode_binary_mask(masks[ii]).decode('ascii')]

      o_line = ','.join(map(str,rslt))

      of.write(o_line + '\n')

of.close()

  
