import os
import json
import csv
import cv2
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    Keypoints,
    PolygonMasks,
    RotatedBoxes,
    polygons_to_bitmask,
)


def get_openimg_dicts(root_path, training=True):
    dataset_dicts = []
    idx = 0
    if training:
        save_path = root_path + 'train/'
        if os.path.exists(save_path + '/openimg_anno.json'):
            with open(save_path + '/openimg_anno.json', "r") as fp:
                print("Load existing json files ...")
                dataset_dicts = json.load(fp)
        else:
            anno_path = root_path + 'train/annotation/train_GT47_small.csv'
            imgs_path = root_path + 'train/OpenImage_smaller'
            for root, dirs, files in os.walk(imgs_path):
                for file_name in files:
                    record = {}
                    imagename = file_name
                    # print('imagename',imagename)
                    filename = os.path.join(imgs_path, imagename)
                    # print(filename)
                    height, width = cv2.imread(filename).shape[:2]
                    record["file_name"] = imgs_path + '/' + imagename
                    record["height"] = height
                    record["width"] = width
                    record["image_id"] = idx
                    print(filename)
                    # seed_num = 2 * frame_num
                    # record["random_seed"] = int(idx / seed_num)
                    idx += 1
                    objs = get_annos_dicts(imagename, anno_path, width, height)
                    record["annotations"] = objs
                    dataset_dicts.append(record)
            with open(save_path + '/openimg_anno.json', "w") as fp:
                print("Now saving imgs ...")
                json.dump(dataset_dicts, fp)
    return dataset_dicts


def get_annos_dicts(img_name, anno_path, w, h):
    annos_dicts = []
    with open(anno_path) as f:
        annos_all = csv.reader(f)
        for row in annos_all:
            if row[0] == img_name[:-4]:
                bbox1 = round(float(row[2]) * w, 2)
                bbox2 = round(float(row[4]) * h, 2)
                bbox3 = round((float(row[3]) - float(row[2])) * w, 2)
                bbox4 = round((float(row[5]) - float(row[4])) * h, 2)
                obj = {
                    "bbox": [bbox1, bbox2, bbox3, bbox4],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "segmentation": None,
                    "category_id": id_map(row[1]),
                    "iscrowd": int(row[6]),
                }
                annos_dicts.append(obj)
    if len(annos_dicts) == 0:
        print('No object: --------------------------------> ', img_name)
    return annos_dicts

def get_openimg_seg_dicts(root_path, training=True):
    dataset_dicts = []
    idx = 0
    if training:
        save_path = root_path + 'train/'
        if os.path.exists(save_path + '/openimg_seg_anno.json'):
            with open(save_path + '/openimg_seg_anno.json', "r") as fp:
                print("Load existing json files ...")
                dataset_dicts = json.load(fp)
        else:
            anno_path = root_path + 'train/annotation/train_GT47_small.csv'
            imgs_path = root_path + 'train/OpenImage_smaller'
            for root, dirs, files in os.walk(imgs_path):
                for file_name in files:
                    record = {}
                    imagename = file_name
                    # print('imagename',imagename)
                    filename = os.path.join(imgs_path, imagename)
                    # print(filename)
                    height, width = cv2.imread(filename).shape[:2]
                    record["file_name"] = imgs_path + '/' + imagename
                    record["height"] = height
                    record["width"] = width
                    record["image_id"] = idx
                    print(filename)
                    # seed_num = 2 * frame_num
                    # record["random_seed"] = int(idx / seed_num)
                    idx += 1
                    objs = get_annos_dicts(imagename, anno_path, width, height)
                    record["annotations"] = objs
                    dataset_dicts.append(record)
            with open(save_path + '/openimg_seg_anno.json', "w") as fp:
                print("Now saving imgs ...")
                json.dump(dataset_dicts, fp)
    return dataset_dicts


def get_seg_annos_dicts(img_name, anno_path, w, h):
    annos_dicts = []
    with open(anno_path) as f:
        annos_all = csv.reader(f)
        for row in annos_all:
            if row[0] == img_name[:-4]:
                bbox1 = round(float(row[2]) * w, 2)
                bbox2 = round(float(row[4]) * h, 2)
                bbox3 = round((float(row[3]) - float(row[2])) * w, 2)
                bbox4 = round((float(row[5]) - float(row[4])) * h, 2)
                obj = {
                    "bbox": [bbox1, bbox2, bbox3, bbox4],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "segmentation": None,
                    "category_id": id_map(row[1]),
                    "iscrowd": int(row[6]),
                }
                annos_dicts.append(obj)
    if len(annos_dicts) == 0:
        print('No object: --------------------------------> ', img_name)
    return annos_dicts

def id_map(label):
    if label == 'person':
        return 0
    if label == 'car':
        return 2
    if label == 'motorcycle':
        return 3
    if label == 'airplane':
        return 4
    if label == 'bus':
        return 5
    if label == 'train':
        return 6
    if label == 'truck':
        return 7
    if label == 'boat':
        return 8
    if label == 'traffic_light':
        return 9
    if label == 'bird':
        return 14
    if label == 'cat':
        return 15
    if label == 'dog':
        return 16
    if label == 'horse':
        return 17
    if label == 'sheep':
        return 18
    if label == 'elephant':
        return 20
    if label == 'zebra':
        return 22
    if label == 'giraffe':
        return 23
    if label == 'backpack':
        return 24
    if label == 'handbag':
        return 26
    if label == 'tie':
        return 27
    if label == 'suitcase':
        return 28
    if label == 'skateboard':
        return 36
    if label == 'surfboard':
        return 37
    if label == 'tennis_rack':
        return 38
    if label == 'bottle':
        return 39
    if label == 'knife':
        return 43
    if label == 'spoon':
        return 44
    if label == 'bowl':
        return 45
    if label == 'apple':
        return 47
    if label == 'sandwich':
        return 48
    if label == 'orange':
        return 49
    if label == 'broccoli':
        return 50
    if label == 'carrot':
        return 51
    if label == 'pizza':
        return 53
    if label == 'cake':
        return 55
    if label == 'couch':
        return 57
    if label == 'toilet':
        return 61
    if label == 'laptop':
        return 63
    if label == 'book':
        return 73
    if label == 'clock':
        return 74
    if label == 'vase':
        return 75
    if label == 'donut':
        return 54
    if label == 'cup':
        return 41
    if label == 'cell_phone':
        return 67
    if label == 'keyboard':
        return 66
    if label == 'cow':
        return 19
    if label == 'sports_ball':
        return 32

root_path = "/media/data/ccr/OIdataset/"

# test = get_openimg_dicts(root_path)
# test_coco = get_coco_dicts()
print('1')
for d in ["train", "val"]:
    # # register training and testing together
    print('start')
    DatasetCatalog.register("openimage2022_" + d,
                            lambda d=d: get_openimg_dicts(root_path))
    #
    # DatasetCatalog.register("coco_" + d, lambda d=d: get_coco_dicts())

    # # can also use metadata for visulising or something fancy
    # MetadataCatalog.get("openimg_" + d).set(thing_classes=["people", "car", "and so on"])
