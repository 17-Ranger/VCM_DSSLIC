import argparse
import csv
import glob
import json
import os

import utils
from eval import DetectEval
# from eval_CCR import DetectEval

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ["DETECTRON2_DATASETS"] = f"/media/data/ccr/OI5000/27_resize"
# os.environ["DETECTRON2_DATASETS"] = '/media/data/liutie/VCM/OpenImageV6-5K'
os.environ["DETECTRON2_DATASETS"] = '/media/data/ccr/OI5000_seg'

# os.environ["DETECTRON2_ANNOTATIONS"] = "./dataset/annotations/instances_OpenImage_v6.json"
os.environ["DETECTRON2_ANNOTATIONS"] = "./dataset/annotations/instances_OpenImage_v6.json"

# os.environ["DETECTRON2_DATASETS"] = '/media/data/ccr/OI_3189'
# os.environ["DETECTRON2_ANNOTATIONS"] = "./dataset/annotations/instances_OpenImage_v6.json"

# os.environ["DETECTRON2_DATASETS"] = '/media/data/ccr/COCO-4660'
# os.environ["DETECTRON2_DATASETS"] = '/media/data/ccr/VCM/datasets/coco/val2017'
# os.environ["DETECTRON2_ANNOTATIONS"] = "./dataset/annotations/instances_COCO.json"

# os.environ["DETECTRON2_DATASETS"] = '/media/data/ccr/COCO_vtm/42_resize'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--index", default=45, type=int) #1
    parser.add_argument("-n", "--number", default=5000, type=int)
    # parser.add_argument("-m", "--mode", default='feature_coding')
    parser.add_argument("-m", "--mode", default='evaluation')

    args = parser.parse_args()
    set_idx = args.index
    number = args.number
    mode = args.mode

    with open(f"settings/{set_idx}.json", "r") as setting_json:
    # with open("settings/{set_idx}.json", "r") as setting_json:
        settings = json.load(setting_json)

    if settings["model_name"] == "x101":
        methods_eval = DetectEval(settings, set_idx)
        picklist = sorted(glob.glob(os.path.join(os.environ["DETECTRON2_DATASETS"], "*.jpg")))[:number]
        picklist = [utils.simple_filename(x) for x in picklist]
        methods_eval.prepare_part(picklist, data_name="pick")

    if mode == "feature_coding":
        filenames = methods_eval.feature_coding()

    elif mode == "feature_coding_resid":
        filenames = methods_eval.feature_coding_resid()

    elif mode == "evaluation":
        filenames = glob.glob(f"ccr_save/feature/{set_idx}_ori/*.png")
        # filenames = glob.glob('/media/data/liutie/VCM/OpenImageV6-5K')
        # filenames = glob.glob(f"feature/{set_idx}_ori/*.png")
        # filenames = glob.glob(f"/media/data/ccr/OI5000/27_resize/*.jpg")
        # filenames = glob.glob(f"/media/data/ccr/COCO_vtm/42_resize/*.jpg")
        # filenames = glob.glob(f"/media/data/liutie/VCM/rcnn/VCM_EE1.2_P-layer_feature_map_anchor_generation_137th_MPEG-VCM-main/feature/{set_idx}_rec/*.png")

        methods_eval.evaluation(filenames)

    elif mode == "evaluation_seg":
        filenames = glob.glob(f"ccr_save/feature/{set_idx}_ori/*.png")

        methods_eval.evaluation_seg(filenames)

    elif mode == "evaluation_offline":
        methods_eval.evaluation_offline()