<<<<<<< HEAD
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo

import cv2
import numpy as np
import utils
import json
import os
import glob
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator

with open(f"settings/45.json", "r") as setting_json:
    # with open("settings/{set_idx}.json", "r") as setting_json:
    settings = json.load(setting_json)

def fname_5000(imagePath):
    ilename_base = os.path.basename(imagePath)
    filename_noext = ilename_base[:-4]
    return filename_noext

class Detector:
    def __init__(self):
        # self.cfg = get_cfg()
        self.settings = settings
        self.model, self.cfg = utils.model_loader(settings)  # load模型进来

        # self.data_loader = build_detection_test_loader(self.cfg, data_name)
        # self.evaluator = COCOEvaluator(data_name, self.cfg, False)
        # self.evaluator.reset()

        # self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x_vcm.yaml"))
        # self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
        self.cfg.MODEL.WEIGHTS = '/media/data/ccr/VCM/output/model_mask_final.pth'
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.cfg.MODEL.DEVICE = "cuda"
        self.predictor = DefaultPredictor(self.cfg)

    def onImage(self, imagePath):
        image = cv2.imread(imagePath)
        predictions = self.predictor(image)
        # viz = Visualizer(image, metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), instance_mode = ColorMode.IMAGE_BW)
        viz = Visualizer(image, metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
        output = viz.draw_instance_predictions(predictions["instances"].to('cpu'))
        fname = fname_5000(imagePath)
        cv2.imwrite(f"mask_output/{fname}_Result_mask.jpg", output.get_image())
        cv2.waitKey((0))
        print(output)
        print('1')

detector = Detector()

# filenames = glob.glob(f"/media/data/ccr/OI5000_seg/*.jpg")
# for i in filenames:
#     temp_i = i
 detector.onImage(f"/media/data/ccr/OI5000_seg/0ac51477636a6933.jpg")
# detector.onImage(f"/media/data/ccr/OI5000_seg/86463a5a7dcb1a69.jpg")
# detector.onImage(f"/media/data/ccr/OI5000/32_resize/0ac51477636a6933.jpg")
# detector.onImage(f"/media/data/ccr/OI5000/42_resize/86463a5a7dcb1a69.jpg")











=======
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo

import cv2
import numpy as np
import utils
import json
import os
import glob
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator

with open(f"settings/45.json", "r") as setting_json:
    # with open("settings/{set_idx}.json", "r") as setting_json:
    settings = json.load(setting_json)

def fname_5000(imagePath):
    ilename_base = os.path.basename(imagePath)
    filename_noext = ilename_base[:-4]
    return filename_noext

class Detector:
    def __init__(self):
        # self.cfg = get_cfg()
        self.settings = settings
        self.model, self.cfg = utils.model_loader(settings)  # load模型进来

        # self.data_loader = build_detection_test_loader(self.cfg, data_name)
        # self.evaluator = COCOEvaluator(data_name, self.cfg, False)
        # self.evaluator.reset()

        # self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x_vcm.yaml"))
        # self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
        self.cfg.MODEL.WEIGHTS = '/media/data/ccr/VCM/output/model_mask_final.pth'
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.cfg.MODEL.DEVICE = "cuda"
        self.predictor = DefaultPredictor(self.cfg)

    def onImage(self, imagePath):
        image = cv2.imread(imagePath)
        predictions = self.predictor(image)
        # viz = Visualizer(image, metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), instance_mode = ColorMode.IMAGE_BW)
        viz = Visualizer(image, metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
        output = viz.draw_instance_predictions(predictions["instances"].to('cpu'))
        fname = fname_5000(imagePath)
        cv2.imwrite(f"mask_output/{fname}_Result_mask.jpg", output.get_image())
        cv2.waitKey((0))
        print(output)
        print('1')

detector = Detector()

# filenames = glob.glob(f"/media/data/ccr/OI5000_seg/*.jpg")
# for i in filenames:
#     temp_i = i
 detector.onImage(f"/media/data/ccr/OI5000_seg/0ac51477636a6933.jpg")
# detector.onImage(f"/media/data/ccr/OI5000_seg/86463a5a7dcb1a69.jpg")
# detector.onImage(f"/media/data/ccr/OI5000/32_resize/0ac51477636a6933.jpg")
# detector.onImage(f"/media/data/ccr/OI5000/42_resize/86463a5a7dcb1a69.jpg")











>>>>>>> be0df11a27051c4085a72cd9a58dcd1fea1d076a
