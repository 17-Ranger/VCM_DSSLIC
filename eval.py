import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
from torch import nn
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator
from detectron2.data import DatasetCatalog
from tqdm import tqdm
import cv2
import numpy as np
import torch.nn.functional as F
import subprocess

import utils
from quantizer import quant_fix, dequant_fix
from quantizer import quant_fix_dsslic, dequant_fix_dsslic
from quantizer import quant_fix_resid, dequant_fix_resid
from VTM_encoder import run_vtm
from VTM_encoder_resid import run_vtm_resid
from cvt_detectron_coco_oid import conversion
import scipy.io as sio

import Anchors.oid_mask_encoding as oid_mask_encoding
import time

class Eval:
    def __init__(self, settings, index) -> None:
        self.settings = settings
        self.set_idx = index
        self.VTM_param = settings["VTM"]
        self.model, self.cfg = utils.model_loader(settings) #load模型进来
        self.prepare_dir()

        self.DSSLIC_mode = 'split'
        self.P3 = False
        self.nofinenet = False
        self.resid_eval = True
        self.ds = 4
        self.channel8 = True
        self.channel16 = False
        self.finenet_output = False
        self.resid_8c = False
        self.P345_ori = False
        self.P2_ori = False #'vvc' for reading image, 'bb' for backbone, False for no ori, 'allo' for all0,
        self.P2_all0 = False
        self.P3_net = True
        self.ori_eval = False
        self.COCO = False
        self.MIF = False
        self.seg = True

        utils.print_settings(settings, index)
        self.pixel_num = settings["pixel_num"]

    def prepare_dir(self):
        os.makedirs(f"info/{self.set_idx}", exist_ok=True)
        os.makedirs(f"feature/{self.set_idx}_ori", exist_ok=True)
        os.makedirs(f"feature/{self.set_idx}_ds", exist_ok=True)
        os.makedirs(f"feature/{self.set_idx}_resid", exist_ok=True)
        os.makedirs(f"output", exist_ok=True)
        os.makedirs(f"feature/{self.set_idx}_p3_ds", exist_ok=True)
        os.makedirs(f"feature/{self.set_idx}_p3_resid", exist_ok=True)
        # os.makedirs(f"feature/{self.set_idx}_finenet", exist_ok=True)

    def forward_front(self, inputs, images, features):
        proposals, _ = self.model.proposal_generator(images, features, None)
        results, _ = self.model.roi_heads(images, features, proposals, None)
        return self.model._postprocess(results, inputs, images.image_sizes)

    def feature_coding(self):   
        print("Saving features maps...")
        filenames = []
        start_time = time.clock()
        with tqdm(total=len(self.data_loader.dataset)) as pbar:
            for inputs in iter(self.data_loader):
                #自己加入的5行，断了之后重新跑，提过feature的不用再提
                fname_temp = utils.simple_filename(inputs[0]["file_name"])
                fea_path = f"feature/{self.set_idx}_ori/"
                # if os.path.isfile(f"{fea_path}{fname_temp}.png"):
                #     print(f"feature extraction: {fname_temp} skip (exist)")
                #     continue
                filenames.append(self._feature_coding(inputs)) #inputs是filename 15d64d, height 680, width 1024, image_id 19877和image这个tensor [3, 800, 1205] uint8 大于1
                pbar.update()
        end_time = time.clock()
        run_time = end_time - start_time
        print('Run time for each', run_time/10)
        print("runvtm---------------------runvtmrunvtmrunvtmrunvtmrunvtmrunvtmrunvtm")
        run_vtm(f"feature/{self.set_idx}_ori", self.VTM_param["QP"], self.VTM_param["threads"])
        return filenames

    def _feature_coding(self, inputs):
        images = self.model.preprocess_image(inputs) #images: device cpu, image_sizes [800, 1205] tensor [1, 3, 800, 1216] torch.float32 cpu
        features = self.model.backbone(images.tensor)
        #################################ccr added

        for p in features:
            if p == 'p2':
                compG_input = features[p]
                comp_image = self.model.compG.forward(compG_input)

                if self.DSSLIC_mode == 'split':
                    features_ds = features.copy()
                    features_ds["p2"] = comp_image
                    ds_feat = quant_fix_dsslic(features_ds.copy())
                if self.DSSLIC_mode == 'together':
                    features["p2"] = comp_image
                image_feat = quant_fix(features.copy())

            if self.P3_net == True:
                if p == 'p3':
                    compG_input_3 = features[p]
                    comp_image_3 = self.model.compG.forward(compG_input_3)

                    if self.DSSLIC_mode == 'split':
                        features_ds_3 = features.copy()
                        features_ds_3["p3"] = comp_image_3
                        ds_feat_3 = quant_fix_dsslic(features_ds_3.copy())

        # #################################ccr added
        fname = utils.simple_filename(inputs[0]["file_name"])
        fname_feat = f"feature/{self.set_idx}_ori/{fname}.png"
        fname_ds_3 = f"feature/{self.set_idx}_p3_ds/{fname}.png"
        fname_ds = f"feature/{self.set_idx}_ds/{fname}.png"
        fname_resid = f"feature/{self.set_idx}_resid/{fname}.png"
        fname_finenet = f"feature/{self.set_idx}_finenet/{fname}.png"

        with open(f"info/{self.set_idx}/{fname}_inputs.bin", "wb") as inputs_f:
            torch.save(inputs, inputs_f)

        if self.DSSLIC_mode == 'split':
            if self.P3_net == True:
                utils.save_feature_map_onlyp2_8channels(fname_ds, ds_feat)
                utils.save_feature_map_onlyp3_8channels(fname_ds_3, ds_feat_3)
                utils.save_feature_map_p45(fname_feat, image_feat)
            else:
                if self.channel8 == True:
                    utils.save_feature_map_onlyp2_8channels(fname_ds, ds_feat)
                    utils.save_feature_map_p345(fname_feat, image_feat)
                elif self.channel16 == True:
                    utils.save_feature_map_onlyp2_16channels(fname_ds, ds_feat)
                    utils.save_feature_map_p345(fname_feat, image_feat)
                else:
                    utils.save_feature_map_onlyp2(fname_ds, ds_feat)
                    utils.save_feature_map_p345(fname_feat, image_feat)
        return fname_feat

    def feature_coding_resid(self):
        print("Saving features maps...")
        filenames = []
        start_time = time.clock()
        with tqdm(total=len(self.data_loader.dataset)) as pbar:
            for inputs in iter(self.data_loader):
                #自己加入的5行，断了之后重新跑，提过feature的不用再提
                fname_temp = utils.simple_filename(inputs[0]["file_name"])
                fea_resid_path = f"feature/{self.set_idx}_resid/"
                if os.path.isfile(f"{fea_resid_path}{fname_temp}.png"):
                    print(f"feature extraction: {fname_temp} skip (exist)")
                    continue
                filenames.append(self._feature_coding_resid(inputs)) #inputs是filename 15d64d, height 680, width 1024, image_id 19877和image这个tensor [3, 800, 1205] uint8 大于1
                pbar.update()

        end_time = time.clock()
        run_time = end_time - start_time
        print('Run time for each', run_time/10)
        print("runvtm---------------------runvtmrunvtmrunvtmrunvtmrunvtmrunvtmrunvtm")
        run_vtm_resid(f"feature/{self.set_idx}_resid", self.VTM_param["QP"], self.VTM_param["threads"],resid = False)
        if self.P3_net == True:
            run_vtm_resid(f"feature/{self.set_idx}_p3_resid", self.VTM_param["QP"], self.VTM_param["threads"], resid=False)
        return filenames

    def _feature_coding_resid(self, inputs):
        print('feature_coding_resid')
        images = self.model.preprocess_image(inputs)  # images: device cpu, image_sizes [800, 1205] tensor [1, 3, 800, 1216] torch.float32 cpu
        features = self.model.backbone(images.tensor)

        # features_resid = features.copy()
        # features_resid["p2"] = resid_pic
        # resid_feat = quant_fix(features_resid.copy())
        #################################ccr added
        fname = utils.simple_filename(inputs[0]["file_name"])
        fname_feat = f"feature/{self.set_idx}_ori/{fname}.png"
        fname_feat_rec = f"feature/{self.set_idx}_rec/{fname}.png"
        fname_ds_rec = f"feature/{self.set_idx}_ds_rec/{fname}.png"
        fname_resid = f"feature/{self.set_idx}_resid/{fname}.png"
        fname_p3_rec = f"feature/{self.set_idx}_p3_ds_rec/{fname}.png"
        fname_resid_3 = f"feature/{self.set_idx}_p3_resid/{fname}.png"
        if self.DSSLIC_mode == 'split':
            if self.channel8 == True:
                features_rec = self.feat2feat_onlyp2_8channels(fname_ds_rec)
            elif self.channel16 == True:
                features_rec = self.feat2feat_onlyp2_16channels(fname_ds_rec)
            else:
                features_rec = self.feat2feat_onlyp2(fname_ds_rec)
            if self.ds == 2:
                upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear')
            if self.ds == 4:
                upsample = torch.nn.Upsample(scale_factor=4, mode='bilinear')
            if self.P3_net == True:
                features_rec_3 = self.feat2feat_onlyp3_8channels(fname_p3_rec)

        up_image = self.model.netG_8to256.forward(features_rec['p2'])
        up_image = upsample(up_image)
        res = self.model.netG_256.forward(up_image)
        fake_image_f = res + up_image
        features_rec['p2'] = fake_image_f

        if self.P3_net == True:
            up_image_3 = self.model.netG_8to256.forward(features_rec_3['p3'])
            up_image_3 = upsample(up_image_3)
            res_3 = self.model.netG_256.forward(up_image_3)
            fake_image_f_3 = res_3 + up_image_3
            features_rec_3['p3'] = fake_image_f_3

        features_afterVVC = features.copy()
        resid_pic = features['p2'] - features_rec['p2']
        features_afterVVC["p2"] = resid_pic
        resid_feat = quant_fix_resid(features_afterVVC.copy())
        utils.save_feature_map_onlyp2(fname_resid, resid_feat)

        if self.P3_net == True:
            resid_pic_3 = features['p3'] - features_rec_3['p3']
            features_afterVVC["p3"] = resid_pic_3
            resid_feat = quant_fix_resid(features_afterVVC.copy())
            utils.save_feature_map_onlyp3(fname_resid_3, resid_feat)
        return fname_resid

    def evaluation(self, inputs):

        with open(f"./output/{self.set_idx}_coco.txt", 'w') as of:
            if self.seg == True:
                of.write('ImageID,LabelName,Score,XMin,XMax,YMin,YMax,ImageWidth,ImageHeight,Mask\n')
            else:
                of.write('ImageID,LabelName,Score,XMin,XMax,YMin,YMax\n')

            coco_classes_fname = 'oi_eval/coco_classes.txt'

            with open(coco_classes_fname, 'r') as f:
                coco_classes = f.read().splitlines()
            # for fname in tqdm(inputs[:2]):
            for fname in tqdm(inputs):
                print(fname)
                if self.ori_eval == True:
                    outputs = self._evaluation_ori(fname)
                else:
                    outputs = self._evaluation(fname)
                outputs = outputs[0]

                imageId = os.path.basename(fname)
                classes = outputs['instances'].pred_classes.to('cpu').numpy()
                scores = outputs['instances'].scores.to('cpu').numpy()
                bboxes = outputs['instances'].pred_boxes.tensor.to('cpu').numpy()
                H, W = outputs['instances'].image_size

                bboxes = bboxes / [W, H, W, H]
                bboxes = bboxes[:, [0, 2, 1, 3]]

                if self.seg == True:
                    masks = outputs['instances'].pred_masks.to('cpu').numpy()

                for ii in range(len(classes)):
                    coco_cnt_id = classes[ii]
                    class_name = coco_classes[coco_cnt_id]

                    rslt = [imageId[:-4], class_name, scores[ii]] + \
                        bboxes[ii].tolist()

                    if self.seg == True:
                        assert (masks[ii].shape[1] == W) and (masks[ii].shape[0] == H)
                        rslt += [masks[ii].shape[1], masks[ii].shape[0], oid_mask_encoding.encode_binary_mask(masks[ii]).decode('ascii')]

                    o_line = ','.join(map(str,rslt))
                    print('1111')
                    of.write(o_line + '\n')

        conversion(self.set_idx)
        print('before cmd')
        cmd = f"python oid_challenge_evaluation.py \
        --input_annotations_boxes   oi_eval/detection_validation_5k_bbox.csv \
        --input_annotations_labels  oi_eval/detection_validation_labels_5k.csv \
        --input_class_labelmap      oi_eval/coco_label_map.pbtxt \
        --input_predictions         output/{self.set_idx}_oi.txt \
        --output_metrics            output/{self.set_idx}_AP.txt"

        if self.COCO == True:
            cmd = f"python oid_challenge_evaluation.py \
                    --input_annotations_boxes   COCO_eval/test.csv \
                    --input_annotations_labels  COCO_eval/useless.csv \
                    --input_class_labelmap      oi_eval/coco_label_map.pbtxt \
                    --input_predictions         output/{self.set_idx}_oi.txt \
                    --output_metrics            output/{self.set_idx}_AP.txt"

        if self.seg == True:
            cmd = f"python oid_challenge_evaluation.py \
                --input_annotations_boxes   dataset/annotations_5k/segmentation_validation_bbox_5k.csv \
                --input_annotations_labels  dataset/annotations_5k/segmentation_validation_labels_5k.csv \
                --input_class_labelmap      dataset/annotations_5k/coco_label_map.pbtxt \
                --input_annotations_segm    dataset/annotations_5k/segmentation_validation_masks_5k.csv \
                --input_predictions         output/{self.set_idx}_oi.txt \
                --output_metrics            output/{self.set_idx}_AP_SEG.txt"
        print(">>>> cmd: ", cmd)
        subprocess.call([cmd], shell=True)

        self.summary()
        
        return



    def evaluation_offline(self):
        cmd = f"python oid_challenge_evaluation.py \
        --input_annotations_boxes   oi_eval/detection_validation_5k_bbox.csv \
        --input_annotations_labels  oi_eval/detection_validation_labels_5k.csv \
        --input_class_labelmap      oi_eval/coco_label_map.pbtxt \
        --input_predictions         /media/data/minglang/data/detection_result/q7_result.oid.txt \
        --output_metrics            inference_ml/origin_AP.txt"
        print(">>>> cmd: ", cmd)
        subprocess.call([cmd], shell=True)

    def _evaluation_ori(self, fname):
        fname_simple = utils.simple_filename(fname)

        with open(f"info/{self.set_idx}/{fname_simple}_inputs.bin", "rb") as inputs_f:
            inputs = torch.load(inputs_f)

        images = self.model.preprocess_image(inputs)
        features = self.model.backbone(images.tensor)

        outputs = self.forward_front(inputs, images, features)
        self.evaluator.process(inputs, outputs)
        return outputs


    def _evaluation(self, fname):
        fname_simple = utils.simple_filename(fname)
        #fname_simple = fname.replace('rec', 'ori')#################test0810
        fname_ds_ori = fname.replace('rec', 'ds')
        fname_ds_rec = fname.replace('rec', 'ds_rec')
        fname_resid_rec = fname.replace('rec', 'resid_rec')
        fname_ori = fname.replace('rec', 'ori')
        fname_resid_ori = fname.replace('rec', 'resid')
        fname_p3_ds_rec = fname.replace('rec', 'p3_ds_rec')
        fname_p3_resid_rec = fname.replace('rec', 'p3_resid_rec')

        with open(f"info/{self.set_idx}/{fname_simple}_inputs.bin", "rb") as inputs_f:
            inputs = torch.load(inputs_f)

        images = self.model.preprocess_image(inputs)
        features_ori = self.model.backbone(images.tensor)
        # features = self.model.backbone(images.tensor)
        # if self.DSSLIC_mode == 'together':
        #     if self.ds == 4:
        #         features = self.feat2feat_4ds(fname)
        #     if self.ds == 2:
        #         features = self.feat2feat_2ds(fname)
        #     if self.P2_ori == True:
        #         features_ori = self.feat2feat_4ds(fname_ori)
        #         features['p2'] = features_ori['p2']
        if self.DSSLIC_mode == 'split':
            if self.P3_net == True:
                features = self.feat2feat_p45(fname)
                features_ds = self.feat2feat_onlyp2_8channels(fname_ds_rec)
                features['p2'] = features_ds['p2']
                features_p3_ds = self.feat2feat_onlyp3_8channels(fname_p3_ds_rec)
                features['p3'] = features_p3_ds['p3']
            else:
                features = self.feat2feat_p345(fname)
                if self.P345_ori == True:
                    for p in features:
                        if p == 'p3' or p == 'p4' or p == 'p5':
                            features[p] = features_ori[p]

                if self.P2_ori == False:
                    if self.channel8 == True:
                        features_ds = self.feat2feat_onlyp2_8channels(fname_ds_rec)
                        features['p2'] = features_ds['p2']
                    elif self.channel16 == True:
                        features_ds = self.feat2feat_onlyp2_16channels(fname_ds_rec)
                        features['p2'] = features_ds['p2']
                    else:
                        features_ds = self.feat2feat_onlyp2(fname_ds_rec)
                        features['p2'] = features_ds['p2']
                elif self.P2_ori == 'vvc':
                    if self.channel16 == True:
                        features_P2ori = self.feat2feat_onlyp2_16channels(fname_ds_ori)
                        features['p2'] = features_P2ori['p2']
                    elif self.channel8 == True:
                        features_P2ori = self.feat2feat_onlyp2_8channels(fname_ds_ori)
                        features['p2'] = features_P2ori['p2']
                    else:
                        features_ds = self.feat2feat_onlyp2(fname_ds_ori)
                        features['p2'] = features_ds['p2']
                elif self.P2_ori == 'bb':
                    features['p2'] = features_ori['p2']
                    features['p2'] = self.model.compG.forward(features['p2'])


        if self.ds == 8:
            upsample = torch.nn.Upsample(scale_factor=8, mode='bilinear')
        if self.ds == 4:
            upsample = torch.nn.Upsample(scale_factor=4, mode='bilinear')
        if self.ds == 2:
            upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear')

        for p in features:
            if p=='p2':
                comp_image = self.model.netG_8to256.forward(features[p])
                if self.DSSLIC_mode == 'split':
                    up_image = upsample(comp_image)

                if self.MIF == True:
                    upsample_ref = torch.nn.Upsample(scale_factor=8, mode='bilinear')
                    up_p5 = upsample_ref(features['p5'])
                    input_fconcat = torch.cat((up_p5, up_image), 1)
                    res = self.model.netG_512.forward(input_fconcat)
                else:
                    input_fconcat = up_image
                    res = self.model.netG_256.forward(input_fconcat)
                fake_image_f = res + up_image
                if self.nofinenet == True:
                    fake_image_f = up_image
                features[p] = fake_image_f
            if self.P3_net == True:
                if p == 'p3':
                    comp_image_p3 = self.model.netG_8to256.forward(features[p])
                    up_image_p3 = upsample(comp_image_p3)
                    if self.MIF == True:
                        upsample_ref = torch.nn.Upsample(scale_factor=4, mode='bilinear')
                        up_p5 = upsample_ref(features['p5'])
                        input_fconcat = torch.cat((up_p5, up_image_p3), 1)
                        res_p3 = self.model.netG_512.forward(input_fconcat)
                    else:
                        input_fconcat_p3 = up_image_p3
                        res_p3 = self.model.netG_256.forward(input_fconcat_p3)
                    fake_image_f_p3 = res_p3 + up_image_p3
                    if self.nofinenet == True:
                        fake_image_f_p3 = up_image_p3
                    features[p] = fake_image_f_p3
        ################################################################
        # maxP3 = features['p3'].max()
        # minP3 = features['p3'].min()
        # maxP2 = features['p2'].max()
        # minP2 = features['p2'].min()
        # scale_dsslic = (maxP3 - minP3) / (maxP2 - minP2)
        # features['p2'] -= minP2
        # features['p2'] = features['p2'] * scale_dsslic
        # features['p2'] += minP3
        ################################################################

                        # #####################################################################################################################
        if self.resid_eval == True:
            if self.resid_8c == True:
                features_resid_rec_8c = self.feat2feat_onlyp2_8channels(fname_resid_rec)
                resid_rec_256c = self.model.netG_8to256.forward(features_resid_rec_8c['p2'])
                up_resid = upsample(resid_rec_256c)
                res_resid = self.model.netG_256.forward(up_resid)
                final_resid = res_resid + up_resid
                features['p2'] = features['p2'] + final_resid
            else:
                features_resid_rec = self.feat2feat_onlyp2_resid(fname_resid_rec)
                features['p2'] = features['p2'] + features_resid_rec['p2']
                if self.P3_net == True:
                    features_resid_rec_p3 = self.feat2feat_onlyp2_resid(fname_p3_resid_rec)
                    features['p3'] = features['p3'] + features_resid_rec_p3['p2']

        if self.P2_all0 == True:
            features['p2'][:, :, :, :] = 0
        ###################################################ccr added
        # features_345 = self.feat2feat(fname_345)
        # compG_input = features['p2']
        # comp_image = self.model.compG.forward(compG_input)
        # upsample = torch.nn.Upsample(scale_factor=8, mode='bilinear')
        # up_image = upsample(comp_image)
        # input_fconcat = up_image
        # res = self.model.netG.forward(input_fconcat)
        # fake_image_f = res + up_image
        # features['p2'] = fake_image_f
        # features['p3'] = features_345['p3']
        # features['p4'] = features_345['p4']
        # features['p5'] = features_345['p5']
        ###################################################ccr added
        outputs = self.forward_front(inputs, images, features)
        self.evaluator.process(inputs, outputs)
        return outputs
    
    def summary(self):
        with open("results.csv", "a") as result_f:
            # with open(f"inference/{self.set_idx}_AP.txt", "rt") as ap_f:
            with open(f"output/{self.set_idx}_AP.txt", "rt") as ap_f:
                ap = ap_f.readline()
                ap = ap.split(",")[1][:-1]

            size_basis = utils.get_size(f'feature/{self.set_idx}_bit/')

            # ml add
            size_coeffs, size_mean, self.qp, self.DeepCABAC_qstep = 0, 0, 0, 0
            # bpp = (size_basis + size_coeffs, + size_mean)/self.pixel_num
            # ml
            bpp = 0

            print(">>>> result: ", f"{self.set_idx},{self.qp},{self.DeepCABAC_qstep},{bpp},{ap}\n")
            if self.P2_all0 == True:
                P2_value = 'All 0'
            else:
                P2_value = self.P2_ori
            print('>>>> config: P2:', f"{P2_value}\n", 'P345:', f"{self.P345_ori}\n")

            result_f.write(f"{self.set_idx},{self.qp},{self.DeepCABAC_qstep},{bpp},{ap}\n")
            
    def feat2feat(self, fname):
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

        pyramid["p2"] = self.feature_slice(v2_blk, [v2_blk.shape[0] // 16, v2_blk.shape[1] // 16 ])
        pyramid["p3"] = self.feature_slice(v3_blk, [v3_blk.shape[0] // 8 , v3_blk.shape[1] // 32 ])
        pyramid["p4"] = self.feature_slice(v4_blk, [v4_blk.shape[0] // 4 , v4_blk.shape[1] // 64 ])
        pyramid["p5"] = self.feature_slice(v5_blk, [v5_blk.shape[0] // 2 , v5_blk.shape[1] // 128])

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

    def feat2feat_2ds(self, fname):
        pyramid = {}

        png = cv2.imread(fname, -1).astype(np.float32)
        vectors_height = png.shape[0]
        v2_h = int(vectors_height / 37 * 16)
        v3_h = int(vectors_height / 37 * 32)
        v4_h = int(vectors_height / 37 * 36)

        v2_blk = png[:v2_h, :]
        v3_blk = png[v2_h:v3_h, :]
        v4_blk = png[v3_h:v4_h, :]
        v5_blk = png[v4_h:vectors_height, :]

        pyramid["p2"] = self.feature_slice(v2_blk, [v2_blk.shape[0] // 8 , v2_blk.shape[1] // 32 ])
        pyramid["p3"] = self.feature_slice(v3_blk, [v3_blk.shape[0] // 8 , v3_blk.shape[1] // 32 ])
        pyramid["p4"] = self.feature_slice(v4_blk, [v4_blk.shape[0] // 4 , v4_blk.shape[1] // 64 ])
        pyramid["p5"] = self.feature_slice(v5_blk, [v5_blk.shape[0] // 2 , v5_blk.shape[1] // 128])

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

    def feat2feat_4ds(self, fname):
        pyramid = {}

        png = cv2.imread(fname, -1).astype(np.float32)
        vectors_height = png.shape[0]
        v2_h = int(vectors_height / 25 * 4)
        v3_h = int(vectors_height / 25 * 20)
        v4_h = int(vectors_height / 25 * 24)

        v2_blk = png[:v2_h, :]
        v3_blk = png[v2_h:v3_h, :]
        v4_blk = png[v3_h:v4_h, :]
        v5_blk = png[v4_h:vectors_height, :]

        pyramid["p2"] = self.feature_slice(v2_blk, [v2_blk.shape[0] // 4 , v2_blk.shape[1] // 64 ])
        pyramid["p3"] = self.feature_slice(v3_blk, [v3_blk.shape[0] // 8 , v3_blk.shape[1] // 32 ])
        pyramid["p4"] = self.feature_slice(v4_blk, [v4_blk.shape[0] // 4 , v4_blk.shape[1] // 64 ])
        pyramid["p5"] = self.feature_slice(v5_blk, [v5_blk.shape[0] // 2 , v5_blk.shape[1] // 128])

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

    def feat2feat_P23_2ds(self, fname):
        pyramid = {}

        png = cv2.imread(fname, -1).astype(np.float32)
        vectors_height = png.shape[0]
        v2_h = int(vectors_height / 25 * 16)
        v3_h = int(vectors_height / 25 * 20)
        v4_h = int(vectors_height / 25 * 24)

        v2_blk = png[:v2_h, :]
        v3_blk = png[v2_h:v3_h, :]
        v4_blk = png[v3_h:v4_h, :]
        v5_blk = png[v4_h:vectors_height, :]

        pyramid["p2"] = self.feature_slice(v2_blk, [v2_blk.shape[0] // 8 , v2_blk.shape[1] // 32 ])
        pyramid["p3"] = self.feature_slice(v3_blk, [v3_blk.shape[0] // 4 , v3_blk.shape[1] // 64 ])
        pyramid["p4"] = self.feature_slice(v4_blk, [v4_blk.shape[0] // 4 , v4_blk.shape[1] // 64 ])
        pyramid["p5"] = self.feature_slice(v5_blk, [v5_blk.shape[0] // 2 , v5_blk.shape[1] // 128])

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

    def feat2feat_onlyp2(self, fname):
        pyramid = {}
        png = cv2.imread(fname, -1).astype(np.float32)
        pyramid["p2"] = self.feature_slice(png, [png.shape[0] // 16, png.shape[1] // 16])
        pyramid["p2"] = dequant_fix(pyramid["p2"])
        pyramid["p2"] = torch.unsqueeze(pyramid["p2"], 0)
        #加了下面这几句弄到cuda
        pyramid["p2"] = pyramid["p2"].cuda()
        return pyramid

    def feat2feat_onlyp2_dsslicquant(self, fname):
        pyramid = {}
        png = cv2.imread(fname, -1).astype(np.float32)
        pyramid["p2"] = self.feature_slice(png, [png.shape[0] // 16, png.shape[1] // 16])
        pyramid["p2"] = dequant_fix_dsslic(pyramid["p2"])
        pyramid["p2"] = torch.unsqueeze(pyramid["p2"], 0)
        #加了下面这几句弄到cuda
        pyramid["p2"] = pyramid["p2"].cuda()
        return pyramid

    def feat2feat_onlyp2_8channels(self, fname):
        pyramid = {}
        png = cv2.imread(fname, -1).astype(np.float32)
        pyramid["p2"] = self.feature_slice(png, [png.shape[0] // 4, png.shape[1] // 2])
        pyramid["p2"] = dequant_fix_dsslic(pyramid["p2"])
        pyramid["p2"] = torch.unsqueeze(pyramid["p2"], 0)
        # 加了下面这几句弄到cuda
        pyramid["p2"] = pyramid["p2"].cuda()
        return pyramid

    def feat2feat_onlyp3_8channels(self, fname):
        pyramid = {}
        png = cv2.imread(fname, -1).astype(np.float32)
        pyramid["p3"] = self.feature_slice(png, [png.shape[0] // 4, png.shape[1] // 2])
        pyramid["p3"] = dequant_fix_dsslic(pyramid["p3"])
        pyramid["p3"] = torch.unsqueeze(pyramid["p3"], 0)
        # 加了下面这几句弄到cuda
        pyramid["p3"] = pyramid["p3"].cuda()
        return pyramid

    def feat2feat_onlyp2_16channels(self, fname):
        pyramid = {}
        png = cv2.imread(fname, -1).astype(np.float32)
        pyramid["p2"] = self.feature_slice(png, [png.shape[0] // 4, png.shape[1] // 4])
        pyramid["p2"] = dequant_fix_dsslic(pyramid["p2"])
        pyramid["p2"] = torch.unsqueeze(pyramid["p2"], 0)
        # 加了下面这几句弄到cuda
        pyramid["p2"] = pyramid["p2"].cuda()
        return pyramid

    def feat2feat_onlyp2_resid(self, fname):
        pyramid = {}
        png = cv2.imread(fname, -1).astype(np.float32)
        pyramid["p2"] = self.feature_slice(png, [png.shape[0] // 16, png.shape[1] // 16])
        pyramid["p2"] = dequant_fix_resid(pyramid["p2"])
        pyramid["p2"] = torch.unsqueeze(pyramid["p2"], 0)
        #加了下面这几句弄到cuda
        pyramid["p2"] = pyramid["p2"].cuda()
        return pyramid

    def feat2feat_p345(self, fname):
        pyramid = {}
        png = cv2.imread(fname, -1).astype(np.float32)
        vectors_height = png.shape[0]
        v3_h = int(vectors_height / 21 * 16)
        v4_h = int(vectors_height / 21 * 20)

        v3_blk = png[:v3_h, :]
        v4_blk = png[v3_h:v4_h, :]
        v5_blk = png[v4_h:vectors_height, :]

        pyramid["p3"] = self.feature_slice(v3_blk, [v3_blk.shape[0] // 8 , v3_blk.shape[1] // 32 ])
        pyramid["p4"] = self.feature_slice(v4_blk, [v4_blk.shape[0] // 4 , v4_blk.shape[1] // 64 ])
        pyramid["p5"] = self.feature_slice(v5_blk, [v5_blk.shape[0] // 2 , v5_blk.shape[1] // 128])

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

    def feat2feat_p45(self, fname):
        pyramid = {}
        png = cv2.imread(fname, -1).astype(np.float32)
        vectors_height = png.shape[0]
        v4_h = int(vectors_height / 5 * 4)

        v4_blk = png[:v4_h, :]
        v5_blk = png[v4_h:vectors_height, :]

        pyramid["p4"] = self.feature_slice(v4_blk, [v4_blk.shape[0] // 16 , v4_blk.shape[1] // 16 ])
        pyramid["p5"] = self.feature_slice(v5_blk, [v5_blk.shape[0] // 8 , v5_blk.shape[1] // 32])

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

    def feature_slice(self, image, shape):
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
        feature = torch.from_numpy(np.array(blk))
        return feature

    def clear(self):
        DatasetCatalog._REGISTERED.clear()


class DetectEval(Eval):
    def prepare_part(self, myarg, data_name="pick"):
        print("Loading", data_name, "...")
        utils.pick_coco_exp(data_name, myarg)
        self.data_loader = build_detection_test_loader(self.cfg, data_name)
        self.evaluator = COCOEvaluator(data_name, self.cfg, False)
        self.evaluator.reset()
        print(data_name, "Loaded")
