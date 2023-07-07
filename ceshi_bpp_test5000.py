<<<<<<< HEAD
import os
import glob
import utils
from PIL import Image
import numpy as np

def filesize(filepath: str) -> int:
    """Return file size in bits of `filepath`."""
    if not os.path.isfile(filepath):
        raise ValueError(f'Invalid file "{filepath}".')
    return os.stat(filepath).st_size

def simple_filename(filename_ext):
    filename_base = os.path.basename(filename_ext)
    filename_noext = os.path.splitext(filename_base)[0]
    return filename_noext

set_idx = 45
DSSLIC_mode = 'split'
resid = True
p3_net = False

filenames = glob.glob(f"/media/data/liutie/VCM/OpenImageV6-5K/*.jpg")
path_img_qianzhui = '/media/data/liutie/VCM/OpenImageV6-5K/'

# filenames = glob.glob(f"/media/data/ccr/OI5000_seg/*.jpg")
# path_img_qianzhui = '/media/data/ccr/OI5000_seg/'

# filenames = glob.glob(f"/media/data/ccr/VCM/datasets/coco/val2017/*.jpg")
# path_img_qianzhui = '/media/data/ccr/VCM/datasets/coco/val2017/'

# filenames = glob.glob(f"/media/data/liutie/VCM/rcnn/VCM_EE1.2_P-layer_feature_map_anchor_generation_137th_MPEG-VCM-main/feature/{set_idx}_rec/*.png")
num_img = len(filenames)
path_vvc_qianzhui = f'./feature/{set_idx}_bit/' #QP35
path_ds_qianzhui = f'./feature/{set_idx}_ds_bit/' #QP35
path_resid_qianzhui = f'./feature/{set_idx}_resid_bit/' #QP35
path_p3_ds_qianzhui = f'./feature/{set_idx}_p3_ds_bit/' #QP35
path_p3_resid_qianzhui = f'./feature/{set_idx}_p3_resid_bit/' #QP35

i_count = 0
bit_sum = 0
num_pixel_sum = 0
bpp_sum = 0
bpp_ds_sum = 0
ds_rate_sum = 0
resid_rate_sum = 0

bit_all = np.zeros((num_img))
num_pixel_all = np.zeros((num_img))
bpp_all = np.zeros((num_img))
for fname in filenames:
    fname_simple = utils.simple_filename(fname) #000a1249af2bc5f0
    #debug确认用
    # if fname_simple != '000a1249af2bc5f0':
    #     continue
    path_img = path_img_qianzhui + fname_simple + '.jpg'
    path_vvc = path_vvc_qianzhui + fname_simple + '.vvc'
    path_ds = path_ds_qianzhui + fname_simple + '.vvc'
    path_resid = path_resid_qianzhui + fname_simple + '.vvc'
    path_ds_p3 = path_p3_ds_qianzhui + fname_simple + '.vvc'
    path_resid_p3 = path_p3_resid_qianzhui + fname_simple + '.vvc'

    im = Image.open(path_img).convert('RGB')
    height = im.size[1] #678
    width = im.size[0] #1024
    bit1 = filesize(path_vvc) * 8.0 #1070872
    # bit = bit1
    if DSSLIC_mode == 'split':
        bit_ds = filesize(path_ds) * 8.0
        if p3_net == True:
            bit_ds_p3 = filesize(path_ds_p3) * 8.0
            bit_ds += bit_ds_p3
        bit_noresid = bit1+bit_ds
        bit = bit_noresid
    if resid == True:
        bit_resid = filesize(path_resid) * 8.0
        if p3_net == True:
            bit_resid_p3 = filesize(path_resid_p3) * 8.0
            bit_resid += bit_resid_p3
        bit = bit_noresid + bit_resid
        resid_rate = bit_resid / (bit_noresid + bit_resid)

    # bit = bit1  ##############only bit
    print(bit1,'---------bit1',path_vvc)
    # print(bit_ds_p3, '---------bitds_p3', path_ds_p3)
    print(bit_ds,'---------bitds',path_ds)
    # print(bit_resid_p3, '---------bitresid', path_resid_p3)
    print(bit_resid,'---------bitresid',path_resid)
    num_pixel = height * width #694272
    bpp = bit / num_pixel #1.542
    bit_sum = bit_sum + bit
    num_pixel_sum = num_pixel_sum + num_pixel
    bpp_sum = bpp_sum + bpp
    bit_all[i_count] = bit
    num_pixel_all[i_count] = num_pixel
    bpp_all[i_count] = bpp
    if DSSLIC_mode == 'split':
        ds_rate = bit_ds/bit
        ds_rate_sum = ds_rate_sum +ds_rate
    if resid == True:
        resid_rate = bit_resid / bit
        resid_rate_sum = resid_rate_sum + resid_rate

    print('%d/%d, hw[%dx%d], bit/pixel/bpp: %d/%d/%6.3f, imgname: %s' %((i_count+1), num_img, height, width, bit, num_pixel, bpp, path_vvc))
    i_count = i_count + 1
bit_avg = bit_sum / num_img
num_pixel_avg = num_pixel_sum / num_img
bpp_avg = bpp_sum / num_img
print('avg_bit: {:.4f}, sum_bit: {:.0f}, num_image: {:.0f}'.format(bit_avg, bit_sum, num_img))
print('avg_numpixel: {:.0f}, sum_numpixel: {:.0f}, num_image: {:.0f}'.format(num_pixel_avg, num_pixel_sum, num_img))
print('avg_bpp: {:.4f}, sum_bpp: {:.4f}, num_image: {:.0f}'.format(bpp_avg, bpp_sum, num_img))
if DSSLIC_mode == 'split':
    print('avg_ds_rate:',ds_rate_sum/5000)
if resid == True:
    print('avg_resid_rate:',resid_rate_sum/5000)



=======
import os
import glob
import utils
from PIL import Image
import numpy as np

def filesize(filepath: str) -> int:
    """Return file size in bits of `filepath`."""
    if not os.path.isfile(filepath):
        raise ValueError(f'Invalid file "{filepath}".')
    return os.stat(filepath).st_size

def simple_filename(filename_ext):
    filename_base = os.path.basename(filename_ext)
    filename_noext = os.path.splitext(filename_base)[0]
    return filename_noext

set_idx = 45
DSSLIC_mode = 'split'
resid = True
p3_net = False

filenames = glob.glob(f"/media/data/liutie/VCM/OpenImageV6-5K/*.jpg")
path_img_qianzhui = '/media/data/liutie/VCM/OpenImageV6-5K/'

# filenames = glob.glob(f"/media/data/ccr/OI5000_seg/*.jpg")
# path_img_qianzhui = '/media/data/ccr/OI5000_seg/'

# filenames = glob.glob(f"/media/data/ccr/VCM/datasets/coco/val2017/*.jpg")
# path_img_qianzhui = '/media/data/ccr/VCM/datasets/coco/val2017/'

# filenames = glob.glob(f"/media/data/liutie/VCM/rcnn/VCM_EE1.2_P-layer_feature_map_anchor_generation_137th_MPEG-VCM-main/feature/{set_idx}_rec/*.png")
num_img = len(filenames)
path_vvc_qianzhui = f'./feature/{set_idx}_bit/' #QP35
path_ds_qianzhui = f'./feature/{set_idx}_ds_bit/' #QP35
path_resid_qianzhui = f'./feature/{set_idx}_resid_bit/' #QP35
path_p3_ds_qianzhui = f'./feature/{set_idx}_p3_ds_bit/' #QP35
path_p3_resid_qianzhui = f'./feature/{set_idx}_p3_resid_bit/' #QP35

i_count = 0
bit_sum = 0
num_pixel_sum = 0
bpp_sum = 0
bpp_ds_sum = 0
ds_rate_sum = 0
resid_rate_sum = 0

bit_all = np.zeros((num_img))
num_pixel_all = np.zeros((num_img))
bpp_all = np.zeros((num_img))
for fname in filenames:
    fname_simple = utils.simple_filename(fname) #000a1249af2bc5f0
    #debug确认用
    # if fname_simple != '000a1249af2bc5f0':
    #     continue
    path_img = path_img_qianzhui + fname_simple + '.jpg'
    path_vvc = path_vvc_qianzhui + fname_simple + '.vvc'
    path_ds = path_ds_qianzhui + fname_simple + '.vvc'
    path_resid = path_resid_qianzhui + fname_simple + '.vvc'
    path_ds_p3 = path_p3_ds_qianzhui + fname_simple + '.vvc'
    path_resid_p3 = path_p3_resid_qianzhui + fname_simple + '.vvc'

    im = Image.open(path_img).convert('RGB')
    height = im.size[1] #678
    width = im.size[0] #1024
    bit1 = filesize(path_vvc) * 8.0 #1070872
    # bit = bit1
    if DSSLIC_mode == 'split':
        bit_ds = filesize(path_ds) * 8.0
        if p3_net == True:
            bit_ds_p3 = filesize(path_ds_p3) * 8.0
            bit_ds += bit_ds_p3
        bit_noresid = bit1+bit_ds
        bit = bit_noresid
    if resid == True:
        bit_resid = filesize(path_resid) * 8.0
        if p3_net == True:
            bit_resid_p3 = filesize(path_resid_p3) * 8.0
            bit_resid += bit_resid_p3
        bit = bit_noresid + bit_resid
        resid_rate = bit_resid / (bit_noresid + bit_resid)

    # bit = bit1  ##############only bit
    print(bit1,'---------bit1',path_vvc)
    # print(bit_ds_p3, '---------bitds_p3', path_ds_p3)
    print(bit_ds,'---------bitds',path_ds)
    # print(bit_resid_p3, '---------bitresid', path_resid_p3)
    print(bit_resid,'---------bitresid',path_resid)
    num_pixel = height * width #694272
    bpp = bit / num_pixel #1.542
    bit_sum = bit_sum + bit
    num_pixel_sum = num_pixel_sum + num_pixel
    bpp_sum = bpp_sum + bpp
    bit_all[i_count] = bit
    num_pixel_all[i_count] = num_pixel
    bpp_all[i_count] = bpp
    if DSSLIC_mode == 'split':
        ds_rate = bit_ds/bit
        ds_rate_sum = ds_rate_sum +ds_rate
    if resid == True:
        resid_rate = bit_resid / bit
        resid_rate_sum = resid_rate_sum + resid_rate

    print('%d/%d, hw[%dx%d], bit/pixel/bpp: %d/%d/%6.3f, imgname: %s' %((i_count+1), num_img, height, width, bit, num_pixel, bpp, path_vvc))
    i_count = i_count + 1
bit_avg = bit_sum / num_img
num_pixel_avg = num_pixel_sum / num_img
bpp_avg = bpp_sum / num_img
print('avg_bit: {:.4f}, sum_bit: {:.0f}, num_image: {:.0f}'.format(bit_avg, bit_sum, num_img))
print('avg_numpixel: {:.0f}, sum_numpixel: {:.0f}, num_image: {:.0f}'.format(num_pixel_avg, num_pixel_sum, num_img))
print('avg_bpp: {:.4f}, sum_bpp: {:.4f}, num_image: {:.0f}'.format(bpp_avg, bpp_sum, num_img))
if DSSLIC_mode == 'split':
    print('avg_ds_rate:',ds_rate_sum/5000)
if resid == True:
    print('avg_resid_rate:',resid_rate_sum/5000)



>>>>>>> be0df11a27051c4085a72cd9a58dcd1fea1d076a
