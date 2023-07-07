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

set_idx = 35

# filenames = glob.glob(f"/media/data/liutie/VCM/OpenImageV6-5K/*.jpg")
filenames = [f"/media/data/liutie/VCM/OpenImageV6-5K/86463a5a7dcb1a69.jpg"]
path_img_qianzhui = '/media/data/liutie/VCM/OpenImageV6-5K/'

# filenames = glob.glob(f"/media/data/ccr/COCO-val2017/*.jpg")
# filenames = [f"/media/data/ccr/COCO-val2017/000000027620.jpg"]
# path_img_qianzhui = '/media/data/ccr/COCO-val2017/'

# filenames = glob.glob(f"/media/data/liutie/VCM/rcnn/VCM_EE1.2_P-layer_feature_map_anchor_generation_137th_MPEG-VCM-main/feature/{set_idx}_rec/*.png")
num_img = len(filenames)

# path_vvc_qianzhui = f'/media/data/ccr/COCO_vtm/{set_idx}_bit/' #QP35
path_vvc_qianzhui = f'/media/data/ccr/OI5000/{set_idx}_bit/' #QP35

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

    im = Image.open(path_img).convert('RGB')
    height = im.size[1] #678
    width = im.size[0] #1024
    bit1 = filesize(path_vvc) * 8.0 #1070872
    bit = bit1

    num_pixel = height * width #694272
    bpp = bit / num_pixel #1.542
    bit_sum = bit_sum + bit
    num_pixel_sum = num_pixel_sum + num_pixel
    bpp_sum = bpp_sum + bpp
    bit_all[i_count] = bit
    num_pixel_all[i_count] = num_pixel
    bpp_all[i_count] = bpp
    print('%d/%d, hw[%dx%d], bit/pixel/bpp: %d/%d/%6.3f, imgname: %s' %((i_count+1), num_img, height, width, bit, num_pixel, bpp, path_vvc))
    i_count = i_count + 1
bit_avg = bit_sum / num_img
num_pixel_avg = num_pixel_sum / num_img
bpp_avg = bpp_sum / num_img
print('avg_bit: {:.4f}, sum_bit: {:.0f}, num_image: {:.0f}'.format(bit_avg, bit_sum, num_img))
print('avg_numpixel: {:.0f}, sum_numpixel: {:.0f}, num_image: {:.0f}'.format(num_pixel_avg, num_pixel_sum, num_img))
print('avg_bpp: {:.4f}, sum_bpp: {:.4f}, num_image: {:.0f}'.format(bpp_avg, bpp_sum, num_img))




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

set_idx = 35

# filenames = glob.glob(f"/media/data/liutie/VCM/OpenImageV6-5K/*.jpg")
filenames = [f"/media/data/liutie/VCM/OpenImageV6-5K/86463a5a7dcb1a69.jpg"]
path_img_qianzhui = '/media/data/liutie/VCM/OpenImageV6-5K/'

# filenames = glob.glob(f"/media/data/ccr/COCO-val2017/*.jpg")
# filenames = [f"/media/data/ccr/COCO-val2017/000000027620.jpg"]
# path_img_qianzhui = '/media/data/ccr/COCO-val2017/'

# filenames = glob.glob(f"/media/data/liutie/VCM/rcnn/VCM_EE1.2_P-layer_feature_map_anchor_generation_137th_MPEG-VCM-main/feature/{set_idx}_rec/*.png")
num_img = len(filenames)

# path_vvc_qianzhui = f'/media/data/ccr/COCO_vtm/{set_idx}_bit/' #QP35
path_vvc_qianzhui = f'/media/data/ccr/OI5000/{set_idx}_bit/' #QP35

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

    im = Image.open(path_img).convert('RGB')
    height = im.size[1] #678
    width = im.size[0] #1024
    bit1 = filesize(path_vvc) * 8.0 #1070872
    bit = bit1

    num_pixel = height * width #694272
    bpp = bit / num_pixel #1.542
    bit_sum = bit_sum + bit
    num_pixel_sum = num_pixel_sum + num_pixel
    bpp_sum = bpp_sum + bpp
    bit_all[i_count] = bit
    num_pixel_all[i_count] = num_pixel
    bpp_all[i_count] = bpp
    print('%d/%d, hw[%dx%d], bit/pixel/bpp: %d/%d/%6.3f, imgname: %s' %((i_count+1), num_img, height, width, bit, num_pixel, bpp, path_vvc))
    i_count = i_count + 1
bit_avg = bit_sum / num_img
num_pixel_avg = num_pixel_sum / num_img
bpp_avg = bpp_sum / num_img
print('avg_bit: {:.4f}, sum_bit: {:.0f}, num_image: {:.0f}'.format(bit_avg, bit_sum, num_img))
print('avg_numpixel: {:.0f}, sum_numpixel: {:.0f}, num_image: {:.0f}'.format(num_pixel_avg, num_pixel_sum, num_img))
print('avg_bpp: {:.4f}, sum_bpp: {:.4f}, num_image: {:.0f}'.format(bpp_avg, bpp_sum, num_img))




>>>>>>> be0df11a27051c4085a72cd9a58dcd1fea1d076a
