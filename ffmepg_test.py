<<<<<<< HEAD
import os
from PIL import Image, ImageOps
import subprocess as subp
import math
import numpy as np
import cv2

file_path = '/media/data/ccr/VCM/ffmpeg_test/ori.png'
path = '/media/data/ccr/VCM/ffmpeg_test/'
file_name_anchor = 'ori'
file_name_sxc = 'sxc'
ori_stdout_fmp = open(f"{path}ffmpeg_log_ori.txt", 'w')
sxc_stdout_fmp = open(f"{path}ffmpeg_log_sxc.txt", 'w')

ds_img = Image.open(file_path)
ds_width = ds_img.size[0]
ds_height = ds_img.size[1]

if (os.path.exists(f"{file_path}_yuv.yuv")):
    os.remove(f"{file_path}_yuv.yuv")
subp.run(
    f"/media/data/liutie/ffmpeg-4.4-amd64-static/ffmpeg -i {file_path} -f rawvideo -pix_fmt gray16le -dst_range 1 {path}{file_name_anchor}_yuv.yuv",
    shell=True, stdout=ori_stdout_fmp, stderr=ori_stdout_fmp)

if (os.path.exists(f"{file_path}_rec.png")):
    os.remove(f"{file_path}_rec.png")

subp.run(
    f"/media/data/liutie/ffmpeg-4.4-amd64-static/ffmpeg -f rawvideo -pix_fmt gray16le -s {ds_width}x{ds_height} -src_range 1 -i {path}{file_name_anchor}_yuv.yuv -frames 1 -pix_fmt gray16le {path}{file_name_anchor}_rec.png",
    shell=True, stdout=ori_stdout_fmp, stderr=ori_stdout_fmp)
# #######################################################################################
# if (os.path.exists(f"{file_path}_yuv.yuv")):
#     os.remove(f"{file_path}_yuv.yuv")
# subp.run(
#     f"/media/data/liutie/ffmpeg-4.4-amd64-static/ffmpeg -i {file_path} -pix_fmt yuv420p {path}{file_name_sxc}_yuv.yuv",
#     shell=True, stdout=sxc_stdout_fmp, stderr=sxc_stdout_fmp)
#
# if (os.path.exists(f"{file_path}_rec.png")): os.remove(f"{file_path}_rec.png")
#
# subp.run(
#     f"/media/data/liutie/ffmpeg-4.4-amd64-static/ffmpeg -pix_fmt yuv420p -s {ds_width}x{ds_height} -i {path}{file_name_anchor}_yuv.yuv {path}{file_name_anchor}_rec.png",
#     shell=True, stdout=sxc_stdout_fmp, stderr=sxc_stdout_fmp)
# #######################################################################################

def get_psnr(img1, img2):
   mse = np.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

ori_img = cv2.imread(file_path, -1).astype(np.float32)
rec_img_anchor = cv2.imread(f'{path}{file_name_anchor}_rec.png', -1).astype(np.float32)
rec_img_sxc = cv2.imread(f'{path}{file_name_sxc}_rec.png', -1).astype(np.float32)

PSNR_anchor = get_psnr(ori_img, rec_img_anchor)
PSNR_sxc = get_psnr(ori_img, rec_img_sxc)

print('PSNR_anchor:', PSNR_anchor)
=======
import os
from PIL import Image, ImageOps
import subprocess as subp
import math
import numpy as np
import cv2

file_path = '/media/data/ccr/VCM/ffmpeg_test/ori.png'
path = '/media/data/ccr/VCM/ffmpeg_test/'
file_name_anchor = 'ori'
file_name_sxc = 'sxc'
ori_stdout_fmp = open(f"{path}ffmpeg_log_ori.txt", 'w')
sxc_stdout_fmp = open(f"{path}ffmpeg_log_sxc.txt", 'w')

ds_img = Image.open(file_path)
ds_width = ds_img.size[0]
ds_height = ds_img.size[1]

if (os.path.exists(f"{file_path}_yuv.yuv")):
    os.remove(f"{file_path}_yuv.yuv")
subp.run(
    f"/media/data/liutie/ffmpeg-4.4-amd64-static/ffmpeg -i {file_path} -f rawvideo -pix_fmt gray16le -dst_range 1 {path}{file_name_anchor}_yuv.yuv",
    shell=True, stdout=ori_stdout_fmp, stderr=ori_stdout_fmp)

if (os.path.exists(f"{file_path}_rec.png")):
    os.remove(f"{file_path}_rec.png")

subp.run(
    f"/media/data/liutie/ffmpeg-4.4-amd64-static/ffmpeg -f rawvideo -pix_fmt gray16le -s {ds_width}x{ds_height} -src_range 1 -i {path}{file_name_anchor}_yuv.yuv -frames 1 -pix_fmt gray16le {path}{file_name_anchor}_rec.png",
    shell=True, stdout=ori_stdout_fmp, stderr=ori_stdout_fmp)
# #######################################################################################
# if (os.path.exists(f"{file_path}_yuv.yuv")):
#     os.remove(f"{file_path}_yuv.yuv")
# subp.run(
#     f"/media/data/liutie/ffmpeg-4.4-amd64-static/ffmpeg -i {file_path} -pix_fmt yuv420p {path}{file_name_sxc}_yuv.yuv",
#     shell=True, stdout=sxc_stdout_fmp, stderr=sxc_stdout_fmp)
#
# if (os.path.exists(f"{file_path}_rec.png")): os.remove(f"{file_path}_rec.png")
#
# subp.run(
#     f"/media/data/liutie/ffmpeg-4.4-amd64-static/ffmpeg -pix_fmt yuv420p -s {ds_width}x{ds_height} -i {path}{file_name_anchor}_yuv.yuv {path}{file_name_anchor}_rec.png",
#     shell=True, stdout=sxc_stdout_fmp, stderr=sxc_stdout_fmp)
# #######################################################################################

def get_psnr(img1, img2):
   mse = np.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

ori_img = cv2.imread(file_path, -1).astype(np.float32)
rec_img_anchor = cv2.imread(f'{path}{file_name_anchor}_rec.png', -1).astype(np.float32)
rec_img_sxc = cv2.imread(f'{path}{file_name_sxc}_rec.png', -1).astype(np.float32)

PSNR_anchor = get_psnr(ori_img, rec_img_anchor)
PSNR_sxc = get_psnr(ori_img, rec_img_sxc)

print('PSNR_anchor:', PSNR_anchor)
>>>>>>> be0df11a27051c4085a72cd9a58dcd1fea1d076a
print('PSNR_sxc:', PSNR_sxc)