import threading
from tqdm import tqdm
import os
import glob
import argparse
from PIL import Image, ImageOps
import subprocess as subp
import sys
from time import sleep
import cv2
# from comparesize import resize_back

class Worker(threading.Thread):
    def __init__(self, file_path, path, qp, resid):
        super().__init__()
        self.file_path = file_path
        self.path = path
        self.qp = qp
        self.DSSLIC_mode = 'split'

    def run(self):
        file_name = os.path.basename(self.file_path)[:-4]
        resize_filepath = f'/media/data/ccr/COCO_vtm/resize_img/{os.path.basename(self.file_path)}'
        new_img = Image.open(resize_filepath)

        width = new_img.size[0]
        height = new_img.size[1]
        print('height: %d, width: %d' % (height, width))

        bitstream_path = self.path.replace("resize_img", f"{self.qp}_bit/")
        recon_path = self.path.replace("resize_img", f"{self.qp}_rec/")
        temp_path = self.path.replace("resize_img", f"{self.qp}_tmp/")
        stdout_vtm = open(f"{temp_path}vtm_log.txt", 'w')
        stdout_fmp = open(f"{temp_path}ffmpeg_log.txt", 'w')
        # Convert png to yuv (ccr:origin)
        print('# Convert png to yuv')
        if (os.path.exists(f"{temp_path}{file_name}_yuv.yuv")): os.remove(f"{temp_path}{file_name}_yuv.yuv")
        # subp.run(f"ffmpeg -i {self.file_path} -f rawvideo -pix_fmt gray16le -dst_range 1 {temp_path}{file_name}_yuv.yuv", shell = True, stdout = stdout_fmp, stderr = stdout_fmp)

        subp.run(
            f"/media/data/liutie/ffmpeg-4.4-amd64-static/ffmpeg -i {resize_filepath} -pix_fmt yuv420p -dst_range 1 {temp_path}{file_name}_yuv.yuv",
            shell=True, stdout=stdout_fmp, stderr=stdout_fmp)

        # Encoding
        print('# Encoding')
        subp.run(
            f"./EncoderAppStatic -c ./cfg/encoder_intra_vtm.cfg -i {temp_path}{file_name}_yuv.yuv -b {bitstream_path}{file_name}.vvc -q {self.qp} -wdt {width} -hgt {height} -f 1 -fr 1 --InternalBitDepth=10 --ConformanceWindowMode=1",
            stdout=stdout_vtm, shell=True)

        # Decoding
        print('# Decoding')
        subp.run(f"./DecoderAppStatic -b {bitstream_path}{file_name}.vvc -o {temp_path}{file_name}_rec.yuv",
                 stdout=stdout_vtm, shell=True)

        # Convert yuv to png
        print('# Convert yuv to png')
        if (os.path.exists(f"{temp_path}{file_name}_rec.png")): os.remove(f"{temp_path}{file_name}_rec.png")
        # subp.run(f"ffmpeg -f rawvideo -pix_fmt gray16le -s {width}x{height} -src_range 1 -i {temp_path}{file_name}_rec.yuv -frames 1 -pix_fmt gray16le {recon_path}{file_name}.png", shell = True, stdout = stdout_fmp, stderr = stdout_fmp)
        subp.run(
            f"/media/data/liutie/ffmpeg-4.4-amd64-static/ffmpeg -pix_fmt yuv420p10le -s {width}x{height} -src_range 1 -i {temp_path}{file_name}_rec.yuv -frames 1 {recon_path}{file_name}.jpg",
            shell=True, stdout=stdout_fmp, stderr=stdout_fmp)

        # Remove tmp files
        try:
            os.remove(f"{temp_path}{file_name}_yuv.yuv")
        except OSError:
            pass
        try:
            os.remove(f"{temp_path}{file_name}_rec.yuv")
        except OSError:
            pass


def run_vtm(path, qp, threads, resid=False):
    FileList = glob.glob(f"{path}/*.jpg")
    bitstream_path = path.replace("resize_img", f"{qp}_bit/")
    recon_path = path.replace("resize_img", f"{qp}_rec/")
    temp_path = path.replace("resize_img", f"{qp}_tmp/")
    ###################################

    os.makedirs(bitstream_path, exist_ok=True)
    os.makedirs(recon_path, exist_ok=True)
    os.makedirs(temp_path, exist_ok=True)

    for file_path in tqdm(FileList):
        file_name = os.path.basename(file_path)[:-4]
        if os.path.isfile(f"{recon_path}{file_name}.png"):
            print(f"{file_name} skip (exist)")
            continue
        while (threads + 1 < threading.active_count()): sleep(1)
        file_path = file_path.encode('utf-8', 'backslashreplace').decode().replace("\\", "/")
        t = Worker(file_path, path, qp, resid)
        t.start()

def all_resize():
    DATADIR = '/media/data/ccr/COCO-4660/'
    # resize_filename = f'/media/data/ccr/COCO_vtm/resize_img/{file_name}'
    path=os.path.join(DATADIR)
    img_list=os.listdir(path)
    ind=0
    for i in img_list:
      img_array=cv2.imread(os.path.join(path,i))

      if img_array.shape[0] % 8 != 0 or img_array.shape[1] % 8 != 0:
          new_height = img_array.shape[0]
          new_width = img_array.shape[1]
          if img_array.shape[0] % 8 != 0:
              new_height = img_array.shape[0] + (8 - (img_array.shape[0] % 8))

          if img_array.shape[1] % 8 != 0:
              new_width = img_array.shape[1] + (8 - (img_array.shape[1] % 8))

      new_array=cv2.resize(img_array, (new_width, new_height))
      save_path = f'/media/data/ccr/COCO_vtm/resize_img/{i}'
      ind=ind+1
      print(ind)
      cv2.imwrite(save_path,new_array)

# all_resize()     #reize
# subp.run(run_vtm("/media/data/ccr/COCO_vtm/resize_img", 42, 24))
# subp.run(run_vtm("/media/data/ccr/COCO_vtm/resize_img", 37, 24))
# subp.run(run_vtm("/media/data/ccr/COCO_vtm/resize_img", 32, 24))
run_vtm("/media/data/ccr/COCO_vtm/resize_img", 47, 20)
run_vtm("/media/data/ccr/COCO_vtm/resize_img", 37, 20)

# subp.run(run_vtm("/media/data/ccr/COCO_vtm/resize_img", 32, 25))
# subp.run(resize_back('/media/data/ccr/COCO-4660/', '/media/data/ccr/COCO_vtm/42_rec/', f'/media/data/ccr/COCO_vtm/42_resize/'))
# subp.run(resize_back('/media/data/ccr/COCO-4660/', '/media/data/ccr/COCO_vtm/32_rec/', f'/media/data/ccr/COCO_vtm/32_resize/'))
