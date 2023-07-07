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
import time

class Worker(threading.Thread):
    def __init__(self, file_path, path, qp, resid):
        super().__init__()
        self.file_path = file_path
        self.path = path
        self.qp = qp
        self.DSSLIC_mode = 'split'

    def run(self):
        file_name = os.path.basename(self.file_path)[:-4]
        resize_filepath = f'/media/data/ccr/OI5000_resized/{os.path.basename(self.file_path)}'
        new_img = Image.open(resize_filepath)

        width = new_img.size[0]
        height = new_img.size[1]
        print('height: %d, width: %d' % (height, width))

        bitstream_path = self.path.replace('/media/data/ccr/OI10_resized', f"/media/data/ccr/OI5000/{self.qp}test_bit/")
        recon_path = self.path.replace('/media/data/ccr/OI10_resized', f"/media/data/ccr/OI5000/{self.qp}test_rec/")
        temp_path = self.path.replace('/media/data/ccr/OI10_resized', f"/media/data/ccr/OI5000/{self.qp}test_tmp/")


        stdout_vtm = open(f"{temp_path}vtm_log.txt", 'w')
        stdout_fmp = open(f"{temp_path}ffmpeg_log.txt", 'w')

        start_time = time.time()
        print(start_time, '------------------------------start time')


        # Convert png to yuv (ccr:origin)
        print('# Convert png to yuv')
        if (os.path.exists(f"{temp_path}{file_name}_yuv.yuv")): os.remove(f"{temp_path}{file_name}_yuv.yuv")

        subp.run(
            f"/media/data/liutie/ffmpeg-4.4-amd64-static/ffmpeg -i {resize_filepath} -pix_fmt yuv420p -dst_range 1 {temp_path}{file_name}_yuv.yuv",
            shell=True, stdout=stdout_fmp, stderr=stdout_fmp)

        # Encoding
        print('# Encoding')
        subp.run(
            f"./EncoderAppStatic -c ./cfg/encoder_intra_vtm.cfg -i {temp_path}{file_name}_yuv.yuv -b {bitstream_path}{file_name}.vvc -q {self.qp} -wdt {width} -hgt {height} -f 1 -fr 1 --InternalBitDepth=10 --ConformanceWindowMode=1",
            stdout=stdout_vtm, shell=True)

        encode_time = time.time()

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

        decode_time = time.time()
        encode_run = encode_time - start_time
        decode_run = decode_time - encode_time
        print('Encode time', encode_run)
        print('Decode time', decode_run)


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
    bitstream_path = path.replace('/media/data/ccr/OI10_resized', f"/media/data/ccr/OI5000/{qp}test_bit/")
    recon_path = path.replace('/media/data/ccr/OI10_resized', f"/media/data/ccr/OI5000/{qp}test_rec/")
    temp_path = path.replace('/media/data/ccr/OI10_resized', f"/media/data/ccr/OI5000/{qp}test_tmp/")
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

# all_resize()     #reize
run_vtm('/media/data/ccr/OI10_resized', 22, 25)
# run_vtm('/media/data/ccr/OI5000_resized', 32, 30)



