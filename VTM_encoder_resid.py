import threading
from tqdm import tqdm
import os
import glob
import argparse
from PIL import Image, ImageOps
import subprocess as subp
import sys
from time import sleep
import time

class Worker(threading.Thread):
    def __init__(self, file_path, path, qp, resid):
        super().__init__()
        self.file_path = file_path
        self.path = path
        self.qp = qp
        self.DSSLIC_mode = 'split'
        self.resid = True
        # if self.path[3] != 'r':
        #     self.resid = False

    def run(self):

        file_name = os.path.basename(self.file_path)[:-4]
        
        img = Image.open(self.file_path)

        width = img.size[0]
        height = img.size[1] 
        print('height: %d, width: %d' %(height, width))

        bitstream_path = self.path.replace("ori", "bit/")
        recon_path = self.path.replace("ori", "rec/") #decode后yuv->png的路径
        temp_path = self.path.replace("ori", "tmp/")

        if self.resid == True:
            self.DSSLIC_mode = 'together'
            bitstream_path = self.path.replace("resid", "resid_bit/")
            recon_path = self.path.replace("resid", "resid_rec/")  # decode后yuv->png的路径
            temp_path = self.path.replace("resid", "resid_tmp/")

        if self.DSSLIC_mode == 'split':
            ds_file_path = self.file_path.replace("ori", "ds/")
            ds_recon_path = self.path.replace("ori", "ds_rec/")
            ds_temp_path = self.path.replace("ori", "ds_tmp/")
            ds_bitstream_path = self.path.replace("ori", "ds_bit/")
            ds_img = Image.open(ds_file_path)
            ds_width = ds_img.size[0]
            ds_height = ds_img.size[1]

        # resid_file_path = self.file_path.replace("ori", "resid/")
        # resid_recon_path = self.path.replace("ori", "resid_rec/")
        # resid_temp_path = self.path.replace("ori", "resid_tmp/")
        # resid_bitstream_path = self.path.replace("ori", "resid_bit/")
        # resid_img = Image.open(resid_file_path)
        # resid_width = resid_img.size[0]
        # resid_height = resid_img.size[1]
        print('sssssssssssssssssssssssssssssssssssssssssss')
        stdout_vtm = open(f"{temp_path}vtm_log.txt", 'w')
        stdout_fmp = open(f"{temp_path}ffmpeg_log.txt", 'w')
        if self.DSSLIC_mode == 'split':
            ds_stdout_vtm = open(f"{ds_temp_path}vtm_log.txt", 'w')
            ds_stdout_fmp = open(f"{ds_temp_path}ffmpeg_log.txt", 'w')
        # resid_stdout_vtm = open(f"{resid_temp_path}vtm_log.txt", 'w')
        # resid_stdout_fmp = open(f"{resid_temp_path}ffmpeg_log.txt", 'w')

        start_time = time.time()
        print(start_time, '------------------------------start time')

        if self.qp == 35:
            realQP = 35
        elif self.qp == 37:
            realQP = 38
        elif self.qp == 39:
            realQP = 41
        elif self.qp == 41:
            realQP = 44
        elif self.qp == 43:
            realQP = 47
        elif self.qp == 45:
            realQP = 50
        else:
            realQP = self.qp
        ########################################ccr added 20220619
        ###################################################################ccr added ds
        # Convert png to yuv (ccr:origin)
        print('# Convert png to yuv')
        if (os.path.exists(f"{temp_path}{file_name}_yuv.yuv")): os.remove(f"{temp_path}{file_name}_yuv.yuv")
        # subp.run(f"ffmpeg -i {self.file_path} -f rawvideo -pix_fmt gray16le -dst_range 1 {temp_path}{file_name}_yuv.yuv", shell = True, stdout = stdout_fmp, stderr = stdout_fmp)
        subp.run(f"/media/data/liutie/ffmpeg-4.4-amd64-static/ffmpeg -i {self.file_path} -f rawvideo -pix_fmt gray16le -dst_range 1 {temp_path}{file_name}_yuv.yuv", shell = True, stdout = stdout_fmp, stderr = stdout_fmp)

        # Encoding
        print('# Encoding')
        subp.run(f"./EncoderAppStatic -c ./cfg/encoder_intra_vtm.cfg -i {temp_path}{file_name}_yuv.yuv -o \"\" -b {bitstream_path}{file_name}.vvc -q {realQP} --ConformanceWindowMode=1 -wdt {width} -hgt {height} -f 1 -fr 1 --InternalBitDepth=10 --InputBitDepth=10 --InputChromaFormat=400 --OutputBitDepth=10", stdout = stdout_vtm, shell = True)

        encode_time = time.time()

        # Decoding
        print('# Decoding')
        subp.run(f"./DecoderAppStatic -b {bitstream_path}{file_name}.vvc -o {temp_path}{file_name}_rec.yuv", stdout = stdout_vtm, shell = True)

        # Convert yuv to png
        print('# Convert yuv to png')
        if (os.path.exists(f"{temp_path}{file_name}_rec.png")): os.remove(f"{temp_path}{file_name}_rec.png")
        # subp.run(f"ffmpeg -f rawvideo -pix_fmt gray16le -s {width}x{height} -src_range 1 -i {temp_path}{file_name}_rec.yuv -frames 1 -pix_fmt gray16le {recon_path}{file_name}.png", shell = True, stdout = stdout_fmp, stderr = stdout_fmp)
        subp.run(f"/media/data/liutie/ffmpeg-4.4-amd64-static/ffmpeg -f rawvideo -pix_fmt gray16le -s {width}x{height} -src_range 1 -i {temp_path}{file_name}_rec.yuv -frames 1 -pix_fmt gray16le {recon_path}{file_name}.png", shell = True, stdout = stdout_fmp, stderr = stdout_fmp)

        end_time = time.time()
        encode = encode_time - start_time
        decode = end_time - encode_time
        print('encode', encode, '--------- decode', decode)
        print('total time', end_time-start_time)


        # Remove tmp files
        try:
            os.remove(f"{temp_path}{file_name}_yuv.yuv")
            if self.DSSLIC_mode == 'split':
                os.remove(f"{ds_temp_path}{file_name}_yuv.yuv")
                # os.remove(f"{resid_temp_path}{file_name}_yuv.yuv")
        except OSError:
            pass
        try:
            os.remove(f"{temp_path}{file_name}_rec.yuv")
            if self.DSSLIC_mode == 'split':
                os.remove(f"{ds_temp_path}{file_name}_yuv.yuv")
                # os.remove(f"{resid_temp_path}{file_name}_yuv.yuv")
        except OSError:
            pass

def run_vtm_resid(path, qp, threads, resid=True):
    FileList = glob.glob(f"{path}/*.png")
    bitstream_path = path.replace("resid", "resid_bit/")
    recon_path = path.replace("resid", "resid_rec/")
    temp_path = path.replace("resid", "resid_tmp/")
    ###################################
    os.makedirs(bitstream_path, exist_ok=True)
    os.makedirs(recon_path, exist_ok=True)
    os.makedirs(temp_path, exist_ok=True)
    ###################################
    for file_path in tqdm(FileList):
        file_name = os.path.basename(file_path)[:-4]
        if os.path.isfile(f"{recon_path}{file_name}.png"):
            print(f"{file_name} skip (exist)")
            continue
        while (threads + 1 < threading.active_count()): sleep(1)
        file_path = file_path.encode('utf-8', 'backslashreplace').decode().replace("\\", "/")
        t = Worker(file_path, path, qp, resid)
        t.start()