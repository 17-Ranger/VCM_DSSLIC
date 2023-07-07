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
        self.resid = False
        self.P3 = True
        # if self.path[3:7] == 'resid':
        #     self.resid = True


    def run(self):

        file_name = os.path.basename(self.file_path)[:-4]
        
        img = Image.open(self.file_path)

        width = img.size[0]
        height = img.size[1] 
        print('height: %d, width: %d' %(height, width))

        bitstream_path = self.path.replace("ori", "bit/")
        recon_path = self.path.replace("ori", "rec/") #decode后yuv->png的路径
        temp_path = self.path.replace("ori", "tmp/")

        if self.DSSLIC_mode == 'split':
            ds_file_path = self.file_path.replace("ori", "ds/")
            ds_recon_path = self.path.replace("ori", "ds_rec/")
            ds_temp_path = self.path.replace("ori", "ds_tmp/")
            ds_bitstream_path = self.path.replace("ori", "ds_bit/")
            ds_img = Image.open(ds_file_path)
            ds_width = ds_img.size[0]
            ds_height = ds_img.size[1]

        if self.P3 == True:
            p3_ds_file_path = self.file_path.replace("ori", "p3_ds/")
            p3_ds_recon_path = self.path.replace("ori", "p3_ds_rec/")
            p3_ds_temp_path = self.path.replace("ori", "p3_ds_tmp/")
            p3_ds_bitstream_path = self.path.replace("ori", "p3_ds_bit/")
            p3_ds_img = Image.open(p3_ds_file_path)
            p3_ds_width = p3_ds_img.size[0]
            p3_ds_height = p3_ds_img.size[1]

        stdout_vtm = open(f"{temp_path}vtm_log.txt", 'w')
        stdout_fmp = open(f"{temp_path}ffmpeg_log.txt", 'w')
        if self.DSSLIC_mode == 'split':
            ds_stdout_vtm = open(f"{ds_temp_path}vtm_log.txt", 'w')
            ds_stdout_fmp = open(f"{ds_temp_path}ffmpeg_log.txt", 'w')
        if self.P3 == True:
            p3_ds_stdout_vtm = open(f"{p3_ds_temp_path}vtm_log.txt", 'w')
            p3_ds_stdout_fmp = open(f"{p3_ds_temp_path}ffmpeg_log.txt", 'w')

        start_time = time.time()
        print(start_time, '------------------------------start time')

        ########################################ccr added 20220619
        ###################################################################ccr added ds
        if self.DSSLIC_mode == 'split':
            print('# DS------------------------------DS')
            if (os.path.exists(f"{ds_temp_path}{file_name}_yuv.yuv")): os.remove(f"{ds_temp_path}{file_name}_yuv.yuv")
            # subp.run(f"ffmpeg -i {self.file_path} -f rawvideo -pix_fmt gray16le -dst_range 1 {temp_path}{file_name}_yuv.yuv", shell = True, stdout = stdout_fmp, stderr = stdout_fmp)
            subp.run(f"/media/data/liutie/ffmpeg-4.4-amd64-static/ffmpeg -i {ds_file_path} -f rawvideo -pix_fmt gray16le -dst_range 1 {ds_temp_path}{file_name}_yuv.yuv", shell=True, stdout=ds_stdout_fmp, stderr=ds_stdout_fmp)

            # Encoding
            print('# Encoding')
            subp.run(f"./EncoderAppStatic -c ./cfg/encoder_intra_vtm.cfg -i {ds_temp_path}{file_name}_yuv.yuv -o \"\" -b {ds_bitstream_path}{file_name}.vvc -q 22 --ConformanceWindowMode=1 -wdt {ds_width} -hgt {ds_height} -f 1 -fr 1 --InternalBitDepth=10 --InputBitDepth=10 --InputChromaFormat=400 --OutputBitDepth=10", stdout=ds_stdout_vtm, shell=True)

            encode_time_ds = time.time()

            # Decoding
            print('# Decoding')
            subp.run(f"./DecoderAppStatic -b {ds_bitstream_path}{file_name}.vvc -o {ds_temp_path}{file_name}_rec.yuv", stdout=ds_stdout_vtm, shell=True)

            # Convert yuv to png
            print('# Convert yuv to png')
            if (os.path.exists(f"{ds_temp_path}{file_name}_rec.png")): os.remove(f"{ds_temp_path}{file_name}_rec.png")
            # subp.run(f"ffmpeg -f rawvideo -pix_fmt gray16le -s {width}x{height} -src_range 1 -i {temp_path}{file_name}_rec.yuv -frames 1 -pix_fmt gray16le {recon_path}{file_name}.png", shell = True, stdout = stdout_fmp, stderr = stdout_fmp)
            subp.run(f"/media/data/liutie/ffmpeg-4.4-amd64-static/ffmpeg -f rawvideo -pix_fmt gray16le -s {ds_width}x{ds_height} -src_range 1 -i {ds_temp_path}{file_name}_rec.yuv -frames 1 -pix_fmt gray16le {ds_recon_path}{file_name}.png", shell=True, stdout=ds_stdout_fmp, stderr=ds_stdout_fmp)

            decode_time_ds = time.time()
        #######################################################################################
        ###################################################################ccr added ds
        if self.P3 == True:
            if (os.path.exists(f"{p3_ds_temp_path}{file_name}_yuv.yuv")): os.remove(f"{p3_ds_temp_path}{file_name}_yuv.yuv")
            subp.run(f"/media/data/liutie/ffmpeg-4.4-amd64-static/ffmpeg -i {p3_ds_file_path} -f rawvideo -pix_fmt gray16le -dst_range 1 {p3_ds_temp_path}{file_name}_yuv.yuv", shell=True, stdout=p3_ds_stdout_fmp, stderr=p3_ds_stdout_fmp)

            # Encoding
            subp.run(f"./EncoderAppStatic -c ./cfg/encoder_intra_vtm.cfg -i {p3_ds_temp_path}{file_name}_yuv.yuv -o \"\" -b {p3_ds_bitstream_path}{file_name}.vvc -q 22 --ConformanceWindowMode=1 -wdt {p3_ds_width} -hgt {p3_ds_height} -f 1 -fr 1 --InternalBitDepth=10 --InputBitDepth=10 --InputChromaFormat=400 --OutputBitDepth=10", stdout=p3_ds_stdout_vtm, shell=True)

            encode_time_p3 = time.time()

            # Decoding
            subp.run(
                f"./DecoderAppStatic -b {p3_ds_bitstream_path}{file_name}.vvc -o {p3_ds_temp_path}{file_name}_rec.yuv",
                stdout=p3_ds_stdout_vtm, shell=True)

            # Convert yuv to png
            if (os.path.exists(f"{p3_ds_temp_path}{file_name}_rec.png")): os.remove(
                f"{p3_ds_temp_path}{file_name}_rec.png")
            # subp.run(f"ffmpeg -f rawvideo -pix_fmt gray16le -s {width}x{height} -src_range 1 -i {temp_path}{file_name}_rec.yuv -frames 1 -pix_fmt gray16le {recon_path}{file_name}.png", shell = True, stdout = stdout_fmp, stderr = stdout_fmp)
            subp.run(
                f"/media/data/liutie/ffmpeg-4.4-amd64-static/ffmpeg -f rawvideo -pix_fmt gray16le -s {p3_ds_width}x{p3_ds_height} -src_range 1 -i {p3_ds_temp_path}{file_name}_rec.yuv -frames 1 -pix_fmt gray16le {p3_ds_recon_path}{file_name}.png",
                shell=True, stdout=p3_ds_stdout_fmp, stderr=p3_ds_stdout_fmp)

            decode_time_p3 = time.time()
        # #######################################################################################

        # Convert png to yuv (ccr:origin)
        # print('# Convert png to yuv')
        if (os.path.exists(f"{temp_path}{file_name}_yuv.yuv")): os.remove(f"{temp_path}{file_name}_yuv.yuv")
        # subp.run(f"ffmpeg -i {self.file_path} -f rawvideo -pix_fmt gray16le -dst_range 1 {temp_path}{file_name}_yuv.yuv", shell = True, stdout = stdout_fmp, stderr = stdout_fmp)
        subp.run(f"/media/data/liutie/ffmpeg-4.4-amd64-static/ffmpeg -i {self.file_path} -f rawvideo -pix_fmt gray16le -dst_range 1 {temp_path}{file_name}_yuv.yuv", shell = True, stdout = stdout_fmp, stderr = stdout_fmp)

        # Encoding
        # print('# Encoding')
        subp.run(f"./EncoderAppStatic -c ./cfg/encoder_intra_vtm.cfg -i {temp_path}{file_name}_yuv.yuv -o \"\" -b {bitstream_path}{file_name}.vvc -q {self.qp} --ConformanceWindowMode=1 -wdt {width} -hgt {height} -f 1 -fr 1 --InternalBitDepth=10 --InputBitDepth=10 --InputChromaFormat=400 --OutputBitDepth=10", stdout = stdout_vtm, shell = True)

        encode_time_p45 = time.time()

        # Decoding
        # print('# Decoding')
        subp.run(f"./DecoderAppStatic -b {bitstream_path}{file_name}.vvc -o {temp_path}{file_name}_rec.yuv", stdout = stdout_vtm, shell = True)

        # Convert yuv to png
        # print('# Convert yuv to png')
        if (os.path.exists(f"{temp_path}{file_name}_rec.png")): os.remove(f"{temp_path}{file_name}_rec.png")
        # subp.run(f"ffmpeg -f rawvideo -pix_fmt gray16le -s {width}x{height} -src_range 1 -i {temp_path}{file_name}_rec.yuv -frames 1 -pix_fmt gray16le {recon_path}{file_name}.png", shell = True, stdout = stdout_fmp, stderr = stdout_fmp)
        subp.run(f"/media/data/liutie/ffmpeg-4.4-amd64-static/ffmpeg -f rawvideo -pix_fmt gray16le -s {width}x{height} -src_range 1 -i {temp_path}{file_name}_rec.yuv -frames 1 -pix_fmt gray16le {recon_path}{file_name}.png", shell = True, stdout = stdout_fmp, stderr = stdout_fmp)

        decode_time = time.time()

        ds_enc_time = encode_time_ds - start_time
        ds_dec_time = decode_time_ds - encode_time_ds
        p3_enc_time = encode_time_p3 - decode_time_ds
        p3_dec_time = decode_time_p3 - encode_time_p3
        p45_enc_time = encode_time_p45 - decode_time_p3
        p45_dec_time = decode_time - encode_time_p45

        encode_run = ds_enc_time + p3_enc_time + p45_enc_time
        decode_run = ds_dec_time + p3_dec_time + p45_dec_time
        print(ds_enc_time, ds_dec_time, p3_enc_time, p3_dec_time, p45_enc_time, p45_dec_time)
        print('Encode time', encode_run)
        print('Decode time', decode_run)
        print('total time', decode_time - start_time)


        # Remove tmp files
        try:
            os.remove(f"{temp_path}{file_name}_yuv.yuv")
            if self.DSSLIC_mode == 'split':
                os.remove(f"{ds_temp_path}{file_name}_yuv.yuv")
            if self.P3 == True:
                os.remove(f"{ds_temp_path}{file_name}_yuv.yuv")
        except OSError:
            pass
        try:
            os.remove(f"{temp_path}{file_name}_rec.yuv")
            if self.DSSLIC_mode == 'split':
                os.remove(f"{ds_temp_path}{file_name}_rec.yuv")
            if self.P3 == True:
                os.remove(f"{ds_temp_path}{file_name}_rec.yuv")
        except OSError:
            pass

def run_vtm(path, qp, threads, resid = False):
    FileList = glob.glob(f"{path}/*.png")
    bitstream_path = path.replace("ori", "bit/")
    recon_path = path.replace("ori", "rec/")
    temp_path = path.replace("ori", "tmp/")
    ###################################
    ds_file_path = path.replace("ori", "ds/")
    ds_recon_path = path.replace("ori", "ds_rec/")
    ds_temp_path = path.replace("ori", "ds_tmp/")
    ds_bitstream_path = path.replace("ori", "ds_bit/")

    p3_ds_file_path = path.replace("ori", "p3_ds/")
    p3_ds_recon_path = path.replace("ori", "p3_ds_rec/")
    p3_ds_temp_path = path.replace("ori", "p3_ds_tmp/")
    p3_ds_bitstream_path = path.replace("ori", "p3_ds_bit/")

    os.makedirs(bitstream_path, exist_ok = True)
    os.makedirs(recon_path, exist_ok = True)
    os.makedirs(temp_path, exist_ok = True)
    ###################################
    os.makedirs(ds_file_path, exist_ok=True)
    os.makedirs(ds_recon_path, exist_ok=True)
    os.makedirs(ds_temp_path, exist_ok=True)
    os.makedirs(ds_bitstream_path, exist_ok=True)

    os.makedirs(p3_ds_file_path, exist_ok=True)
    os.makedirs(p3_ds_recon_path, exist_ok=True)
    os.makedirs(p3_ds_temp_path, exist_ok=True)
    os.makedirs(p3_ds_bitstream_path, exist_ok=True)
    start_time = time.clock()
    print(start_time, '------------------------------start time')
    for file_path in tqdm(FileList):

        file_name = os.path.basename(file_path)[:-4] 
        
        if os.path.isfile(f"{recon_path}{file_name}.png"):
            print(f"{file_name} skip (exist)")
            continue

        while (threads + 1 < threading.active_count()): sleep(1)
        
        file_path = file_path.encode('utf-8','backslashreplace').decode().replace("\\","/")     

        t = Worker(file_path, path, qp, resid)
        print('1')
        t.start()