<<<<<<< HEAD
import threading
from tqdm import tqdm
import os
import glob
import argparse
from PIL import Image, ImageOps
import subprocess as subp
import sys
from time import sleep


class Worker(threading.Thread):
    def __init__(self, file_path, path, qp):
        super().__init__()
        self.file_path = file_path
        self.path = path
        self.qp = qp
        self.DSSLIC_mode = 'together'

    def run(self):

        file_name = os.path.basename(self.file_path)[:-4]

        img = Image.open(self.file_path)

        width = img.size[0]
        height = img.size[1]
        print('height: %d, width: %d' % (height, width))

        bitstream_path = self.path.replace("ori", "bit/")
        recon_path = self.path.replace("ori", "rec/")  # decode后yuv->png的路径
        temp_path = self.path.replace("ori", "tmp/")

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

        ########################################ccr added 20220619
        ###################################################################ccr added ds
        if self.DSSLIC_mode == 'split':
            print('# DS------------------------------DS')
            if (os.path.exists(f"{ds_temp_path}{file_name}_yuv.yuv")): os.remove(f"{ds_temp_path}{file_name}_yuv.yuv")
            # subp.run(f"ffmpeg -i {self.file_path} -f rawvideo -pix_fmt gray16le -dst_range 1 {temp_path}{file_name}_yuv.yuv", shell = True, stdout = stdout_fmp, stderr = stdout_fmp)
            subp.run(
                f"/media/data/liutie/ffmpeg-4.4-amd64-static/ffmpeg -i {ds_file_path} -f rawvideo -pix_fmt gray16le -dst_range 1 {ds_temp_path}{file_name}_yuv.yuv",
                shell=True, stdout=ds_stdout_fmp, stderr=ds_stdout_fmp)

            # Encoding
            print('# Encoding')
            subp.run(
                f"./EncoderAppStatic -c ./cfg/encoder_intra_vtm.cfg -i {ds_temp_path}{file_name}_yuv.yuv -o \"\" -b {ds_bitstream_path}{file_name}.vvc -q {self.qp} --ConformanceWindowMode=1 -wdt {ds_width} -hgt {ds_height} -f 1 -fr 1 --InternalBitDepth=10 --InputBitDepth=10 --InputChromaFormat=400 --OutputBitDepth=10",
                stdout=ds_stdout_vtm, shell=True)

            # Decoding
            print('# Decoding')
            subp.run(f"./DecoderAppStatic -b {ds_bitstream_path}{file_name}.vvc -o {ds_temp_path}{file_name}_rec.yuv",
                     stdout=ds_stdout_vtm, shell=True)

            # Convert yuv to png
            print('# Convert yuv to png')
            if (os.path.exists(f"{ds_temp_path}{file_name}_rec.png")): os.remove(f"{ds_temp_path}{file_name}_rec.png")
            # subp.run(f"ffmpeg -f rawvideo -pix_fmt gray16le -s {width}x{height} -src_range 1 -i {temp_path}{file_name}_rec.yuv -frames 1 -pix_fmt gray16le {recon_path}{file_name}.png", shell = True, stdout = stdout_fmp, stderr = stdout_fmp)
            subp.run(
                f"/media/data/liutie/ffmpeg-4.4-amd64-static/ffmpeg -f rawvideo -pix_fmt gray16le -s {ds_width}x{ds_height} -src_range 1 -i {ds_temp_path}{file_name}_rec.yuv -frames 1 -pix_fmt gray16le {ds_recon_path}{file_name}.png",
                shell=True, stdout=ds_stdout_fmp, stderr=ds_stdout_fmp)

        # Convert png to yuv (ccr:origin)
        print('# Convert png to yuv')
        if (os.path.exists(f"{temp_path}{file_name}_yuv.yuv")): os.remove(f"{temp_path}{file_name}_yuv.yuv")
        # subp.run(f"ffmpeg -i {self.file_path} -f rawvideo -pix_fmt gray16le -dst_range 1 {temp_path}{file_name}_yuv.yuv", shell = True, stdout = stdout_fmp, stderr = stdout_fmp)
        subp.run(
            f"/media/data/liutie/ffmpeg-4.4-amd64-static/ffmpeg -i {self.file_path} -f rawvideo -pix_fmt gray16le -dst_range 1 {temp_path}{file_name}_yuv.yuv",
            shell=True, stdout=stdout_fmp, stderr=stdout_fmp)

        # Encoding
        print('# Encoding')
        subp.run(
            f"./EncoderAppStatic -c ./cfg/encoder_intra_vtm.cfg -i {temp_path}{file_name}_yuv.yuv -o \"\" -b {bitstream_path}{file_name}.vvc -q {self.qp} --ConformanceWindowMode=1 -wdt {width} -hgt {height} -f 1 -fr 1 --InternalBitDepth=10 --InputBitDepth=10 --InputChromaFormat=400 --OutputBitDepth=10",
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
            f"/media/data/liutie/ffmpeg-4.4-amd64-static/ffmpeg -f rawvideo -pix_fmt gray16le -s {width}x{height} -src_range 1 -i {temp_path}{file_name}_rec.yuv -frames 1 -pix_fmt gray16le {recon_path}{file_name}.png",
            shell=True, stdout=stdout_fmp, stderr=stdout_fmp)

        # ###################################################################ccr added resid
        # print('# Resid---------------------------Resid')
        # if (os.path.exists(f"{resid_temp_path}{file_name}_yuv.yuv")): os.remove(f"{resid_temp_path}{file_name}_yuv.yuv")
        # # subp.run(f"ffmpeg -i {self.file_path} -f rawvideo -pix_fmt gray16le -dst_range 1 {temp_path}{file_name}_yuv.yuv", shell = True, stdout = stdout_fmp, stderr = stdout_fmp)
        # subp.run(f"/media/data/liutie/ffmpeg-4.4-amd64-static/ffmpeg -i {resid_file_path} -f rawvideo -pix_fmt gray16le -dst_range 1 {resid_temp_path}{file_name}_yuv.yuv", shell=True, stdout=resid_stdout_fmp, stderr=resid_stdout_fmp)
        #
        # # Encoding
        # print('# Encoding')
        # subp.run(f"./EncoderAppStatic -c ./cfg/encoder_intra_vtm.cfg -i {resid_temp_path}{file_name}_yuv.yuv -o \"\" -b {resid_bitstream_path}{file_name}.vvc -q {self.qp} --ConformanceWindowMode=1 -wdt {resid_width} -hgt {resid_height} -f 1 -fr 1 --InternalBitDepth=10 --InputBitDepth=10 --InputChromaFormat=400 --OutputBitDepth=10", stdout=resid_stdout_vtm, shell=True)
        #
        # # Decoding
        # print('# Decoding')
        # subp.run(f"./DecoderAppStatic -b {resid_bitstream_path}{file_name}.vvc -o {resid_temp_path}{file_name}_rec.yuv", stdout=resid_stdout_vtm, shell=True)
        #
        # # Convert yuv to png
        # print('# Convert yuv to png')
        # if (os.path.exists(f"{resid_temp_path}{file_name}_rec.png")): os.remove(f"{resid_temp_path}{file_name}_rec.png")
        # # subp.run(f"ffmpeg -f rawvideo -pix_fmt gray16le -s {width}x{height} -src_range 1 -i {temp_path}{file_name}_rec.yuv -frames 1 -pix_fmt gray16le {recon_path}{file_name}.png", shell = True, stdout = stdout_fmp, stderr = stdout_fmp)
        # subp.run(f"/media/data/liutie/ffmpeg-4.4-amd64-static/ffmpeg -f rawvideo -pix_fmt gray16le -s {resid_width}x{resid_height} -src_range 1 -i {resid_temp_path}{file_name}_rec.yuv -frames 1 -pix_fmt gray16le {resid_recon_path}{file_name}.png", shell=True, stdout=resid_stdout_fmp, stderr=resid_stdout_fmp)

        ############################################################################ccr added 20220619

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


def run_vtm(path, qp, threads):
    FileList = glob.glob(f"{path}/*.png")
    bitstream_path = path.replace("ori", "bit/")
    recon_path = path.replace("ori", "rec/")
    temp_path = path.replace("ori", "tmp/")
    ###################################
    ds_file_path = path.replace("ori", "ds/")
    ds_recon_path = path.replace("ori", "ds_rec/")
    ds_temp_path = path.replace("ori", "ds_tmp/")
    ds_bitstream_path = path.replace("ori", "ds_bit/")

    resid_file_path = path.replace("ori", "resid/")
    resid_recon_path = path.replace("ori", "resid_rec/")
    resid_temp_path = path.replace("ori", "resid_tmp/")
    resid_bitstream_path = path.replace("ori", "resid_bit/")

    os.makedirs(bitstream_path, exist_ok=True)
    os.makedirs(recon_path, exist_ok=True)
    os.makedirs(temp_path, exist_ok=True)
    ###################################
    os.makedirs(ds_file_path, exist_ok=True)
    os.makedirs(ds_recon_path, exist_ok=True)
    os.makedirs(ds_temp_path, exist_ok=True)
    os.makedirs(ds_bitstream_path, exist_ok=True)

    os.makedirs(resid_file_path, exist_ok=True)
    os.makedirs(resid_recon_path, exist_ok=True)
    os.makedirs(resid_temp_path, exist_ok=True)
    os.makedirs(resid_bitstream_path, exist_ok=True)

    for file_path in tqdm(FileList):

        file_name = os.path.basename(file_path)[:-4]

        if os.path.isfile(f"{recon_path}{file_name}.png"):
            print(f"{file_name} skip (exist)")
            continue

        while (threads + 1 < threading.active_count()): sleep(1)

        file_path = file_path.encode('utf-8', 'backslashreplace').decode().replace("\\", "/")

        t = Worker(file_path, path, qp)

=======
import threading
from tqdm import tqdm
import os
import glob
import argparse
from PIL import Image, ImageOps
import subprocess as subp
import sys
from time import sleep


class Worker(threading.Thread):
    def __init__(self, file_path, path, qp):
        super().__init__()
        self.file_path = file_path
        self.path = path
        self.qp = qp
        self.DSSLIC_mode = 'together'

    def run(self):

        file_name = os.path.basename(self.file_path)[:-4]

        img = Image.open(self.file_path)

        width = img.size[0]
        height = img.size[1]
        print('height: %d, width: %d' % (height, width))

        bitstream_path = self.path.replace("ori", "bit/")
        recon_path = self.path.replace("ori", "rec/")  # decode后yuv->png的路径
        temp_path = self.path.replace("ori", "tmp/")

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

        ########################################ccr added 20220619
        ###################################################################ccr added ds
        if self.DSSLIC_mode == 'split':
            print('# DS------------------------------DS')
            if (os.path.exists(f"{ds_temp_path}{file_name}_yuv.yuv")): os.remove(f"{ds_temp_path}{file_name}_yuv.yuv")
            # subp.run(f"ffmpeg -i {self.file_path} -f rawvideo -pix_fmt gray16le -dst_range 1 {temp_path}{file_name}_yuv.yuv", shell = True, stdout = stdout_fmp, stderr = stdout_fmp)
            subp.run(
                f"/media/data/liutie/ffmpeg-4.4-amd64-static/ffmpeg -i {ds_file_path} -f rawvideo -pix_fmt gray16le -dst_range 1 {ds_temp_path}{file_name}_yuv.yuv",
                shell=True, stdout=ds_stdout_fmp, stderr=ds_stdout_fmp)

            # Encoding
            print('# Encoding')
            subp.run(
                f"./EncoderAppStatic -c ./cfg/encoder_intra_vtm.cfg -i {ds_temp_path}{file_name}_yuv.yuv -o \"\" -b {ds_bitstream_path}{file_name}.vvc -q {self.qp} --ConformanceWindowMode=1 -wdt {ds_width} -hgt {ds_height} -f 1 -fr 1 --InternalBitDepth=10 --InputBitDepth=10 --InputChromaFormat=400 --OutputBitDepth=10",
                stdout=ds_stdout_vtm, shell=True)

            # Decoding
            print('# Decoding')
            subp.run(f"./DecoderAppStatic -b {ds_bitstream_path}{file_name}.vvc -o {ds_temp_path}{file_name}_rec.yuv",
                     stdout=ds_stdout_vtm, shell=True)

            # Convert yuv to png
            print('# Convert yuv to png')
            if (os.path.exists(f"{ds_temp_path}{file_name}_rec.png")): os.remove(f"{ds_temp_path}{file_name}_rec.png")
            # subp.run(f"ffmpeg -f rawvideo -pix_fmt gray16le -s {width}x{height} -src_range 1 -i {temp_path}{file_name}_rec.yuv -frames 1 -pix_fmt gray16le {recon_path}{file_name}.png", shell = True, stdout = stdout_fmp, stderr = stdout_fmp)
            subp.run(
                f"/media/data/liutie/ffmpeg-4.4-amd64-static/ffmpeg -f rawvideo -pix_fmt gray16le -s {ds_width}x{ds_height} -src_range 1 -i {ds_temp_path}{file_name}_rec.yuv -frames 1 -pix_fmt gray16le {ds_recon_path}{file_name}.png",
                shell=True, stdout=ds_stdout_fmp, stderr=ds_stdout_fmp)

        # Convert png to yuv (ccr:origin)
        print('# Convert png to yuv')
        if (os.path.exists(f"{temp_path}{file_name}_yuv.yuv")): os.remove(f"{temp_path}{file_name}_yuv.yuv")
        # subp.run(f"ffmpeg -i {self.file_path} -f rawvideo -pix_fmt gray16le -dst_range 1 {temp_path}{file_name}_yuv.yuv", shell = True, stdout = stdout_fmp, stderr = stdout_fmp)
        subp.run(
            f"/media/data/liutie/ffmpeg-4.4-amd64-static/ffmpeg -i {self.file_path} -f rawvideo -pix_fmt gray16le -dst_range 1 {temp_path}{file_name}_yuv.yuv",
            shell=True, stdout=stdout_fmp, stderr=stdout_fmp)

        # Encoding
        print('# Encoding')
        subp.run(
            f"./EncoderAppStatic -c ./cfg/encoder_intra_vtm.cfg -i {temp_path}{file_name}_yuv.yuv -o \"\" -b {bitstream_path}{file_name}.vvc -q {self.qp} --ConformanceWindowMode=1 -wdt {width} -hgt {height} -f 1 -fr 1 --InternalBitDepth=10 --InputBitDepth=10 --InputChromaFormat=400 --OutputBitDepth=10",
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
            f"/media/data/liutie/ffmpeg-4.4-amd64-static/ffmpeg -f rawvideo -pix_fmt gray16le -s {width}x{height} -src_range 1 -i {temp_path}{file_name}_rec.yuv -frames 1 -pix_fmt gray16le {recon_path}{file_name}.png",
            shell=True, stdout=stdout_fmp, stderr=stdout_fmp)

        # ###################################################################ccr added resid
        # print('# Resid---------------------------Resid')
        # if (os.path.exists(f"{resid_temp_path}{file_name}_yuv.yuv")): os.remove(f"{resid_temp_path}{file_name}_yuv.yuv")
        # # subp.run(f"ffmpeg -i {self.file_path} -f rawvideo -pix_fmt gray16le -dst_range 1 {temp_path}{file_name}_yuv.yuv", shell = True, stdout = stdout_fmp, stderr = stdout_fmp)
        # subp.run(f"/media/data/liutie/ffmpeg-4.4-amd64-static/ffmpeg -i {resid_file_path} -f rawvideo -pix_fmt gray16le -dst_range 1 {resid_temp_path}{file_name}_yuv.yuv", shell=True, stdout=resid_stdout_fmp, stderr=resid_stdout_fmp)
        #
        # # Encoding
        # print('# Encoding')
        # subp.run(f"./EncoderAppStatic -c ./cfg/encoder_intra_vtm.cfg -i {resid_temp_path}{file_name}_yuv.yuv -o \"\" -b {resid_bitstream_path}{file_name}.vvc -q {self.qp} --ConformanceWindowMode=1 -wdt {resid_width} -hgt {resid_height} -f 1 -fr 1 --InternalBitDepth=10 --InputBitDepth=10 --InputChromaFormat=400 --OutputBitDepth=10", stdout=resid_stdout_vtm, shell=True)
        #
        # # Decoding
        # print('# Decoding')
        # subp.run(f"./DecoderAppStatic -b {resid_bitstream_path}{file_name}.vvc -o {resid_temp_path}{file_name}_rec.yuv", stdout=resid_stdout_vtm, shell=True)
        #
        # # Convert yuv to png
        # print('# Convert yuv to png')
        # if (os.path.exists(f"{resid_temp_path}{file_name}_rec.png")): os.remove(f"{resid_temp_path}{file_name}_rec.png")
        # # subp.run(f"ffmpeg -f rawvideo -pix_fmt gray16le -s {width}x{height} -src_range 1 -i {temp_path}{file_name}_rec.yuv -frames 1 -pix_fmt gray16le {recon_path}{file_name}.png", shell = True, stdout = stdout_fmp, stderr = stdout_fmp)
        # subp.run(f"/media/data/liutie/ffmpeg-4.4-amd64-static/ffmpeg -f rawvideo -pix_fmt gray16le -s {resid_width}x{resid_height} -src_range 1 -i {resid_temp_path}{file_name}_rec.yuv -frames 1 -pix_fmt gray16le {resid_recon_path}{file_name}.png", shell=True, stdout=resid_stdout_fmp, stderr=resid_stdout_fmp)

        ############################################################################ccr added 20220619

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


def run_vtm(path, qp, threads):
    FileList = glob.glob(f"{path}/*.png")
    bitstream_path = path.replace("ori", "bit/")
    recon_path = path.replace("ori", "rec/")
    temp_path = path.replace("ori", "tmp/")
    ###################################
    ds_file_path = path.replace("ori", "ds/")
    ds_recon_path = path.replace("ori", "ds_rec/")
    ds_temp_path = path.replace("ori", "ds_tmp/")
    ds_bitstream_path = path.replace("ori", "ds_bit/")

    resid_file_path = path.replace("ori", "resid/")
    resid_recon_path = path.replace("ori", "resid_rec/")
    resid_temp_path = path.replace("ori", "resid_tmp/")
    resid_bitstream_path = path.replace("ori", "resid_bit/")

    os.makedirs(bitstream_path, exist_ok=True)
    os.makedirs(recon_path, exist_ok=True)
    os.makedirs(temp_path, exist_ok=True)
    ###################################
    os.makedirs(ds_file_path, exist_ok=True)
    os.makedirs(ds_recon_path, exist_ok=True)
    os.makedirs(ds_temp_path, exist_ok=True)
    os.makedirs(ds_bitstream_path, exist_ok=True)

    os.makedirs(resid_file_path, exist_ok=True)
    os.makedirs(resid_recon_path, exist_ok=True)
    os.makedirs(resid_temp_path, exist_ok=True)
    os.makedirs(resid_bitstream_path, exist_ok=True)

    for file_path in tqdm(FileList):

        file_name = os.path.basename(file_path)[:-4]

        if os.path.isfile(f"{recon_path}{file_name}.png"):
            print(f"{file_name} skip (exist)")
            continue

        while (threads + 1 < threading.active_count()): sleep(1)

        file_path = file_path.encode('utf-8', 'backslashreplace').decode().replace("\\", "/")

        t = Worker(file_path, path, qp)

>>>>>>> be0df11a27051c4085a72cd9a58dcd1fea1d076a
        t.start()