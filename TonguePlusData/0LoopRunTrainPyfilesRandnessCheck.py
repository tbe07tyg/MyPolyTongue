import os
from glob import glob
import numpy as np
try:
    def run_case(file_dir,
                 global_randomness_count,
                 rotation_range=90,
                 width_shift_range = 0.3,
                 height_shift_range = 0.3,
                 zoom_range = 0.2,
                 shear_range=0.3,
                 brightness_range_start=0.5,
                 brightness_range_stop=0.3,
                 max_run=5,

                 ):
        """
        # rotation_range = 45
        # width_shift_range = 0.3
        # height_shift_range = 0.3
        # zoom_range = 0.2
        # shear_range=0.35
        # brightness_range=[0.5, 1.5]

        :param file_dir:
        :param rotation_rangemax_run:
        :return:
        """
        fileIdx =1
        # for root, dirs, files in os.walk(file_dir):
        #     print(files, "to be run")
        #     for i in files:
        all_files = glob(file_dir + '/*')
        print("all files:", all_files)
        for file in all_files:
            if file.endswith('Train.py'):
                for i in range(max_run):
                    print("run file: {}  {} ".format(file, fileIdx))
                    cmd = 'python ' + file + ' {} {} {} {} {} {} {} {} {}'.format(fileIdx,
                                                                            rotation_range,
                                                                            width_shift_range,
                                                                            height_shift_range,
                                                                            zoom_range,
                                                                            shear_range,
                                                                            brightness_range_start, brightness_range_stop,
                                                                            global_randomness_count
                                                                                  )
                    print(cmd)
                    os.system(cmd)
                    fileIdx += 1
                    if fileIdx > max_run:
                        break
        print("total run {} training scripts!".format(fileIdx))
except Exception as e:
    print("Has some error", e)
#
# # Case 1  BaseExp
# run_case('E:\\Projects\\MyPolyTongue\\TonguePlusData\\EXP_5_CSP_SAE_FrontNeck')

# # Case 1_2 BASE NO SAE
# run_case('C:\\myProjects\\MyPolyTongue\\TonguePlusData\\EXP_1_2_base_noSAE')

# # Case 2  Exp Mish
# run_case('C:\\myProjects\\MyPolyTongue\\TonguePlusData\\EXP_2_Mish')


# # Case 4  EXP_4_SAE_MidNeck
# run_case('C:\\myProjects\\MyPolyTongue\\TonguePlusData\\EXP_4_SAE_MidNeck')

#
# # Case 5  EXP_4_SAE_FrontNeck
# run_case('C:\\myProjects\\MyPolyTongue\\TonguePlusData\\EXP_5_CSP_SAE_FrontNeck')


# # Case 6  EXP_4_SAE_MidNeck
# run_case('C:\\myProjects\\MyPolyTongue\\TonguePlusData\\EXP_6_CSP_SAE_MidNeck')

# # Case 7  best randomness Checking

# rotation_range = 45
# width_shift_range = 0.3
# height_shift_range = 0.3
# zoom_range = 0.2
# shear_range=0.35
# brightness_range=[0.5, 1.5]
rotation_range = np.linspace(0, 90, 10)
width_shift_range = np.linspace(0, 0.3, 10)
height_shift_range = np.linspace(0, 0.3, 10)
zoom_range = np.linspace(0, 0.3, 10)
shear_range = np.linspace(0, 0.35, 10)
brightness_range = np.linspace(0.5, 1.3, 20)
# ranges = [rotation_range,width_shift_range, height_shift_range, zoom_range,shear_range, brightness_range]
# for rangeInterval in ranges:
#
#     print(len(rangeInterval))
for i in range(10):
    print("i:",i)
    rotation_range_in = rotation_range[len(rotation_range)-1-i]
    width_shift_range_in = width_shift_range[len(width_shift_range)-1 - i]
    height_shift_range_in = width_shift_range[len(height_shift_range)-1 - i]
    zoom_range_in = zoom_range[len(zoom_range)-1 - i]
    shear_range_in = shear_range[len(shear_range)-1 - i]
    brightness_range_start_in = brightness_range[i]
    brightness_range_stop_in = brightness_range[len(brightness_range)-1 - i]
    # brightness_range_in =  [brightness_range_start, brightness_range_stop]
    # if i ==4:
    #     print("rotation_range_in:", rotation_range_in)
    #     print("width_shift_range_in:", width_shift_range_in)
    #     print("height_shift_range_in:", height_shift_range_in)
    #     print("zoom_range_in:", zoom_range_in)
    #     print("shear_range_in:", shear_range_in)
    #     print("brightness_range_in:", [brightness_range_start_in, brightness_range_stop_in])
    #     print()
    #
    # run_case('E:\\Projects\\MyPolyTongue\\TonguePlusData\\P_Part2_RandomNess_Exp11_Methodexp8',
    #          global_randomness_count=i,
    #          rotation_range=rotation_range_in,
    #          width_shift_range=width_shift_range_in,
    #          height_shift_range=height_shift_range_in,
    #          zoom_range=zoom_range_in,
    #          shear_range=shear_range_in,
    #          brightness_range_start=brightness_range_start_in,
    #          brightness_range_stop=brightness_range_stop_in
    #
    #          )


    run_case('C:\\myProjects\\MyPolyTongue\\TonguePlusData\\P_Part2_RandomNess_Exp12_Xception',
             global_randomness_count= i,
             rotation_range=rotation_range_in,
             width_shift_range=width_shift_range_in,
             height_shift_range=height_shift_range_in,
             zoom_range=zoom_range_in,
             shear_range=shear_range_in,
             brightness_range_start=brightness_range_start_in,
             brightness_range_stop=brightness_range_stop_in

             )