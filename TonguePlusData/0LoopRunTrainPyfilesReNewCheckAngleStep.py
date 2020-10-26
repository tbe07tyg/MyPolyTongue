import os
from glob import glob
import numpy as np
import shutil
try:
    def run_case(file_dir,global_randomness_count, angle_step, max_run=5):
        """
        # rotation_range = 45
        # width_shift_range = 0.3
        # height_shift_range = 0.3
        # zoom_range = 0.2
        # shear_range=0.35
        # horizontal_flip=True
        # brightness_range=[0.5, 1.3]

        :param file_dir:
        :param rotation_rangemax_run:
        :return:
        """
        if global_randomness_count == 8:
            fileIdx = 3
        else:
            fileIdx = 1
        # for root, dirs, files in os.walk(file_dir):
        #     print(files, "to be run")
        #     for i in files:
        all_files = glob(file_dir + '/*')
        print("all files:", all_files)
        for file in all_files:
            if file.endswith('Train.py'):
                for i in range(max_run):
                    print("run file: {}  {} ".format(file, fileIdx))
                    cmd = 'python ' + file + ' {} {}'.format(fileIdx, angle_step)
                    print(cmd)
                    os.system(cmd)
                    fileIdx += 1
                    if fileIdx > max_run:
                        break
        print("total run {} training scripts!".format(fileIdx-1))
except Exception as e:
    print("Has some error", e)

# for the paper

# Paper real experiments start:  -------------------------------->

AngleSteps = np.linspace(0.5, 13, 10)
print("angle_steps:", AngleSteps)

# Create_ANGLE_STEP_training root folder:
# folder to where to save all the cases
current_file_dir_path = os.path.dirname(os.path.realpath(__file__))
AngleSteps_root =  os.path.join(current_file_dir_path, "0000NewAngleStepCheckForMultiClsPrimitives")
raw_design_folder = "E:\\Projects\\MyPolyTongue\\TonguePlusData/TrainSPBaseDS35_BA_MultiClasses_ImageNet_SearchAngleStep"




if not os.path.exists(AngleSteps_root):
    os.makedirs(AngleSteps_root)

src_files = glob(raw_design_folder+ "/*")
print("src_files:", print(src_files))
for i, angle_step in enumerate(AngleSteps):
    print("i:", i)
    print("angle_step", angle_step)
    sub_angle_step_case =  "i{}_AS{}".format(i, angle_step)
    each_case_root = os.path.join(AngleSteps_root, sub_angle_step_case)
    if not os.path.exists(each_case_root):
        os.makedirs(each_case_root)

    # # copy file into the specified
    # for file_name in src_files:
    #     print("copy file:", file_name)
    #     # full_file_name = os.path.join(each_case_root, file_name)
    #     print("copy to:", each_case_root)
    #     # print("full_file_name:", file_name)
    #     if os.path.isfile(file_name):
    #         print("full name:", file_name)
    #         shutil.copy(file_name, each_case_root)
    # train:
#     if i < 8:
#         continue
    # # # P_PartF_BK_FinalFull_CioUDMaskDicePolarDiouConfidence
    # run_case('C:\\MyProjects\\projectFiles\\TonguePlusData/PaperF3_FinalRedo_Xception_FixedV2_bestAug_AngleStepCheck',i , angle_step=angle_step)
    run_case('E:\\Projects\\MyPolyTongue\\TonguePlusData/TrainSPBaseDS35_BA_MultiClasses_ImageNet_SearchAngleStep', global_randomness_count=i, angle_step=angle_step)
