import os
from glob import glob
try:
    def run_case(file_dir, max_run=5, fileIdx=1):
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

        # for root, dirs, files in os.walk(file_dir):
        #     print(files, "to be run")
        #     for i in files:
        all_files = glob(file_dir + '/*')
        print("all files:", all_files)
        for file in all_files:
            if file.endswith('Train.py'):
                for i in range(max_run):
                    print("run file: {}  {} ".format(file, fileIdx))
                    cmd = 'python ' + file + ' {}'.format(fileIdx)
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

# # PaperExp_Part1_Backbone_Exp1_rawModel
# run_case('C:\\myProjects\\MyPolyTongue\\TonguePlusData\\PaperExp_Part1_Backbone_Exp1_rawModel')
#
# # PaperExp_Part1_Backbone_Exp2_Xception
# run_case('C:\\myProjects\\MyPolyTongue\\TonguePlusData\\PaperExp_Part1_Backbone_Exp2_Xception')
#
# # # PaperExp_Part1_Backbone_Exp3_ResNet101V2
# run_case('C:\\myProjects\\MyPolyTongue\\TonguePlusData\\PaperExp_Part1_Backbone_Exp3_ResNet101V2')

# # # # PaperExp_Part1_Backbone_Exp4_InceptionV3
# run_case('C:\\myProjects\\MyPolyTongue\\TonguePlusData\\PaperExp_Part1_Backbone_Exp4_InceptionV3')
#
# #
# # # PaperExp_Part1_Backbone_Exp5_MobileNetV3_Large
# run_case('C:\\myProjects\\MyPolyTongue\\TonguePlusData\\PaperExp_Part1_Backbone_Exp5_MobileNetV3_Large')

# # # PaperExp_Part1_Backbone_Exp6_EfficientNetB4
# run_case('C:\\myProjects\\MyPolyTongue\\TonguePlusData\\PaperExp_Part1_Backbone_Exp6_EfficientNetB4')
# #

# # My method cases:
# case_list = ['C:\\myProjects\\MyPolyTongue\\TonguePlusData\\P_PartF_BK_FinalFull_CioUDMaskDicePolarDioU',
#              'C:\\myProjects\\MyPolyTongue\\TonguePlusData\\P_PartF_BK_FinalNoMask_CioUDPolarDioU',
#              'C:\\myProjects\\MyPolyTongue\\TonguePlusData\\P_PartF_BK_FinalNoMask_CioUDPolarDioUSAEBeforeAdd'
#              ]
# count =0
# for each_case in case_list:
#     count=+1
#     if count==1:
#         run_case(each_case, max_run=5, fileIdx=3)
#     else:
#         run_case(each_case)
#
# # My method cases:
# case_list = ['C:\\MyProjects\\projectFiles\\TonguePlusData\\PaperF2_FinalRedo_Xception_FixedV2_bestAug',
#              'C:\\MyProjects\\projectFiles\\TonguePlusData\\PaperF2_FinalRedo_ResNet101V2_FixedV2_bestAug'
#              ]
#
# for each_case in case_list:
#     run_case(each_case)

# # My method cases:
# case_list = ['C:\\MyProjects\\projectFiles\\TonguePlusData\\PaperF2_FinaRedo_YoloFixedV2_Mish',
#              'C:\\MyProjects\\projectFiles\\TonguePlusData\\PaperF4_FinalRedo_Xception_FixedV2_bestAug_Primitives']

cases_root = "C:\\MyProjects\\projectFiles\\TonguePlusData\\PaperF4_FinalRedo_Xception_FixedV2_bestAug_P_DScheck/*"

case_list = sorted(glob(cases_root))
print("case_list:", case_list)
print(len(case_list))

# case_list = ['C:\\MyProjects\\projectFiles\\TonguePlusData\\PaperF4_FinalRedo_Xception_FixedV2_bestAug_Primitives_newDS']
for i, each_case in enumerate(case_list):
    print("i:", i)
    print(each_case)
    if i < 10:
        continue
    elif i ==10:
        run_case(each_case, fileIdx=2)
    else:
        run_case(each_case, fileIdx=1)
 # Case11 EXP_11_E:\\Projects\\MyPolyTongue\\TonguePlusData\\EXP_11_Mish_WithMyDataNpInterpDistRegL2CEOnly_PolarDIoULoss_bestAug
# run_case('E:\\Projects\\MyPolyTongue\\TonguePlusData/PaperF4_FinalRedo_Xception_FixedV2_bestAug_Primitives_newDS')

# #     # # Case11 MISH NP INTERP
# # run_case('C:\\myProjects\\MyPolyTongue\\TonguePlusData\\EXP_11_Mish_WithMyDataNpInterpDistRegL2CEOnly_PolarDIoULoss_bestAug')