import os
from glob import glob
try:
    def run_case(file_dir, max_run=5):
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
#
# # # P_Part1_Backbone_Exp7_Xception_Our_DpolarIoU_L2_loss
# run_case('/TonguePlusData/P_Part1_Backbone_Exp7_Xception_Our_DpolarIoU_L2_loss')


#
# # # P_Part1_Backbone_Exp8_Xception_DpolarIoU_BoxScale_L2_loss
# run_case('C:\\myProjects\\MyPolyTongue\\TonguePlusData/P_Part1_Backbone_Exp8_Xception_DpolarIoU_BoxScale_L2_loss')
#


#
# # # P_Part1_Backbone_Exp9_Xception_DpolarIoU_BoxScale_L2_loss
# run_case('C:\\myProjects\\MyPolyTongue\\TonguePlusData/P_Part1_BK_Exp9_Xception_DpolarIoU_BoxScale_L2_Mask_loss')
#


#
# # # Exp10_Xception_DpolarIoU_BoxScale_L2_MaskIoU_loss
# run_case('/TonguePlusData/Exp10_Xception_DpolarIoU_BoxScale_L2_MaskIoU_loss')


#
# # # # P_PartF_BK_FinalFull_CioUDMaskDicePolarDiouConfidence
# run_case('C:\\myProjects\\MyPolyTongue\\TonguePlusData/P_PartF_BK_FinalFull_CioUDMaskDicePolarDiouConfidence')


# # # P_PartF_BK_FinalFull_CioUDMaskDicePolarDiouConfidence
# run_case('C:\\myProjects\\MyPolyTongue\\TonguePlusData/P_OurFinal_CioUDMaskDicePolarDioU')



# # # Previous the all detection features in model is wrong, redo the experiment  P_PartF_BK_FinalFull_CioUDMaskDicePolarDioUFixedV2
# run_case('C:\\myProjects\\MyPolyTongue\\TonguePlusData/P_PartF_BK_FinalFull_CioUDMaskDicePolarDioUFixedV2')



# # #
# # Case 1  BaseExp
# run_case('/TonguePlusData/PaperExp_Part1_Backbone_Exp5_MobileNetV3_Large')
#
# Case 1  BaseExp
# run_case('/TonguePlusData/P_Part1_BK_Exp9_Xception_DpolarIoU_BoxScale_L2_Mask_loss')
#
# # # Case 1_2 BASE NO SAE
# # run_case('C:\\myProjects\\MyPolyTongue\\TonguePlusData\\EXP_1_2_base_noSAE')
#
# # # Case 2  Exp Mish
# # run_case('C:\\myProjects\\MyPolyTongue\\TonguePlusData\\EXP_2_Mish')
#
#
# # # Case 4  EXP_4_SAE_MidNeck
# # run_case('C:\\myProjects\\MyPolyTongue\\TonguePlusData\\EXP_4_SAE_MidNeck')
#
# #
# # # Case 5  EXP_4_SAE_FrontNeck
# # run_case('C:\\myProjects\\MyPolyTongue\\TonguePlusData\\EXP_5_CSP_SAE_FrontNeck')
#
#
# # # Case 6  EXP_4_SAE_MidNeck
# # run_case('C:\\myProjects\\MyPolyTongue\\TonguePlusData\\EXP_6_CSP_SAE_MidNeck')
#
# # # Case 7  best randomness Checking
#
# # rotation_range = sys.argv[2]
# # width_shift_range = sys.argv[3]
# # height_shift_range = sys.argv[4]
# # zoom_range = sys.argv[5]
# # shear_range=sys.argv[6]
# # horizontal_flip=True
# # brightness_range=sys.argv[7]
# # run_case('C:\\myProjects\\MyPolyTongue\\TonguePlusData\\EXP_6_CSP_SAE_MidNeck')
#
# #     # # Case 8 MISH NP INTERP
# # run_case('C:\\myProjects\\MyPolyTongue\\TonguePlusData\\EXP_8_Mish_WithMyDataNpInterpDistRegOnly')
# #
#  # # Case11 EXP_11_E:\\Projects\\MyPolyTongue\\TonguePlusData\\EXP_11_Mish_WithMyDataNpInterpDistRegL2CEOnly_PolarDIoULoss_bestAug
run_case('E:\\Projects\\MyPolyTongue\\TonguePlusData/PaperF2_FinalRedo_EfficientNetB4_FixedV2_bestAug')

# #     # # Case11 MISH NP INTERP
# # run_case('C:\\myProjects\\MyPolyTongue\\TonguePlusData\\EXP_11_Mish_WithMyDataNpInterpDistRegL2CEOnly_PolarDIoULoss_bestAug')