import os
from glob import glob
import argparse
import json
import time

import cv2
import numpy as np
import pandas as pd

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

try:
    def infer_case(Inference_scripts_root, Saved_model_file_root, output_folder):

        # create inference summary: ---->
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        inferenc_summary_txt= output_folder +"/details.txt"
        inferenc_summary_paper_need_txt = output_folder +"/paperNeed.txt"
        if not os.path.exists(inferenc_summary_txt):
            with open(inferenc_summary_txt, 'w'): pass
        else:
            os.remove(inferenc_summary_txt)
            with open(inferenc_summary_txt, 'w'):
                pass
        if not os.path.exists(inferenc_summary_paper_need_txt):
            with open(inferenc_summary_paper_need_txt, 'w'): pass
        else:
            os.remove(inferenc_summary_paper_need_txt)
            with open(inferenc_summary_paper_need_txt, 'w'):
                pass


        all_scripts_files = glob(Inference_scripts_root + '/*.py')
        print("all_scripts_files:",all_scripts_files)
        class_file = glob(Inference_scripts_root + '/*classesTongue.txt')[0]
        print("class_file:", class_file)
        # start to parsing the inference-->
        fileIdx =1
        # model_saved_name_idx = os.path.basename(Saved_model_file_root)


        all_files = glob(Saved_model_file_root + '/*')
        all_dirs = [path for path in all_files if os.path.isdir(path)]
        print("allfiles:", all_files)
        print("all_dirs:", all_dirs)
        print()
        model_folder_names_list =[]
        best_h5_list = []
        # found saved_folder_name and h5 path ----------------->
        for dir in all_dirs:
            # model_saved_name_idx = os.path.basename(dir)
            # print("model_saved_name:", model_saved_name_idx )
            # # found the model saved path and the best h5 file
            # if dir.endswith(model_saved_name_idx+"{}".format(fileIdx)):
            print("model saved dir:", dir)
            output_folder_name =  os.path.basename(dir)
            print("output_foldername:", output_folder_name)
            model_folder_names_list.append(output_folder_name)
            fileIdx += 1

            all_h5_files =  glob(dir + '/*.h5')
            print()
            print("all_h5_files:", all_h5_files)
            if len(all_h5_files) > 1:
                all_h5_files.sort(key=os.path.getctime)
            best_h5 = all_h5_files[-1]
            best_h5_list.append(best_h5)
            print("best h5:", best_h5)

                    # for root, dirs, _ in os.walk(file_dir):
        print("found total {} cases need to be infrerence:".format(len(best_h5_list)))
        # Running inference scripts and passing the vars: model_folder_name,    best_h5, output_folder --------------->
        for model_folder_name, best_h5_path in zip(model_folder_names_list, best_h5_list):
            print("start to run inference --------------->")
            print("model_folder_name", model_folder_name)
            id = model_folder_name[-1]
            print("id:", id)
            print("best_h5_path", best_h5_path)

            for script in all_scripts_files:
               if script.endswith("inference.py"):
                   print("id:", id)
                   print("script:", script)
                   # model_folder_name,    best_h5, output_folder
                   cmd = 'python ' + script + ' {} {} {} {}'.format(model_folder_name, best_h5_path,output_folder, inferenc_summary_txt)
                   print("cmd:", cmd)

                   os.system(cmd)

        # return avg FPS value  ---------------->
        with open(inferenc_summary_txt) as f:
            lines = f.readlines()
        print(lines)
        for i in range (0, len(lines)):

            lines[i] = lines[i].split(',')
        print(lines)
        Total_avg_fps_list = []
        Total_std_fps_list = []
        for each_case in lines:
            avg_fps_element =  each_case[-2]
            avg_fps_value =  float(avg_fps_element.split()[-1])
            std_fps_element = each_case[-1]
            std_fps_value = float(std_fps_element.split()[-1])
            # print(type(fps_value))
            Total_avg_fps_list.append(avg_fps_value)
            Total_std_fps_list.append(std_fps_value)
        Total_avg_fps = sum(Total_avg_fps_list)/len(Total_avg_fps_list)
        Total_std_fps = sum(Total_std_fps_list) / len(Total_std_fps_list)
        print("Total_avg_fps:", Total_avg_fps)
        print("Total_std_fps:", Total_std_fps)
        print("total_cases:", len(lines))
        with open(inferenc_summary_txt, 'a') as f:
            f.write("Total_avg_fps {}\n".format(Total_avg_fps))
            f.write("Total_std_fps {}\n".format(Total_std_fps))
        # search saved txt predictions and compared with text label with coco evaluation  ---------------->
        pred_txt_list = []
        label_txt_list = []
        for root, dirs, files in os.walk(output_folder):
            for i in files:
                if "predictResult" in i:
                    print("root", root)
                    print("dirs", dirs)
                    print("files", i)
                    pred_txt_list.append(root+'/'+i)
                if "labelResult" in i:
                    label_txt_list.append(root + '/' + i)
        print("pred_txt_list:", pred_txt_list)
        print("label_txt_list:", label_txt_list)

        # start to evaluate:
        eval_types_list =["bbox", "segm"]


        for type in eval_types_list:
            AP_5to95_list = []
            AP_5_list = []
            AP_75_list = []
            with open(inferenc_summary_txt, 'a') as f:
                f.write("\n")
            for pred, gt in zip(pred_txt_list, label_txt_list):
                print("pred:", pred)
                # this neeed to be changed for the lab computer gt:
                # gt = os.path.dirname(os.path.realpath(__file__)) + "/myTongueTestLab.txt"
                print("cwd", os.path.dirname(os.path.realpath(__file__)))

                yolo_to_coco(pred, gt, class_file)
                mean_s=coco_eval(type)
                print("tyep:", type)
                print("mean_s:", mean_s)

                # write to our txt
                AP_5to95=mean_s[0]
                AP_5 = mean_s[1]
                AP_75 = mean_s[2]
                print("AP_5to95", AP_5to95)
                print("AP_5", AP_5)
                print("AP_75", AP_75)
                AP_5to95_list.append(AP_5to95)
                AP_5_list.append(AP_5)
                AP_75_list.append(AP_75)

                # write all the evaluations into  txt
                with open(inferenc_summary_txt, 'a') as f:
                    f.write("infer_path {}, type {}, AP_5to95|AP_5|AP_75 {}|{}|{}\n".format(pred, type, AP_5to95, AP_5, AP_75))

            # write the summary for the paper needed --------------->
            with open(inferenc_summary_paper_need_txt, 'a') as f:
                m_AP_5to95 = np.array(AP_5to95_list).mean()
                std_AP_5to95 = np.array(AP_5to95_list).std()

                m_AP_5 = np.array(AP_5_list).mean()
                std_AP_5 = np.array(AP_5_list).std()

                m_AP_75 = np.array(AP_75_list).mean()
                std_AP_75 = np.array(AP_75_list).std()
                content= 'type {}, AP_5to95_m(AP_5to95_std)|AP_5_m(AP_5_std)|AP_75_m(AP_75_std) ' \
                         '{:.3f} ({:.3f})|{:.3f} ({:.3f})|{:.3f} ({:.3f})\n'.format(type, m_AP_5to95, std_AP_5to95, m_AP_5, std_AP_5, m_AP_75, std_AP_75)
                f.write(content)

        # print("total found and evaluated {} training models!".format(fileIdx-1))


    def yolo_to_coco(pred_path: str, gt_path: str, classes_path: str) -> None:
        """
        Converts predictions and labels in yolo format into coco compatible json files that are then evaluated.
        :param pred_path: Path to the text file with predictions in yolo format.
        :param gt_path: Path to the text file with labels in yolo format.
        :param classes_path: Path to the text file with classes contained in labels/predictions. One per line.
        """
        print('beginning conversion from yolo format to coco json files ...')

        coco_pred = []
        current_file_dir_path = os.path.dirname(os.path.realpath(__file__))
        clss = []
        with open(classes_path, 'r') as f:
            lines = f.readlines()
            for i in range(len(lines)):
                clss.append({'id': i, 'name': lines[i].rstrip()})
        coco_gt = {'annotations': [],
                   'images': [],
                   # TODO:categories could be loaded from classes.txt
                   'categories': clss}
        img_name_to_id = {}
        img_id = 0
        with open(pred_path, 'r') as f1:
            pred_lines = f1.readlines()
        with open(gt_path, 'r') as f2:
            gt_lines = f2.readlines()

        id = 0
        for gt_line in gt_lines:
            gt_line = gt_line.rstrip()
            gt_img_path = gt_line.split(' ')[0]
            gt_data = gt_line.split(' ')
            gt_vectors = gt_data[1:]

            for gt_vector in gt_vectors:
                gt_vector = str_list_to_float_list(gt_vector)
                annotation = {'image_id': img_id,
                              'id': id,
                              'iscrowd': 0,
                              # area does not matter - in evaluation we are looking for area: all
                              'area': 1,
                              'category_id': int(gt_vector[4]),
                              'bbox': [gt_vector[0], gt_vector[1], gt_vector[2] - gt_vector[0],
                                       gt_vector[3] - gt_vector[1]]}
                if len(gt_vector) > 5:
                    annotation['segmentation'] = [gt_vector[5:]]
                coco_gt['annotations'].append(annotation)
                id += 1
            img_name_to_id[gt_img_path] = img_id
            img = cv2.imread(gt_img_path)
            coco_gt['images'].append(
                {'file_name': gt_img_path, 'id': img_id, 'height': img.shape[0], 'width': img.shape[1]})
            img_id += 1

        for pred_line in pred_lines:
            pred_line = pred_line.rstrip()
            pred_img_name = pred_line.split(' ')[0]
            pred_data = pred_line.split(' ')
            pred_vectors = pred_data[1:]

            for pred_vector in pred_vectors:
                pred_vector = str_list_to_float_list(pred_vector)
                annotation = {'image_id': img_name_to_id[pred_img_name],
                              'category_id': int(pred_vector[5]),
                              'bbox': [pred_vector[0], pred_vector[1], pred_vector[2] - pred_vector[0],
                                       pred_vector[3] - pred_vector[1]],
                              'segmentation': [pred_vector[6:]],
                              'score': pred_vector[4]
                              }
                if len(pred_vector) > 6:
                    annotation['segmentation'] = [pred_vector[6:]]
                coco_pred.append(annotation)
        with open(current_file_dir_path + '/tmp_coco_pred.json', 'w') as ccp:
            json.dump(coco_pred, ccp)
        with open(current_file_dir_path + '/tmp_coco_gt.json', 'w') as ccg:
            json.dump(coco_gt, ccg)
        print('conversion done successfully!')


    def coco_eval(type):
        current_file_dir_path = os.path.dirname(os.path.realpath(__file__))
        cocoGt = COCO(current_file_dir_path+ '/tmp_coco_gt.json')
        cocoDt = cocoGt.loadRes(current_file_dir_path+ '/tmp_coco_pred.json')
        cocoEval = COCOeval(cocoGt, cocoDt, type)
        cocoEval.evaluate()
        cocoEval.accumulate()
        mean_s = cocoEval.summarize()

        return mean_s

    def str_list_to_float_list(string):
        return list(map(float, string.split(',')))

except Exception as e:
    print("Has some error", e)


if __name__ == '__main__':


    # remember to change the angles step to the fixed model before run it
    # Saved_model_file_root = "F:\\AngleStepCheck\\SavedModels\\PaperF3_FinalRedo_Xception_FixedV2_bestAug_AngleStepCheck\\AS0"
    # Inference_scripts_root = "C:\\MyProjects\\projectFiles\\TonguePlusData\\PaperF3_FinalRedo_Xception_FixedV2_bestAug_AngleStepCheck"
    # #
    # output_folder = "F:\\AngleStepCheck\\output\\AS0"  # the name path can not be too long
    # infer_case(Inference_scripts_root=Inference_scripts_root,
    #            Saved_model_file_root=Saved_model_file_root,
    #            output_folder=output_folder)


    # # AS =1.8888888888888888
    # Saved_model_file_root = "F:\\AngleStepCheck\\SavedModels\\PaperF3_FinalRedo_Xception_FixedV2_bestAug_AngleStepCheck\\AS1"
    # Inference_scripts_root = "C:\\MyProjects\\projectFiles\\TonguePlusData\\PaperF3_FinalRedo_Xception_FixedV2_bestAug_AngleStepCheck"
    # #
    # output_folder = "F:\\AngleStepCheck\\output\\AS1"  # the name path can not be too long
    # infer_case(Inference_scripts_root=Inference_scripts_root,
    #            Saved_model_file_root=Saved_model_file_root,
    #            output_folder=output_folder)
    #
    # # AS =3.2777777777777777
    # Saved_model_file_root = "F:\\AngleStepCheck\\SavedModels\\PaperF3_FinalRedo_Xception_FixedV2_bestAug_AngleStepCheck\\AS2"
    # Inference_scripts_root = "C:\\MyProjects\\projectFiles\\TonguePlusData\\PaperF3_FinalRedo_Xception_FixedV2_bestAug_AngleStepCheck"
    # #
    # output_folder = "F:\\AngleStepCheck\\output\\AS2"  # the name path can not be too long
    # infer_case(Inference_scripts_root=Inference_scripts_root,
    #            Saved_model_file_root=Saved_model_file_root,
    #            output_folder=output_folder)

    # # AS4.666666666666666
    # Saved_model_file_root = "F:\\AngleStepCheck\\SavedModels\\PaperF3_FinalRedo_Xception_FixedV2_bestAug_AngleStepCheck\\AS3"
    # Inference_scripts_root = "C:\\MyProjects\\projectFiles\\TonguePlusData\\PaperF3_FinalRedo_Xception_FixedV2_bestAug_AngleStepCheck"
    # #
    # output_folder = "F:\\AngleStepCheck\\output\\AS3"  # the name path can not be too long
    # infer_case(Inference_scripts_root=Inference_scripts_root,
    #            Saved_model_file_root=Saved_model_file_root,
    #            output_folder=output_folder)

    # # AS6.055555555555555
    # Saved_model_file_root = "F:\\AngleStepCheck\\SavedModels\\PaperF3_FinalRedo_Xception_FixedV2_bestAug_AngleStepCheck\\AS4"
    # Inference_scripts_root = "C:\\MyProjects\\projectFiles\\TonguePlusData\\PaperF3_FinalRedo_Xception_FixedV2_bestAug_AngleStepCheck"
    # #
    # output_folder = "F:\\AngleStepCheck\\output\\AS4"  # the name path can not be too long
    # infer_case(Inference_scripts_root=Inference_scripts_root,
    #            Saved_model_file_root=Saved_model_file_root,
    #            output_folder=output_folder)

    # # AS7.444444444444445
    # Saved_model_file_root = "F:\\AngleStepCheck\\SavedModels\\PaperF3_FinalRedo_Xception_FixedV2_bestAug_AngleStepCheck\\AS5"
    # Inference_scripts_root = "C:\\MyProjects\\projectFiles\\TonguePlusData\\PaperF3_FinalRedo_Xception_FixedV2_bestAug_AngleStepCheck"
    # #
    # output_folder = "F:\\AngleStepCheck\\output\\AS5"  # the name path can not be too long
    # infer_case(Inference_scripts_root=Inference_scripts_root,
    #            Saved_model_file_root=Saved_model_file_root,
    #            output_folder=output_folder)

    # # AS8.833333333333332
    # Saved_model_file_root = "F:\\AngleStepCheck\\SavedModels\\PaperF3_FinalRedo_Xception_FixedV2_bestAug_AngleStepCheck\\AS6"
    # Inference_scripts_root = "C:\\MyProjects\\projectFiles\\TonguePlusData\\PaperF3_FinalRedo_Xception_FixedV2_bestAug_AngleStepCheck"
    # #
    # output_folder = "F:\\AngleStepCheck\\output\\AS6"  # the name path can not be too long
    # infer_case(Inference_scripts_root=Inference_scripts_root,
    #            Saved_model_file_root=Saved_model_file_root,
    #            output_folder=output_folder)

    # # AS10.222222222222221
    # Saved_model_file_root = "F:\\AngleStepCheck\\SavedModels\\PaperF3_FinalRedo_Xception_FixedV2_bestAug_AngleStepCheck\\AS7"
    # Inference_scripts_root = "C:\\MyProjects\\projectFiles\\TonguePlusData\\PaperF3_FinalRedo_Xception_FixedV2_bestAug_AngleStepCheck"
    # #
    # output_folder = "F:\\AngleStepCheck\\output\\AS7"  # the name path can not be too long
    # infer_case(Inference_scripts_root=Inference_scripts_root,
    #            Saved_model_file_root=Saved_model_file_root,
    #            output_folder=output_folder)

    # # ASAS11.61111111111111
    # Saved_model_file_root = "F:\\AngleStepCheck\\SavedModels\\PaperF3_FinalRedo_Xception_FixedV2_bestAug_AngleStepCheck\\AS8"
    # Inference_scripts_root = "C:\\MyProjects\\projectFiles\\TonguePlusData\\PaperF3_FinalRedo_Xception_FixedV2_bestAug_AngleStepCheck"
    # #
    # output_folder = "F:\\AngleStepCheck\\output\\AS8"  # the name path can not be too long
    # infer_case(Inference_scripts_root=Inference_scripts_root,
    #            Saved_model_file_root=Saved_model_file_root,
    #            output_folder=output_folder)

    # AS13.0
    Saved_model_file_root = "F:\\AngleStepCheck\\SavedModels\\PaperF3_FinalRedo_Xception_FixedV2_bestAug_AngleStepCheck\\AS9"
    Inference_scripts_root = "C:\\MyProjects\\projectFiles\\TonguePlusData\\PaperF3_FinalRedo_Xception_FixedV2_bestAug_AngleStepCheck"
    #
    output_folder = "F:\\AngleStepCheck\\output\\AS9"  # the name path can not be too long
    infer_case(Inference_scripts_root=Inference_scripts_root,
               Saved_model_file_root=Saved_model_file_root,
               output_folder=output_folder)