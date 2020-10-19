import os
import matplotlib
# matplotlib.use('agg')

from glob import glob
import argparse
import json
import time
import re
import cv2
import numpy as np
import pandas as pd
from pandas import DataFrame
import pandas
pandas.set_option('display.max_rows', None)
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import FuncFormatter, MaxNLocator
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['savefig.transparent'] = True
# plt.yticks(fontsize=11)
# #
# plt.xlabel('Aug', fontdict={'color': 'black',
#                              'family': 'Times New Roman',
#                              'weight': 'normal',
#                              'size': 15})
# plt.ylabel('mAP', fontdict={'color': 'black',
#                           'family': 'Times New Roman',
#                           'weight': 'normal',
#                           'size': 15})

def evaluate_sub_folder(sub_root_dir, infer_script, result_txt, output_folder):
    # found all folders under sub root
    all_files = glob(sub_root_dir + '/*')
    all_dirs = [path for path in all_files if os.path.isdir(path)]

    model_folder_names_list =[]
    best_h5_list = []
    for dir in all_dirs:
        print("model saved dir:", dir)
        output_folder_name = os.path.basename(dir)
        print("output_foldername:", output_folder_name)
        model_folder_names_list.append(output_folder_name)

        # found best h5
        all_h5_files = glob(dir + '/*.h5')
        print("all_h5_files:", all_h5_files)
        if len(all_h5_files) > 1:
            all_h5_files.sort(key=os.path.getctime)
        best_h5 = all_h5_files[-1]
        best_h5_list.append(best_h5)
        print("best h5:", best_h5)
    print("found total {} cases need to be inference in one subfolder:".format(len(best_h5_list)))

    for model_folder_name, best_h5_path in zip(model_folder_names_list, best_h5_list):
        print("start to run inference --------------->")
        print("model_folder_name", model_folder_name)
        print("best_h5_path", best_h5_path)

        if infer_script.endswith("inference.py"):
           print("infer script:", infer_script)
           # model_folder_name,    best_h5, output_folder
           cmd = 'python ' + infer_script + ' {} {} {} {}'.format(model_folder_name, best_h5_path,output_folder, result_txt)
           print("cmd:", cmd)

           os.system(cmd)





try:
    def infer_case(Saved_model_file_root, output_folder, inference_folder, class_path):
        inferenc_summary_txt= output_folder +"/details.txt"
        if not os.path.exists(inferenc_summary_txt):
            with open(inferenc_summary_txt, 'w'): pass
        # create inference summary: ---->
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        all_Top_cases = glob(Saved_model_file_root + '/*')
        print("found {} top cases".format(len(all_Top_cases)))
        print("all_Top_cases:", all_Top_cases)
        count =0
        for each_top_case in all_Top_cases:
            # count += 1
            # if count ==1:
            #     continue

            case_name = os.path.basename(each_top_case)


            infer_script_path_list= []
            print("current check case:", each_top_case)
            all_contents = glob(each_top_case + '/*')
            print("all contents in current top case:", all_contents)

        #     all_sub_roots = [path for path in all_contents if os.path.isdir(path)]
        #     print("all_sub_root_folders in current top case:", all_sub_roots)
        #
        #     infer_scripts =  [path for path in all_contents if path.endswith("inference.py")]
        #     print("infer_script in current top case:",infer_scripts )
        #     infer_script_name =  os.path.basename(infer_scripts[0])
        #     print("infer_script_name:", infer_script_name)
        #     # search in the inference root: try to find the inference script in current working directory
        #     for root, dirs, files in os.walk(inference_folder):
        #         for file in files:
        #             if infer_script_name in file:
        #                 infer_script_path = root +'/'+ file
        #                 print("found the corresponding inference script file in current working dir:", infer_script_path)
        #                 infer_script_path_list.append(infer_script_path)
        #     sorted_all_sub_roots =  []
        #     eval_means = []
        #     eval_stds = []
        #     for each_root in all_sub_roots:
        #         root_name =  os.path.basename(each_root)
        #         print("root name:", root_name)
        #         re_digits = re.findall(r'\d+', root_name)  # reg search digits in string
        #         print("re_digits:", re_digits)
        #         root_index =  int(re_digits[-1])
        #         group_root_withIndex =  (root_index, each_root)
        #         sorted_all_sub_roots.append(group_root_withIndex)
        #     print("paired root index:",sorted_all_sub_roots )
        #     sorted_all_sub_roots = sorted(sorted_all_sub_roots)
        #     print("sorted paired root index:", sorted_all_sub_roots)
        #     for index, root in sorted_all_sub_roots:
        #         print("index:", index)
        #         print("root:", root)
        #         root_name = os.path.basename(root)
        #         # start to evaluate for each
        #         evaluate_sub_folder(root, infer_script_path_list[0], inferenc_summary_txt, output_folder + "/"+ case_name + "/" + root_name)
        #
        # finish generating all txt data ...................for coco evaluate..---->
        # select txt according to top cases name:
        infered_Top_cases = glob(output_folder + '/*')
        infered_Top_cases = [path for path in infered_Top_cases if os.path.isdir(path)]
        print("infered_top cases:",infered_Top_cases)
        #
        # colors=["g", "r"]
        markers= ["o", "x"]
        legends_text = ["worst", "best"]
        #
        case_count =0
        axes =None
        for inferd_case_dir in infered_Top_cases:  # case level
            case_name = os.path.basename(inferd_case_dir)
            print("case name:", case_name)
        #     infered_sub_aug_cases = [path for path in glob(inferd_case_dir + '/*') if os.path.isdir(path)]
        #     print("infered_sub_aug_cases:", infered_sub_aug_cases)
        #
        #
        #
        #
        #     # for sort order only
        #     sorted_all_sub_infered_roots = []
        #
        #     for each_root in infered_sub_aug_cases:
        #         root_name = os.path.basename(each_root)
        #         print("root name:", root_name)
        #         re_digits = re.findall(r'\d+', root_name)  # reg search digits in string
        #         print("re_digits:", re_digits)
        #         root_index = int(re_digits[-1])
        #         group_root_withIndex = (root_index, each_root)
        #         sorted_all_sub_infered_roots.append(group_root_withIndex)
        #
        #     sorted_all_sub_infered_roots = sorted(sorted_all_sub_infered_roots)
        #     print("sorted_all_sub_infered_roots:", sorted_all_sub_infered_roots)
        #
        #
        #     # start real estimate
        #     summary_for_plot = []
        #
        #     for index, infered_sub_root in sorted_all_sub_infered_roots:
        #         print("index:", index)
        #         print("infered_sub_root:", infered_sub_root)
        #         infered_sub_root_name = os.path.basename(infered_sub_root)
        #         # start to find the txt result from sub inferred root
        #         pred_txt_list = []
        #         label_txt_list = []
        #         for root, dirs, files in os.walk(infered_sub_root):
        #             for i in files:
        #                 if "predictResult" in i:
        #                     print("root", root)
        #                     print("dirs", dirs)
        #                     print("files", i)
        #                     pred_txt_list.append(root + '/' + i)
        #                 if "labelResult" in i:
        #                     label_txt_list.append(root + '/' + i)
        #         print("pred_txt_list:", pred_txt_list)
        #         print("label_txt_list:", label_txt_list)
        #
        #         # start coco evlauate for one aug case
        #         # start to evaluate:
        #         eval_types_list =["bbox", "segm"]
        #
        #         for type in eval_types_list:
        #             AP_5to95_list = []
        #             AP_5_list = []
        #             AP_75_list = []
        #             with open(inferenc_summary_txt, 'a') as f:
        #                 f.write("\n")
        #             for pred, gt in zip(pred_txt_list, label_txt_list):
        #                 print("pred:", pred)
        #                 # this neeed to be changed for the lab computer gt:
        #                 # gt = os.path.dirname(os.path.realpath(__file__)) + "/myTongueTestLab.txt"
        #                 # gt = os.path.dirname(os.path.realpath(__file__)) + "/myTongueTest.txt"  # for labe top
        #                 print("cwd", os.path.dirname(os.path.realpath(__file__)))
        #
        #                 yolo_to_coco(pred, gt, class_path)
        #                 mean_s=coco_eval(type)
        #                 print("tyep:", type)
        #                 print("mean_s:", mean_s)
        #
        #                 # write to our txt
        #                 AP_5to95=mean_s[0]
        #                 AP_5 = mean_s[1]
        #                 AP_75 = mean_s[2]
        #                 print("AP_5to95", AP_5to95)
        #                 print("AP_5", AP_5)
        #                 print("AP_75", AP_75)
        #                 # AP_5to95_list.append(AP_5to95)
        #                 # AP_5_list.append(AP_5)
        #                 # AP_75_list.append(AP_75)
        #
        #             # # with open(inferenc_summary_paper_need_txt, 'a') as f:
        #             # m_AP_5to95 = np.array(AP_5to95_list).mean()
        #             # std_AP_5to95 = np.array(AP_5to95_list).std()
        #             #
        #             # m_AP_5 = np.array(AP_5_list).mean()
        #             # std_AP_5 = np.array(AP_5_list).std()
        #             #
        #             # m_AP_75 = np.array(AP_75_list).mean()
        #             # std_AP_75 = np.array(AP_75_list).std()
        #                 summary_for_plot.append([index, case_count, infered_sub_root_name, type, AP_5to95, AP_5,AP_75])
        #                 # content= 'type {}, AP_5to95_m(AP_5to95_std)|AP_5_m(AP_5_std)|AP_75_m(AP_75_std) ' \
        #                 #          '{:.3f}({:.3f})|{:.3f}({:.3f})|{:.3f}({:.3f})\n'.format(type, m_AP_5to95, std_AP_5to95, m_AP_5, std_AP_5, m_AP_75, std_AP_75)
        #                 # f.write(content)
        #
        #     data_to_plot = DataFrame(summary_for_plot, columns=["x","top_case",'infered_sub_root', 'type',
        #                                                         "$\mathregular{mAP_{0.5, 0.95}}$",
        #                                                         "$\mathregular{mAP_{0.5}}$",
        #                                                         "$\mathregular{mAP_{0.75}}$"])
        #     print("plot_summary data:",data_to_plot )
        #     # save to the disk for avoid next run long time analyze:
        #     data_to_plot.to_pickle("randomness_case_"+ case_name)
            #
            #
            # ## Analyzing is finished. ------------------->
            # # read_from_disk and start to plot:
            data_to_plot = pd.read_pickle("randomness_case_"+ case_name)
            print("each case for plot data:", data_to_plot)
            print("columns:", data_to_plot.columns)
            # for each case
            # customPalette = sns.set_palette(sns.color_palette(colors[case_count], n_colors=1))
            # print("color:", c)
            # sns.set_palette(sns.color_palette(c))
            sns.set_color_codes("dark")

            map_range_data = data_to_plot.melt('x', var_name='$\mathregular{mAP_{IoU}}$',  value_name='mAP', value_vars=["$\mathregular{mAP_{0.5}}$","$\mathregular{mAP_{0.5, 0.95}}$","$\mathregular{mAP_{0.75}}$"])
            tyep_data = data_to_plot.melt('type', var_name='$\mathregular{mAP_{IoU}}$', value_name='mAP',
                                          value_vars=["$\mathregular{mAP_{0.5}}$", "$\mathregular{mAP_{0.5, 0.95}}$",
                                                      "$\mathregular{mAP_{0.75}}$"])
            topcase_data = data_to_plot.melt('top_case', var_name='$\mathregular{mAP_{IoU}}$', value_name='mAP',
                                          value_vars=["$\mathregular{mAP_{0.5}}$", "$\mathregular{mAP_{0.5, 0.95}}$",
                                                      "$\mathregular{mAP_{0.75}}$"])
            # print("map_range_data,", map_range_data)
            # print("map_range_data,", map_range_data)
            # print("map_range_data,", tyep_data)
            merged_new_data_plot = pd.merge(map_range_data,tyep_data)
            merged_new_data_plot = pd.merge(merged_new_data_plot,topcase_data )
            print("merged_new_data_plot,", merged_new_data_plot)
            #
            # # # take out mAP according to the type
            # # seg_Data =merged_new_data_plot.loc[merged_new_data_plot['type']=="segm", "mAP"]
            # seg_Data = merged_new_data_plot[(merged_new_data_plot['type']=="segm") * (merged_new_data_plot['$\mathregular{mAP_{IoU}}$']=="$\mathregular{mAP_{0.5}}$")]
            # seg_Data_mAP_marker = seg_Data.groupby('x', as_index=False)['mAP'].mean()
            # seg_data_line_plot = merged_new_data_plot[merged_new_data_plot['type']=="segm"]

            # print("seg_data_line_plot:", seg_data_line_plot)

            axes = sns.lineplot(x="x",
                         y="mAP",
                         data=merged_new_data_plot, hue= "$\mathregular{mAP_{IoU}}$", markers=[markers[case_count], markers[case_count]], style="type")

            legend = axes.legend()
            # print(legend.texts)
            # if case_count ==0:
            #     legend.texts[4].set_text(legends_text[case_count])
            #     print(legend.texts[4])
            # else:
            #     # print(legend.texts)
            #     legend.texts[11].set_text(legends_text[case_count])
            #     print(legend.texts[11])
            # print(legend.texts)
            # axes = sns.lineplot(x="x",
            #                     y="mAP",
            #                     data=merged_new_data_plot, markers=True, style="type", hue="top_case")

            # sns.scatterplot(x="x", y="mAP",data=seg_Data_mAP_marker, markers="o")

            axes.set_title("mAP(Aug. Randomness)", fontweight='bold')
            box = axes.get_position()
            axes.set_position([box.x0, box.y0, box.width * 0.90, box.height])  # resize position

            # Put a legend to the right side



            xs = range(0, 10)
            labels = ["0", "-11.11", "-22.22", "-33.33", "-44.44", "-55.55", "-66.66", "-77.77", "-88.88", "-100"]
            def format_fn(tick_val, tick_pos):
                if int(tick_val) in xs:
                    return labels[int(tick_val)]
                else:
                    return ''
            #
            axes.xaxis.set_major_formatter(FuncFormatter(format_fn))
            axes.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.axvline(xs[4], color='red', linestyle="--")

            # plt.legend(bbox_to_anchor=(1.05, 1), loc="best", borderaxespad=0.)
            # plt.xlim([0, 10])
            # plt.xlim([0, 1])


            #              y="$\mathregular{mAP_{0.5_0.95}}$",
            #              data=data_to_plot,markers=True, style="type", color=colors[case_count])
            # sns.lineplot(x="x",
            #              y="$\mathregular{mAP_{0.75}}$",
            #              data=data_to_plot, markers=True, style="type",color=colors[case_count])
        # plt.show()
            case_count +=1
        # handles, labels = axes.get_legend_handles_labels()
        # print("handles", handles)
        # print("labels", labels)
        # labels[4] = "worst"
        # labels[1] = "best"
        # new_labels =  labels
        # print("new_labels", new_labels)

        axes.tick_params(axis='both', which='major', labelsize=10)
        axes.tick_params(axis='both', which='minor', labelsize=8)
        axes.set_xlabel("Aug. Reduction (%)", fontweight='bold')
        axes.set_ylabel("mAP",fontweight='bold')
        plt.xticks(rotation=45)
        axes.xaxis.labelpad = 0

        # plt.subplots_adjust(top=0.92, left= 0.1, right=0.9)
        # plt.tight_layout()
        plt.subplots_adjust(left=0.1, bottom=0.2, right=0.75)

        # sns.lineplot(x="x",
        # legend = axes.legend()
        # print(legend.texts)
        # labels =  legend.texts
        # del labels[7:9]
        # legend.texts = labels
        le_handles, le_labels = axes.get_legend_handles_labels()
        print("le_labels:", le_labels)
        head_handles =  le_handles[0:7]
        tail_handles = le_handles[-2:]
        head_labels = le_labels[0:7]
        tail_labels = le_labels[-2:]
        head_handles.extend(tail_handles)
        head_labels.extend(tail_labels)

        print("head_labels:", head_labels)
        # print("tail_labels:", tail_labels)
        # print("labels", labels)
        head_labels[-2] ="Our-bbox"
        head_labels[-1] = "Our-segm"
        head_labels[-4] = "Raw-bbox"
        head_labels[-3] = "Raw-segm"
        axes.legend(handles=head_handles, labels=head_labels, bbox_to_anchor=(1.05, 1), loc="best", borderaxespad=0. )

        # legend.texts[4].set_text("worst")
        # legend.texts[11].set_text("best")
        # print(legend.texts)
        plt.savefig("randomnessCheck.jpg", dpi=300)

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
    # # CASE 5:  code test
    # Saved_model_file_root = "E:\\MyWritings\\Tongue\\2020IEEE\\traindCodesAndLogs\\EXP_8_Mish_WithMyDataNpInterpDistRegOnlyFirstTempCheck"
    # Inference_scripts_root = "E:\\Projects\\MyPolyTongue\\TonguePlusData\\EXP_8_Mish_WithMyDataNpInterpDistRegOnly"
    #
    # output_folder = "E:\\MyWritings\\Tongue\\2020IEEE\\Inferenced_Results\\exp8_MyNpInterpeEval"
    # infer_case(Inference_scripts_root=Inference_scripts_root,
    #            Saved_model_file_root=Saved_model_file_root,
    #            output_folder=output_folder)


    # #CASE 1: base
    # Saved_model_file_root =  "F:\\TonguePolyYOLOLOGS\\MYAugGenerator\\Exp_1_Base"
    # Inference_scripts_root = "C:\\myProjects\\MyPolyTongue\\TonguePlusData\\EXP_1_base_RunningScripts"
    #
    # output_folder =  "F:\\TonguePolyYOLOLOGS\\MYAugGenerator\\paperresults\\exp1"
    # infer_case(Inference_scripts_root= Inference_scripts_root,
    #        Saved_model_file_root=Saved_model_file_root,
    #        output_folder= output_folder)

    # # CASE 2: 1_2 base
    # Saved_model_file_root = "F:\\TonguePolyYOLOLOGS\\MYAugGenerator\\EXP_1_2_base_noSAE"
    # Inference_scripts_root = "C:\\myProjects\\MyPolyTongue\\TonguePlusData\\EXP_1_2_base_noSAE"
    #
    # output_folder = "F:\\TonguePolyYOLOLOGS\\MYAugGenerator\\paperresults\\exp1_2"
    # infer_case(Inference_scripts_root=Inference_scripts_root,
    #            Saved_model_file_root=Saved_model_file_root,
    #            output_folder=output_folder)

    #
    # #CASE 2: Mish
    # Saved_model_file_root =  "F:\\TonguePolyYOLOLOGS\\MYAugGenerator\\Exp_2_Mish"
    # Inference_scripts_root = "C:\\myProjects\\MyPolyTongue\\TonguePlusData\\EXP_2_Mish"
    #
    # output_folder =  "F:\\TonguePolyYOLOLOGS\\MYAugGenerator\\paperresults\\exp2"
    # infer_case(Inference_scripts_root= Inference_scripts_root,
    #        Saved_model_file_root=Saved_model_file_root,
    #        output_folder= output_folder)

    # CASE 3: SAE FRONT NECK
    # Saved_model_file_root = "F:\\TonguePolyYOLOLOGS\\MYAugGenerator\\Exp_3_SAE_FrontNeck"
    # Inference_scripts_root = "C:\\myProjects\\MyPolyTongue\\TonguePlusData\\EXP_3_SAE_FrontNeck"
    #
    # output_folder = "F:\\TonguePolyYOLOLOGS\\MYAugGenerator\\paperresults\\exp3"
    # infer_case(Inference_scripts_root=Inference_scripts_root,
    #            Saved_model_file_root=Saved_model_file_root,
    #            output_folder=output_folder)

    # # CASE 4: SAE mid NECK
    # Saved_model_file_root = "F:\\TonguePolyYOLOLOGS\\MYAugGenerator\\EXP_4_SAE_MidNeck"
    # Inference_scripts_root = "C:\\myProjects\\MyPolyTongue\\TonguePlusData\\EXP_4_SAE_MidNeck"
    #
    # output_folder = "F:\\TonguePolyYOLOLOGS\\MYAugGenerator\\paperresults\\exp4"
    # infer_case(Inference_scripts_root=Inference_scripts_root,
    #            Saved_model_file_root=Saved_model_file_root,
    #            output_folder=output_folder)

    # # CASE 4: SAE mid NECK
    # Saved_model_file_root = "F:\\TonguePolyYOLOLOGS\\MYAugGenerator\\EXP_5_CSP_SAE_FrontNeck"
    # Inference_scripts_root = "C:\\myProjects\\MyPolyTongue\\TonguePlusData\\EXP_5_CSP_SAE_FrontNeck"
    #
    # output_folder = "F:\\TonguePolyYOLOLOGS\\MYAugGenerator\\paperresults\\exp5"
    # infer_case(Inference_scripts_root=Inference_scripts_root,
    #            Saved_model_file_root=Saved_model_file_root,
    #            output_folder=output_folder)

    # # CASE 6: EXP_6_CSP_SAE_MidNeck
    # Saved_model_file_root = "F:\\TonguePolyYOLOLOGS\\MYAugGenerator\\EXP_6_CSP_SAE_MidNeck"
    # Inference_scripts_root = "C:\\myProjects\\MyPolyTongue\\TonguePlusData\\EXP_6_CSP_SAE_MidNeck"
    #
    # output_folder = "F:\\TonguePolyYOLOLOGS\\MYAugGenerator\\paperresults\\exp6"
    # infer_case(Inference_scripts_root=Inference_scripts_root,
    #            Saved_model_file_root=Saved_model_file_root,
    #            output_folder=output_folder)

    # Case1  Exp Mish inference
    # Codes and logs folder
    Saved_model_file_root =  "F:\\randomNess\\TonguePolayPaper1"
    # Inference_scripts_root = "E:\\Projects\\MyPolyTongue\\TonguePlusData\\EXP_2_Mish"

    output_folder =  "F:\\randomNess\\outputs"
    infer_case(Saved_model_file_root=Saved_model_file_root,
           output_folder= output_folder,
               inference_folder= "E:\\Projects\\MyPolyTongue\\TonguePlusData",
               class_path="E:\\Projects\\MyPolyTongue\\TonguePlusData\\yolo_classesTongue.txt")