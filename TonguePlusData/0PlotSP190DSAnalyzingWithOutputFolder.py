# output folder root
from glob import glob
import os
import re

# output_root = "F:\\AngleStepCheck\\output"
output_root = "D:\\RandomShape\\output\\AS1.8\\DS_CHECK"
all_subfolders = glob(output_root + '/*')
print("found all plot points for ", all_subfolders)
NUM_POINTS =  len(all_subfolders)

# resort the subfolders
sub_folder_path = []
path_ids = []
for each in all_subfolders:
    folder_name = os.path.basename(each)
    print("folder_name:", folder_name)
    id =  int(re.findall('\d+', folder_name)[0])
    print("id:", id)
    sub_folder_path.append(each)
    path_ids.append(id)

zipped_list =  zip( path_ids, sub_folder_path)
print("zipped_list:", zipped_list)
sorted_folder_paths =  sorted(zipped_list)
print("sorted_folder_paths:", sorted_folder_paths)


path_ids, sorted_sub_folder_path = zip(*sorted_folder_paths)
print("sorted_sub_folder_path", sorted_sub_folder_path)
AP_paths_list = []
FPS_paths_list = []
import matplotlib.pyplot as plt
import numpy as np
for id, each_folder in sorted_folder_paths:
    print("AS instance:", each_folder)

    for root, dirs, files in os.walk(each_folder):
        for i in files:
            if "paperNeed" in i:
                print("root", root)
                print("dirs", dirs)
                print("files", i)
                result_AP_path =  root+'/'+i
                AP_paths_list.append(result_AP_path)
                # print("result_AP_path:", result_AP_path)
            # if "details" in i:
            #     result_FPS_path=  root+'/'+i
            #     FPS_paths_list.append(result_FPS_path)


print("result_AP_paths:", AP_paths_list)
# print("result_FPS_paths:", FPS_paths_list)

bbox_ap5_95= []
bbox_ap_5 = []
bbox_ap_75 = []
bbox_ap5_95_std= []
bbox_ap_5_std = []
bbox_ap_75_std = []


segm_ap5_95= []
segm_ap_5 = []
segm_ap_75 = []
segm_ap5_95_std= []
segm_ap_5_std = []
segm_ap_75_std = []

# avg_fps_list = []
# std_fps_list = []
avg_box_iou_list = []
std_box_iou_list = []
avg_segm_iou_list = []
std_segm_iou_list = []
# for each_ap_path, each_fps_path in zip(AP_paths_list, FPS_paths_list):
for each_ap_path in AP_paths_list:
    print(each_ap_path)
    # print(each_fps_path)
    with open(each_ap_path) as f:
        ap_line_result = f.readlines()

        bbox = ap_line_result[0]
        segm =  ap_line_result[1]
        print("bbox result:", bbox)
        print("segm result:", segm)
        # for bbox ---->
        bbox_element =  bbox.split(",")
        print("bbox_r elements:", bbox_element)
        bbox_element_num =  bbox_element[1].split()[1:]
        print("bbox_r bbox_element_num:", bbox_element_num)
        for idx, each in enumerate (bbox_element_num):
            each = each.split("|")
            print("each", each)
            if idx == 0:
                bbox_ap5_95.append(float(each[0]))
            elif idx ==1:
                std0 = each[0].replace('(', '').replace(')', '')
                bbox_ap_5.append(float(each[1]))
                bbox_ap5_95_std.append(float(std0))
            elif idx ==2:
                std1 = each[0].replace('(', '').replace(')', '')
                bbox_ap_75.append(float(each[1]))
                bbox_ap_5_std.append(float(std1))
            else:
                std2 = each[0].replace('(', '').replace(')', '')
                bbox_ap_75_std.append(float(std2))
#
        # for segm ---->
        segm_element = segm.split(",")
        print("segm_r elements:", segm_element)
        segm_element_num = segm_element[1].split()[1:]
        print("segm_r segm_element_num:", segm_element_num)
        for idx, each in enumerate(segm_element_num):
            each = each.split("|")
            print("each", each)
            if idx == 0:
                segm_ap5_95.append(float(each[0]))
            elif idx == 1:
                std0 = each[0].replace('(', '').replace(')', '')
                segm_ap_5.append(float(each[1]))
                segm_ap5_95_std.append(float(std0))
            elif idx == 2:
                std1 = each[0].replace('(', '').replace(')', '')
                segm_ap_75.append(float(each[1]))
                segm_ap_5_std.append(float(std1))
            else:
                std2 = each[0].replace('(', '').replace(')', '')
                segm_ap_75_std.append(float(std2))

        # for avg IoU
        avg_box_iou =  bbox_element[-1].split()[2]
        avg_mask_iou = segm_element[-1].split()[2]
        std_box_iou = bbox_element[-1].split()[-1].replace('(', '').replace(')', '')
        std_mask_iou = segm_element[-1].split()[-1].replace('(', '').replace(')', '')
        print("avg_box_iou:", avg_box_iou)
        print("avg_mask_iou:", avg_mask_iou)
        print("std_box_iou:", std_box_iou)
        print("std_mask_iou:", std_mask_iou)
        avg_box_iou_list.append(float(avg_box_iou))
        std_box_iou_list.append(float(std_box_iou))
        avg_segm_iou_list.append(float(avg_mask_iou))
        std_segm_iou_list.append(float(std_mask_iou))
        # temp_means  =  bbox_element_num.split()
        # print("bbox_element_num:", bbox_element_num)
        # for segm
#
#     with open(each_fps_path) as f:
#         fps_line_result = f.readlines()
#         # print("fps readlines:", fps_line_result)
#         avg_fps = float(fps_line_result[5].split()[1])
#         std_fps = float(fps_line_result[6].split()[1])
#         avg_fps_list.append(avg_fps)
#         std_fps_list.append(std_fps)
#
#
print("bbox_ap5_95", bbox_ap5_95)
print("bbox_ap5_95_std", bbox_ap5_95_std)
print("bbox_ap_5", bbox_ap_5)
print("bbox_ap_5_std", bbox_ap_5_std)
print("bbox_ap_75", bbox_ap_75)
print("bbox_ap_75_std", bbox_ap_75_std)
print()
print("segm_ap5_95", segm_ap5_95)
print("segm_ap5_95_std", segm_ap5_95_std)
print("segm_ap_5", segm_ap_5)
print("segm_ap_5_std", segm_ap_5_std)
print("segm_ap_75", segm_ap_75)
print("segm_ap_75_std", segm_ap_75_std)
#
print("avg_box:", avg_box_iou_list)
print("std_box:", std_box_iou)
# print("avg_fps readlines:", avg_fps_list)
# print("std_fps readlines:", std_fps_list)
#
DS = list(np.linspace(5, 55, 11))
DS.insert(0, 1)
print("DS axis:",DS)
#
# plot
fig = plt.figure()
ax1 = fig.add_subplot(111)
avg_box_perform =  np.array(bbox_ap5_95) + np.array(bbox_ap_5) +  np.array(bbox_ap_75)
avg_box_perform = avg_box_perform/3
print("avg_box_perform:", avg_box_perform)
std_box_perform =  np.array(bbox_ap5_95_std) + np.array(bbox_ap_5_std) +  np.array(bbox_ap_75_std)
std_box_perform = std_box_perform/3
print("std_box_perform:", std_box_perform)


avg_mask_perform =  np.array(segm_ap5_95) + np.array(segm_ap_5) + np.array(segm_ap_75)
avg_mask_perform = avg_mask_perform/3
print("avg_mask_perform:", avg_mask_perform)
std_segm_perform =  np.array(segm_ap5_95_std) + np.array(segm_ap_5_std) +  np.array(segm_ap_75_std)
std_segm_perform = std_segm_perform/3
print("std_box_perform:", std_segm_perform)
ax1.errorbar(DS, avg_box_perform, std_box_perform, linestyle='--', marker='o', label="box mAP", capsize=3, c="black", alpha=0.75)
ax1.errorbar(DS, avg_box_iou_list, std_box_iou_list, linestyle='--', marker='o', label="box IoU", capsize=3, c="red", alpha=0.75)


ax1.errorbar(DS, avg_mask_perform, std_segm_perform, linestyle='--', marker='x', label="mask mAP", capsize=3, c="black", alpha=0.75)
ax1.errorbar(DS, avg_segm_iou_list, std_segm_iou_list, linestyle='--', marker='x', label="mask IoU", capsize=3, c="red", alpha=0.75)

idx_largest = np.argmax(avg_segm_iou_list)

plt.annotate("max({:.2f}, {:.2f})" .format(DS[idx_largest], avg_segm_iou_list[idx_largest] ),
             xy=(DS[idx_largest], avg_segm_iou_list[idx_largest]),
             xytext=(8, -32), textcoords='offset points',
            arrowprops=dict(facecolor='green', shrink=0.05))

# ax1.errorbar(DS, bbox_ap5_95, bbox_ap5_95_std, linestyle='--', marker='o', label="box: $\mathregular{mAP_{0.5, 0.95}}$", capsize=3, c="orange", alpha=0.75)
# ax1.errorbar(DS, bbox_ap_5, bbox_ap_5_std, linestyle='--', marker='o', label="box: $\mathregular{mAP_{0.5}}$", capsize=3, c="blue", alpha=0.75)
# ax1.errorbar(DS, bbox_ap_75, bbox_ap_75_std, linestyle='--', marker='o', label="box: $\mathregular{mAP_{0.5}}$", capsize=3, c="green", alpha=0.75)

# ax1.errorbar(DS, segm_ap5_95, segm_ap5_95_std, linestyle='--', marker='x', label="mask: $\mathregular{mAP_{0.5, 0.95}}$", capsize=3, c="orange", alpha=0.75)
# ax1.errorbar(DS, segm_ap_5, segm_ap_5_std, linestyle='--', marker='x', label="mask: $\mathregular{mAP_{0.5}}$", capsize=3, c="blue", alpha=0.75)
# ax1.errorbar(DS, segm_ap_75, segm_ap_75_std, linestyle='--', marker='x', label="mask: $\mathregular{mAP_{0.5}}$", capsize=3, c="green", alpha=0.75)


ax1.set_ylabel('mAP', fontweight='bold')
ax1.set_xlabel('A', fontweight='bold')
ax1.set_title("Distance Scale Check", fontweight='bold')
ax1.tick_params(axis='both', which='major', labelsize=10)
ax1.tick_params(axis='both', which='minor', labelsize=8)
ax1.set_ylim([0.82, 0.98])
# ax2 = ax1.twinx()
# ax2.errorbar(AngleSteps, avg_fps_list, std_fps_list, linestyle='--', marker='>', label="FPS", capsize=3, c="black", alpha=0.5)
# ax2.set_ylabel('FPS', fontweight='bold')
# ax2.tick_params(axis='both', which='major', labelsize=10)
# ax2.tick_params(axis='both', which='minor', labelsize=8)
# ax2.set_ylim([0, 120])
xs = range(0, 13)
print(xs[7])
ds_xs = DS[xs[7]]
ax1.axvline(ds_xs, color='m', linestyle="--")
ax1.legend(loc='lower left', ncol=2, prop={'size': 11})
# ax2.legend(loc='upper left', prop={'size': 10})
plt.savefig('Distance Scale Check.jpg', format='jpg', dpi=300)
plt.show()