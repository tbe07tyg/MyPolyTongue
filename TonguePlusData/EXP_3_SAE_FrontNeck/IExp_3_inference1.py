import cv2
import numpy as np
import os
import time
# need to change
from glob import glob
from TonguePlusData.EXP_3_SAE_FrontNeck.Exp_3_SAE_FrontNeck_Poly_yoloTongue1 import YOLO, my_Gnearator, get_anchors, my_get_random_data #or "import poly_yolo_lite as yolo" for the lite version  ### need to change for different model design
import sys


saved_model_name =  sys.argv[1]
best_h5_path =  sys.argv[2]
output_folder =  sys.argv[3]
FPS_txt = sys.argv[4]


def get_classes(classes_path):
    """loads the classes"""
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

print("cwd:", os.getcwd())
current_file_dir_path = os.path.dirname(os.path.realpath(__file__))
print("current file dir:", current_file_dir_path)
classes_path = current_file_dir_path+'/yolo_classesTongue.txt'
class_names = get_classes(classes_path)
print("class names:", class_names)

# # EXP BASE_1
# modelname =  "Exp_base1"
# h5_name = "ep087-loss21.695-val_loss22.960.h5"
# # EXP BASE_2
# modelname =  "Exp_base2"
# h5_name = "ep061-loss19.139-val_loss20.301.h5"
# # EXP BASE_2
# modelname =  "Exp_base3"
# h5_name = "ep077-loss22.085-val_loss23.323.h5"

# saved_model_folder = "F:\\TonguePolyYOLOLOGS\\MYAugGenerator/" + modelname

output_root = output_folder + "/" + saved_model_name

if not os.path.exists(output_root):
    os.makedirs(output_root)
# inference txt name
inferTXTName = output_root+ '/predictResult_{}.txt'.format(saved_model_name)
file = open(inferTXTName, "w")

#if you want to detect more objects, lower the score and vice versa
trained_model = YOLO(model_path=best_h5_path,  ## need to change
                          classes_path=current_file_dir_path+'/yolo_classesTongue.txt', # this need to specified for your model used classes
                          anchors_path = current_file_dir_path+'/yolo_anchorsTongue.txt',
                          iou=0.5, score=0.5)

#helper function
def translate_color(cls):
    if cls == 0: return (230, 25, 75)
    if cls == 1: return (60, 180, 75)
    if cls == 2: return (255, 225, 25)
    if cls == 3: return (0, 130, 200)
    if cls == 4: return (245, 130, 48)
    if cls == 5: return (145, 30, 180)
    if cls == 7: return (70, 240, 240)
    if cls == 8: return (240, 50, 230)
    if cls == 9: return (210, 245, 60)
    if cls == 10: return (250, 190, 190)
    if cls == 11: return (0, 128, 128)
    if cls == 12: return (230, 190, 255)
    if cls == 13: return (170, 110, 40)
    if cls == 14: return (255, 250, 200)
    if cls == 15: return (128, 0, 128)
    if cls == 16: return (170, 255, 195)
    if cls == 17: return (128, 128, 0)
    if cls == 18: return (255, 215, 180)
    if cls == 19: return (80, 80, 128)

# dir_imgs_name = 'E:\\dataset\\Tongue\\mytonguePolyYolo\\test\\test_inputs' #path_where_are_images_to_clasification
# dir_imgs_name = 'E:\\dataset\\Tongue\\mytonguePolyYolo\\test\\test_inputs' #path_where_are_images_to_clasification
# test_txt_path = current_file_dir_path+'/myTongueTestLab.txt'
# FOR THE LAB
# test_txt_path = current_file_dir_path+'/myTongueTestLab.txt'   # need to change
out_path       = output_root+'/PredOut/' #path, where the images will be saved. The path must exist
if not os.path.exists(out_path):
    os.makedirs(out_path)
MAX_VERTICES = 1000 #that allows the labels to have 1000 vertices per polygon at max. They are reduced for training

# # read test lines from txt file
# with open(test_txt_path) as f:
#     text_lines = f.readlines()
#     print("total {} test samples read".format(len(text_lines)))
#
# # print(text_)
# for i in range(0, len(text_lines)):
#
#     text_lines[i] = text_lines[i].split()
#     #     print(text_lines[i])
#     for element in range(1, len(text_lines[i])):
#         for symbol in range(text_lines[i][element].count(',') - 4, MAX_VERTICES * 2, 2):
#             text_lines[i][element] = text_lines[i][element] + ',0,0'
#         # print(text_lines)
#
# # %%
#
# # browse all images
# print("cwd:", os.getcwd())
# cwd = os.getcwd()
#
# # os.chdir("E:\\Projects\\poly-yolo\\simulator_dataset\\imgs")
classes_path = current_file_dir_path + '/yolo_classesTongue.txt'
anchors_path = current_file_dir_path + '/yolo_anchorsTongue.txt'
class_names = get_classes(classes_path)
num_classes = len(class_names)
anchors = get_anchors(anchors_path)

# raw_input_shape = (416,832) # multiple of 32, hw
input_shape = (256, 256)  # multiple of 32, hw

# for validation dataset  # we need or label and masks are the same shape
# test_input_paths = glob('E:\\dataset\\Tongue\\mytonguePolyYolo\\test\\test_inputs/*')
# test_mask_paths = glob('E:\\dataset\\Tongue\\mytonguePolyYolo\\test\\testLabel\\label512640/*.jpg')

# # for validation dataset  # we need or label and masks are the same shape
# test_input_paths = glob('F:\\dataset\\mytonguePolyYolo\\test\\test_inputs/*')
# test_mask_paths = glob('F:\\dataset\\mytonguePolyYolo\\test\\testLabel\\label512640/*.jpg')
assert len(test_input_paths) == len(test_mask_paths), "test imgs and mask are not the same"
print("total {} testsamples read".format(len(test_input_paths)))
# create data_generator
#
# test_Gen = my_Gnearator(test_input_paths, test_mask_paths, batch_size=4, input_shape=[256, 256],
#                        anchors=anchors, num_classes=num_classes,
#                        train_flag="test")
total_boxes = 0
imgs = 0
fps_list=[]
input_shape=[256,256]
for test_path, mask_path in zip(test_input_paths,test_mask_paths):
    input_img, _ = my_get_random_data(test_path, mask_path, input_shape, None, None, train_or_test="Test", max_boxes=80)
    input_img = np.expand_dims(input_img,0)
    print("input _img shape:", input_img.shape)
    print("test_path:", test_path)
    img = cv2.imread(test_path)
    imgs += 1

    #     print(img)
    overlay = img.copy()
    boxes = []
    scores = []
    classes = []

    # realize prediciction using poly-yolo
    startx = time.time()
    box, score, classs, polygons = trained_model.detect_image(input_img,[img.shape[0], img.shape[1]])
    tmp_fps = 1.0 / (time.time() - startx)
    print('Prediction speed: ', tmp_fps, 'fps')
    fps_list.append(tmp_fps)
    # example, hw to reshape reshape y1,x1,y2,x2 into x1,y1,x2,y2
    if len(box)>0:
        print("there is a box prediction")
    for k in range(0, len(box)):
        boxes.append((box[k][1], box[k][0], box[k][3], box[k][2]))
        scores.append(score[k])
        classes.append(classs[k])

        cv2.rectangle(img, (box[k][1], box[k][0]), (box[k][3], box[k][2]), translate_color(classes[k]), 3, 1)
        cv2.putText(img, "{}:{:.2f}".format(class_names[classs[k]], score[k]), (int(box[k][1]), int(box[k][0])-3 ),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
    total_boxes += len(boxes)

    if len(boxes) == 0:
        continue

    print("write image path")
    file.write(test_path + " ")
    # browse all boxes
    for b in range(0, len(boxes)):

        # draw box and masks on the raw images:-------->
        f = translate_color(classes[b])
        points_to_draw = []
        offset = len(polygons[b]) // 3

        # filter bounding polygon vertices
        for dst in range(0, len(polygons[b]) // 3):
            if polygons[b][dst + offset * 2] > 0.3:
                points_to_draw.append([int(polygons[b][dst]), int(polygons[b][dst + offset])])

        points_to_draw = np.asarray(points_to_draw)
        points_to_draw = points_to_draw.astype(np.int32)
        if points_to_draw.shape[0] > 0:
            cv2.polylines(img, [points_to_draw], True, f, thickness=2)
            cv2.fillPoly(overlay, [points_to_draw], f)

        # write into txt:-------->
        str_to_write = ''

        str_to_write += str(float(boxes[b][0])) + "," + str(float(boxes[b][1])) + "," + str(
            float(boxes[b][2])) + "," + str(float(boxes[b][3])) + ","
        str_to_write += str(scores[b]) + ","
        str_to_write += str(int(classes[b]))

        offset = len(polygons[b]) // 3  # 72 for 24 vertexes. offset = 24
        vertices = 0
        for dst in range(0, len(polygons[b]) // 3):  # 下取整
            if polygons[b][dst + offset * 2] > 0.2:

                str_to_write += "," + str(float(polygons[b][dst])) + "," + str(float(polygons[b][dst + offset]))
                vertices += 1
        str_to_write += " "
        if vertices < 3:
            print("No mask found")
            print('found not correct polygon with ', vertices, ' vertices')
            continue
        # print(str_to_write)
        file.write(str_to_write)
    file.write("\n")

    img = cv2.addWeighted(overlay, 0.4, img, 1 - 0.4, 0)
    cv2.imwrite(out_path + str(imgs) + '.jpg', img)
file.close()
print('total detected boxes: ', total_boxes)
print('imgs: ', imgs)
print("avg fps:", sum(fps_list)/len(fps_list))
with open(FPS_txt, 'a') as f:
    f.write("saved_model_name {}, num_imgs {}, total_detected_box {}, avg_fps {}\n".format (saved_model_name, imgs, total_boxes, sum(fps_list)/len(fps_list)))
