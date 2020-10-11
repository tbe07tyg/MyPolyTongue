# raw binary label folder
from glob import glob
import os
from glob import glob
import sys
import numpy as np
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import random
import pandas as pd
import numpy as np
from glob import glob
from tensorflow.keras.preprocessing import image as krs_image
from random import randint
from PIL import Image
# plt.figure()
import math
from matplotlib.patches import Polygon
from scipy import interpolate
import cv2
import pickle
from scipy import signal
from skimage.draw import random_shapes

ANGLE_STEP  = 14
NUM_ANGLES  = int(360 // ANGLE_STEP)

def My_bilinear_decode_annotationlineNP_inter(encoded_annotationline, MAX_VERTICES=1000, max_boxes=80):
    """
    :param encoded_annotationline: string for lines of img_path and objects c and its contours
    :return: box_data(min_x, min_y, max_x, max_y, c, dists1.dist2...) shape(b, NUM_ANGLE+5)
    """
    # print(COUNT_F)
    # preprocessing of lines from string  ---> very important otherwise can not well split
    annotation_line = encoded_annotationline.split()
    # print(lines[i])
    for element in range(1, len(annotation_line)):
        # print(element)
        for symbol in range(annotation_line[element].count(',') - 4, MAX_VERTICES * 2, 2):
            annotation_line[element] = annotation_line[element] + ',0,0'
    box = np.array([np.array(list(map(float, box.split(','))))
                    for box in annotation_line[1:]])
    # print("box:", box[0])
    # correct boxes
    box_data = np.zeros((max_boxes, 5 + NUM_ANGLES))
    if len(box) > 0:
        np.random.shuffle(box)
        if len(box) > max_boxes:
            box = box[:max_boxes]
        box_data[:len(box), 0:5] = box[:, 0:5]
    # start polygon --->
    # print("len(b):", len(box))
    for b in range(0, len(box)):
        dist_signal = []
        angle_signal = []
        x_old = []
        y_old = []
        boxes_xy = (box[b, 0:2] + box[b, 2:4]) // 2
        # print(boxes_xy)
        # print("b:", b)
        for i in range(5, MAX_VERTICES * 2, 2):

            if box[b, i] == 0 and box[b, i + 1] == 0 and i!=5:
                # print("i:", i)
                # print(box[b, i])
                # print(box[b, i + 1])
                # plt.plt(range(len(box[b]), box[b]))
                # plt.show()
                break
            dist_x = boxes_xy[0] - box[b, i]
            dist_y = boxes_xy[1] - box[b, i + 1]

            dist = np.sqrt(np.power(dist_x, 2) + np.power(dist_y, 2))

            # if (dist < 1):
            #     print("there is dist< 1")
            #     dist = 1

            angle = np.degrees(np.arctan2(dist_y, dist_x))
            if (angle < 0): angle += 360

            dist_signal.append(dist)
            angle_signal.append(angle)
            # print("dist_signal.len,", len(dist_signal))
            # print("dist_signal.len,", len(angle_signal))
            pair_signal = sorted(zip(angle_signal, dist_signal))
            # for signal in pair_signal:
            #     print(signal)
            # print(pair_signal)
            sorted_angle_signal, sorted_distance =  zip(*pair_signal)
            # print("sorted_angle_signal:", len(sorted_angle_signal))
            # print("sorted_distance:", len(sorted_distance))
            x_old = list(sorted_angle_signal)
            y_old =  list(sorted_distance)
        # print("x_old:", np.array(x_old).shape)
        # print("y_old:", np.array(y_old).shape)

       # use numpy.interp
        x_new = np.linspace(0, 359, NUM_ANGLES, endpoint=False)
        dist_angle_new = np.interp(x_new, x_old,  y_old)
        box_data[b, 5 :] = dist_angle_new

    return box_data

def encode_polygone(img_path, contours, MAX_VERTICES =1000):
    "give polygons and encode as angle, ditance , probability"
    skipped = 0
    polygons_line = ''
    c = 0
    my_poly_list =[]
    for obj in contours:
        # print(obj.shape)
        myPolygon = obj.reshape([-1, 2])
        # print("mypolygon:", myPolygon.shape)
        if myPolygon.shape[0] > MAX_VERTICES:
            print()
            print("too many polygons")
            break
        my_poly_list.append(myPolygon)

        min_x = sys.maxsize
        max_x = 0
        min_y = sys.maxsize
        max_y = 0
        polygon_line = ''

        # for po
        for x, y in myPolygon:
            # print("({}, {})".format(x, y))
            if x > max_x: max_x = x
            if y > max_y: max_y = y
            if x < min_x: min_x = x
            if y < min_y: min_y = y
            polygon_line += ',{},{}'.format(x, y)
        if max_x - min_x <= 1.0 or max_y - min_y <= 1.0:
            skipped += 1
            continue

        polygons_line += ' {},{},{},{},{}'.format(min_x, min_y, max_x, max_y, c) + polygon_line

    annotation_line = img_path + polygons_line

    return annotation_line, my_poly_list


def my_get_random_data(img_path, mask_path, input_shape, image_datagen, mask_datagen, train_or_test):
    # load data ------------------------>
    # image_name = os.path.basename(img_path).replace('.JPG', '')
    # mask_name = os.path.basename(mask_path).replace('.JPG', '')
    # print("img name:", image_name)
    # print("mask name:", mask_name)
    image = krs_image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))
    mask = krs_image.load_img(mask_path, grayscale=True, target_size=(input_shape[0], input_shape[1]))
    image = krs_image.img_to_array(image)
    mask = krs_image.img_to_array(mask)
    # image = np.expand_dims(image, 0)
    # mask = np.expand_dims(mask, 0)
    # print("img shape before aug:", image.shape)
    # print("mask shape before aug:", mask.shape)
    # augment data ----------------------->
    if train_or_test == "Train":
        # print("train aug")
        seed = np.random.randint(0, 2147483647)

        aug_image = image_datagen.random_transform(image, seed=seed)

        aug_mask = mask_datagen.random_transform(mask, seed=seed)

        copy_mask = aug_mask.copy().astype(np.uint8)
    else:
        # print("Test no aug")
        aug_image = image
        copy_mask = mask.copy().astype(np.uint8)

    # print("mask shape after aug:", np.squeeze(aug_mask).shape)
    # aug_image = krs_image.img_to_array(aug_image)
    # aug_mask = krs_image.img_to_array(aug_mask)
    # find polygons with augmask ------------------------------------>
    # imgray = cv2.cvtColor(np.squeeze(copy_mask), cv2.COLOR_BGR2GRAY)
    # print(copy_mask)
    ret, thresh = cv2.threshold(copy_mask, 127, 255, 0)  # this require the numpy array has to be the uint8 type
    aug_mask =thresh
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    selected_coutours = []
    for x in range(len(contours)):
        # print("countour x:", x, contours[x].shape)
        if contours[x].shape[0] > 8:  # require one contour at lest 8 polygons(360/40=9)
            selected_coutours.append(contours[x])
    # print("# selected_coutours:", len(selected_coutours))



    # encode contours into annotation lines ---->
    annotation_line, myPolygon = encode_polygone(img_path, selected_coutours)
    # decode contours annotation line into distance
    box_data = My_bilinear_decode_annotationlineNP_inter(annotation_line)

    # normal the image ----------------->
    aug_image = aug_image / 255.0
    aug_mask = aug_mask / 255.0
    aug_mask =  np.expand_dims(aug_mask, -1)  # since in our case ,we only have one class, if multiple classes binary labels concate at the last dimension
    # print("aug_mask.shape:", aug_mask.shape)
    return aug_image, box_data, myPolygon, aug_mask, annotation_line

# # main
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
if __name__ == '__main__':
    # # # for train dataset
    train_input_paths = glob('E:\\dataset\\Tongue\\tongue_dataset_tang_plus\\backup\\inputs\\Tongue/*')
    train_mask_paths = glob('E:\\dataset\\Tongue\\tongue_dataset_tang_plus\\backup\\binary_labels\\Tongue/*.jpg')
    print("len of train imgs:", len(train_input_paths))
    input_shape = [256,256]

    class_names =  ["Kidney", "Stomach", "Liver", "Lung"]
    num_classes = len(class_names)
    classes = range(num_classes)

    save_multi_region = "E:\\dataset\\Tongue\\tongue_dataset_tang_plus\\backup\\MultiRegions"
    if not os.path.exists(save_multi_region):
        os.makedirs(save_multi_region)
    count=0
    for input_path, mask_path in zip(train_input_paths, train_mask_paths):
        input_img, box_data, myPolygon, aug_mask, annotation_line = my_get_random_data(input_path, mask_path, input_shape, None, None,
                                                                train_or_test="Test")
        count+=1
        myPolygon= np.array(myPolygon)
        print("myPolygon.shape", myPolygon.shape)
        # print(box_data[:, 0:4])
        box =  box_data[:, 0:4]
        background = input_img.copy() *255
        background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
        # print("raw box:", box)

        background2 =  background.copy()



        for k in range(0, len(box)):
            if not all(box[k] == 0):
                print("raw box:", box[k])
                # box[k] = [int(s) for s in  box[k]]
                # print([int(s) for s in  box[k]])

                box_temp = [int(s) for s in  box[k]] #

                print("box_temp :",box_temp)   # (min_x, min_y, max_x, max_y)

                myPolygon_temp =  myPolygon[0]
                print("f.shape:", myPolygon_temp.shape)

                min_x =  myPolygon_temp[:, 0].min()
                min_y =  myPolygon_temp[:, 1].min()
                max_x =  myPolygon_temp[:, 0].max()
                max_y =  myPolygon_temp[:, 1].max()

                print("min_x, min_y, max_x, max_y:", min_x, min_y, max_x, max_y)
                # # create mask seperately
                # raw_ptx = pts = np.array([[min_x, min_y],[max_x, min_y],
                #                           [min_x, max_y],[max_x, max_y]])
                # print("raw_ptx.shape", raw_ptx.shape)
                # raw_ptx = raw_ptx.reshape((-1, 1, 2))
                # print("raw_ptx.shape", raw_ptx.shape)

                # Position Threshold for regions
                pos_th = 0.2
                # calculate the largest width and height

                w_x =  max_x - min_x
                h_y =  max_y - min_y
                pos_x_51 =  pos_th * w_x
                pos_y_51 =  pos_th * h_y
                print("pos_x_51:", pos_x_51)
                print("pos_y_51:", pos_y_51)

                th1 = (min_x + pos_x_51, min_y + pos_y_51)
                th2 = (max_x - pos_x_51, min_y + pos_y_51)
                th3 = (min_x + pos_x_51, max_y - pos_y_51)
                th4 = (max_x - pos_x_51, max_y - pos_y_51)
                print("th1:", th1)
                print("th2:", th2)
                print("th3:", th3)
                print("th4:", th4)

                # poly_left_liver_index = np.where(myPolygon_temp[:, 0] < th1[0] and myPolygon_temp[:, 1] < th3[1] and myPolygon_temp[:, 1] > th1[1] )


                # cv2.fillPoly(background2,raw_ptx, translate_color(classes[k]))
                # cv2.fillPoly(background2, raw_ptx, True, (0, 255, 255))

                cv2.rectangle(background, (box_temp[0], box_temp[1]), (box_temp[2], box_temp[3]), translate_color(classes[k]), 3, 1)
                cv2.putText(background, "{}:{:.2f}".format("Tongue", 1),
                            (int(box_temp[0]), int(box_temp[1]) - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1)

                # fig, ax = plt.subplots(2, 2)
                # ax[0, 0].imshow(np.squeeze(input_img))
                # ax[0, 0].set_title('Input')
                # ax[0, 1].imshow(np.squeeze(aug_mask), cmap="gray")
                # ax[0, 1].set_title('Raw Mask')
                #
                # ax[1, 0].imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB).astype(np.int))
                # ax[1, 0].set_title('Input with raw bbox')
                # print("with bbox range [{}, {}]".format(background.min(), background.max()))

                # cv2.polylines(background2, [myPolygon], True, translate_color(classes[k]), thickness=2)
                # ax[1, 1].imshow(cv2.cvtColor(background2, cv2.COLOR_BGR2RGB).astype(np.int))
                # ax[1, 1].set_title('Input with raw bbox')



                # top Kidney ---------------->
                th11 = (th1[0] - 1, th1[1] - 1)
                # th31 = (th3[0] - 1, th3[1] - 1)
                # print(myPolygon[0][:, 0] <= th13[0])
                poly_left_Kidney_index = np.where(myPolygon[0][:, 1] < th11[1])
                print("poly_left_Kidney_index:", poly_left_Kidney_index)
                print("myPolygon.shape:", myPolygon.shape)
                left_Kidney_poly = myPolygon[:, poly_left_Kidney_index[0], :]
                th11= np.reshape(np.array( th11), [1, 1, -1])
                print(" th11.shape",  th11)
                print("left_Kidney_poly:", left_Kidney_poly)
                # left_Kidney_poly = np.concatenate([left_Kidney_poly, th13], axis=1).astype(int)
                # print("left_Kidney_poly:", left_Kidney_poly)

                min_Kidney_left_x = left_Kidney_poly[0][:, 0].min()
                max_Kidney_left_x = left_Kidney_poly[0][:, 0].max()
                min_Kidney_left_y = left_Kidney_poly[0][:, 1].min()
                max_Kidney_left_y = left_Kidney_poly[0][:, 1].max()

                # c_left_liver =  (int((min_liver_left_x + max_liver_left_x)/2), int((min_liver_left_y+max_liver_left_y)/2))
                # print("c_left_liver:", c_left_liver)
                cv2.polylines(background2, [left_Kidney_poly], True, translate_color(classes[1]), thickness=2)
                cv2.putText(background2, "{}:{:.2f}".format("Kidney", 1),
                            (min_Kidney_left_x, min_Kidney_left_y - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            translate_color(classes[1]), 1)

                # Bottom Lung ---------------->
                th34 = (th3[0] +1, th3[1] + 1)
                # th31 = (th3[0] - 1, th3[1] - 1)
                # print(myPolygon[0][:, 0] <= th13[0])
                poly_Lung_index = np.where(myPolygon[0][:, 1] > th34[1])
                print("poly_Lung_index:", poly_Lung_index)
                print("myPolygon.shape:", myPolygon.shape)
                Lung_poly = myPolygon[:, poly_Lung_index[0], :]
                th34 = np.reshape(np.array(th34), [1, 1, -1])
                print(" th34.shape", th34)
                print("Lung_poly:", Lung_poly)
                # left_Kidney_poly = np.concatenate([left_Kidney_poly, th13], axis=1).astype(int)
                # print("left_Kidney_poly:", left_Kidney_poly)

                min_Lung_x = Lung_poly[0][:, 0].min()
                max_Lung_x = Lung_poly[0][:, 0].max()
                min_Lung_y = Lung_poly[0][:, 1].min()
                max_Lung_y = Lung_poly[0][:, 1].max()

                # c_left_liver =  (int((min_liver_left_x + max_liver_left_x)/2), int((min_liver_left_y+max_liver_left_y)/2))
                # print("c_left_liver:", c_left_liver)
                cv2.polylines(background2, [Lung_poly], True, translate_color(5), thickness=2)
                cv2.putText(background2, "{}:{:.2f}".format("Lung", 1),
                            (min_Lung_x +25, max_Lung_y + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            translate_color(5), 1)

                # left liver ---------------->
                th13 = (th1[0] - 1, th1[1] + 1)
                th31 = (min_Lung_x - 1, min_Lung_y - 1)
                print(myPolygon[0][:, 0] <= th13[0])
                poly_left_liver_index = np.where((myPolygon[0][:, 0] <= th13[0]) &
                                                 (myPolygon[0][:, 1] >= th13[1]) &
                                                 (myPolygon[0][:, 1] <= th31[1]) &
                                                 (myPolygon[0][:, 0] <= th31[0]))
                print("poly_left_liver_index:", poly_left_liver_index)
                print("myPolygon.shape:", myPolygon.shape)
                left_liver_poly = myPolygon[:, poly_left_liver_index[0], :]
                th13 = np.reshape(np.array(th13), [1, 1, -1])
                th31 = np.reshape(np.array(th31), [1, 1, -1])
                print("th13.shape", th13)
                print("left_liver_poly:", left_liver_poly)
                left_liver_poly = np.concatenate([left_liver_poly, th13], axis=1).astype(int)
                print("left_liver_poly:", left_liver_poly)

                min_liver_left_x = left_liver_poly[0][:, 0].min()
                max_liver_left_x = left_liver_poly[0][:, 0].max()
                min_liver_left_y = left_liver_poly[0][:, 1].min()
                max_liver_left_y = left_liver_poly[0][:, 1].max()

                # c_left_liver =  (int((min_liver_left_x + max_liver_left_x)/2), int((min_liver_left_y+max_liver_left_y)/2))
                # print("c_left_liver:", c_left_liver)
                cv2.polylines(background2, [left_liver_poly], True, translate_color(classes[0]), thickness=2)
                cv2.putText(background2, "{}:".format("liver"),
                            (min_liver_left_x-10, max_liver_left_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            translate_color(classes[0]), 1)
                cv2.putText(background2, "{:.2f}".format(1),
                            (min_liver_left_x-10, max_liver_left_y + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            translate_color(classes[0]), 1)

                # right liver ---------------->
                th24 = (th2[0] + 1, th2[1] + 1)
                th42 = (max_Lung_x+ 1, min_Lung_y - 1)
                poly_right_liver_index = np.where(
                    (myPolygon[0][:, 0] >=th24[0]) & (myPolygon[0][:, 1] >= th24[1]) & (myPolygon[0][:, 1] <= th42[1]))
                print("poly_right_liver_index:", poly_right_liver_index)
                print("myPolygon.shape:", myPolygon.shape)
                right_liver_poly = myPolygon[:, poly_right_liver_index[0], :]
                th24 = np.reshape(np.array(th24), [1, 1, -1])
                print("th24.shape", th24)
                print("right_liver_poly:", right_liver_poly)
                right_liver_poly = np.concatenate([right_liver_poly, th24], axis=1).astype(int)
                print(" right_liver_poly:",  right_liver_poly)

                min_liver_right_x = right_liver_poly[0][:, 0].min()
                max_liver_right_x = right_liver_poly[0][:, 0].max()
                min_liver_right_y = right_liver_poly[0][:, 1].min()
                max_liver_right_y = right_liver_poly[0][:, 1].max()

                # c_left_liver =  (int((min_liver_left_x + max_liver_left_x)/2), int((min_liver_left_y+max_liver_left_y)/2))
                # print("c_left_liver:", c_left_liver)
                cv2.polylines(background2, [right_liver_poly], True, translate_color(classes[0]), thickness=2)
                cv2.putText(background2, "{}:".format("liver"),
                            (min_liver_right_x+15, max_liver_right_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            translate_color(classes[0]), 1)
                cv2.putText(background2, "{:.2f}".format(1),
                            (min_liver_right_x +15, max_liver_right_y + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            translate_color(classes[0]), 1)

                # Stomach ---------------->
                th14 = (th1[0] + 1, th1[1] + 1)
                th23 = (th2[0] - 1, th2[1] + 1)
                th1423 = ((th14[0]+th23[0])/2 , (th14[1]+th23[1])/2)
                th32 = (min_Lung_x + 1, min_Lung_y - 1)
                th41 = (max_Lung_x - 1, min_Lung_y  - 1)
                th2341 = (( th41[0] + th23[0]) / 2, (th41[1] + th23[1]) / 2)
                th4132 = ((th41[0] + th32[0]) / 2, (th41[1] + th32[1]) / 2)
                th3214 = ((th14[0] + th32[0]) / 2, (th14[1] + th32[1]) / 2)
                # poly_Stomach_index = np.where(
                #     (myPolygon[0][:, 0] >= th14[0]) & (myPolygon[0][:, 1] >= th14[1]) & (myPolygon[0][:, 0] <= th41[1]) & (myPolygon[0][:, 1] <= th41[1]))
                # print("poly_Stomach_index:", poly_Stomach_index)
                # print("myPolygon.shape:", myPolygon.shape)
                # Stomach_poly = myPolygon[:, poly_Stomach_index[0], :]

                th14 = np.reshape(np.array(th14), [1, 1, -1])
                th23 = np.reshape(np.array(th23), [1, 1, -1])
                th32 = np.reshape(np.array(th32), [1, 1, -1])
                th41 = np.reshape(np.array(th41), [1, 1, -1])

                th1423 = np.reshape(np.array(th1423), [1, 1, -1])
                th2341 = np.reshape(np.array(th2341), [1, 1, -1])
                th4132 = np.reshape(np.array(th4132), [1, 1, -1])
                th3214 = np.reshape(np.array(th3214), [1, 1, -1])
                # # print("th24.shape", th24)
                # print("Stomach_poly:", right_liver_poly)
                Stomach_poly = np.concatenate([th14, th1423, th23, th2341, th41,th4132,  th32, th3214], axis=1).astype(int)
                print("Stomach_poly:", Stomach_poly)

                min_Stomach_x = Stomach_poly[0][:, 0].min()
                max_Stomach_x = Stomach_poly[0][:, 0].max()
                min_Stomach_y = Stomach_poly[0][:, 1].min()
                max_Stomach_y = Stomach_poly[0][:, 1].max()

                c_Stomach = (
                int((min_Stomach_x + max_Stomach_x) / 2), int((min_Stomach_y + max_Stomach_y) / 2))
                # c_left_liver =  (int((min_liver_left_x + max_liver_left_x)/2), int((min_liver_left_y+max_liver_left_y)/2))
                # print("c_left_liver:", c_left_liver)
                cv2.polylines(background2, [Stomach_poly], True, translate_color(classes[2]), thickness=2)
                cv2.putText(background2, "{}:".format("Stomach"),
                            (c_Stomach[0] -30, c_Stomach[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            translate_color(classes[2]), 1)
                cv2.putText(background2, "{:.2f}".format(1),
                            (c_Stomach[0] - 15, c_Stomach[1]+15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            translate_color(classes[2]), 1)


                plt.imshow(cv2.cvtColor(background2, cv2.COLOR_BGR2RGB).astype(np.int))
                plt.title('Input with multi-contours')
                print("with bbox range [{}, {}]".format(background2.min(), background2.max()))
                plt.tight_layout()
                plt.savefig(save_multi_region + "/{}.jpg".format(count))


