import cv2
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

COUNT_F = 0
# ANGLE_STEP = 20
# NUM_ANGLES3 = int(360 // ANGLE_STEP * 3)  # 72 = (360/15)*3
# NUM_ANGLES = int(360 // ANGLE_STEP)

MAX_num_epochs = 4# 10 *443 =  4430 image
seed_list= range(MAX_num_epochs)
print(seed_list)
rotation_range = 90
width_shift_range = 0.3
height_shift_range = 0.3
zoom_range = 0.2
max_v= 1000
root= "E:\\dataset\\Tongue\\tongue_dataset_tang_plus\\backup\\Angumented_dataset/" + "rotate{}_widthshift{}_heightshift{}_zoom{}_num{}".format(rotation_range,width_shift_range, height_shift_range, zoom_range,MAX_num_epochs*443)
augmentation_input_folder =root + "/input"
augmentation_mask_folder =root + "/label"
save_dis_angle_folder = "E:\\dataset\\Tongue\\tongue_dataset_tang_plus\\backup\\Dist_ANGLEComparewithUpsampling/"


if not os.path.exists(save_dis_angle_folder):
    os.makedirs(save_dis_angle_folder)

#
if not os.path.exists(augmentation_input_folder):
    os.makedirs(augmentation_input_folder)

if not os.path.exists(augmentation_mask_folder):
    os.makedirs(augmentation_mask_folder)
seed =0
# for i in range(MAX_num_epochs):
#     seed = seed_list[i]
#
#
#

# Provide the same seed and keyword arguments to the fit and flow methods

print("seed:", seed)
# image_datagen.fit(images, augment=True, seed=seed)
# mask_datagen.fit(masks, augment=True, seed=seed)
# image_generator = image_datagen.flow_from_directory(
#     'E:/dataset/Tongue/tongue_dataset_tang_plus//backup/inputs',
#     class_mode=None,
#     color_mode="rgb",
#     shuffle=False,
#     save_to_dir=augmentation_input_folder,
#     save_format="jpg",
#     seed=seed)
# mask_generator = mask_datagen.flow_from_directory(
#     'E:/dataset/Tongue/tongue_dataset_tang_plus/backup/binary_labels',
#     class_mode=None,
#     shuffle=False,
#     color_mode="grayscale",
#     save_format="jpg",
#     save_to_dir=augmentation_mask_folder,
#     seed=seed)
# # combine generators into one which yields image and masks
# Augmentation_train_generator = zip(image_generator, mask_generator)
# model.fit_generator(
#     train_generator,
#     steps_per_epoch=2000,
#     epochs=50)

# numble of sumples
i=1
# number of batches needd
batchsize = 4

num_batches = math.ceil(443/batchsize)
print("we need batches:", num_batches)
# for input, mask in Augmentation_train_generator :
#     print("batch {}".format(i))
#
#
#     print(image_generator.filenames[i])
#     print("input shape:", input[0].shape)
#     print(type(input))
#     print("mask shape:", mask[0].shape)
#     print(type(mask))
#     # input = Image.fromarray(input)
#     # input.save("your_file.jpeg")
#     # input = Image.fromarray(input)
#     # input.save("your_file.jpeg")
#     # plt.subplot(121)
#     # plt.imshow(input[0]/255)
#     # plt.subplot(122)
#     # plt.imshow(np.squeeze(mask[0]),cmap="gray")
#     # plt.show()
#     i+=1
#     if i > num_batches:
#         break




def my_Gnearator(images_list, masks_list, batch_size, input_shape, train_or_test,decode_method, ANGLE_STEP):
    n = len(images_list)
    print("total_images:", n)
    img_data_gen_args = dict(rotation_range=rotation_range,
                         width_shift_range=width_shift_range,
                         height_shift_range=height_shift_range,
                         zoom_range=zoom_range,
                         shear_range=0.35,
                         horizontal_flip=True,
                         brightness_range=[0.2, 1.3]
                         )
    mask_data_gen_args = dict(rotation_range=rotation_range,
                             width_shift_range=width_shift_range,
                             height_shift_range=height_shift_range,
                             zoom_range=zoom_range,
                             shear_range=0.35,
                             horizontal_flip=True
                             )
    image_datagen = ImageDataGenerator(**img_data_gen_args)
    mask_datagen = ImageDataGenerator(**mask_data_gen_args)
    count = 0
    while True:
        image_data = []
        # box_data = []
        mask_data = []
        raw_img_path =[]
        raw_mask_path = []
        mypolygon_data = []
        depolygon_data = []
        empty_count_data = []
        # my_annotation = []
        # print(images_list)
        # print(masks_list)
        ziped_img_mask_list =  list(zip(images_list, masks_list))
        b =0
        while  b< batch_size:
            print("True")
            if count == 0 and train_or_test == "Train":
                np.random.shuffle(ziped_img_mask_list)
            images_list, masks_list = zip(*ziped_img_mask_list)

            temp_img_path = images_list[count]
            temp_mask_path =  masks_list[count]
            aug_image , aug_mask, myPolygon, decoded_polys, count_emptys= get_random_data(temp_img_path, temp_mask_path,
                                                                                          input_shape, image_datagen,
                                                                                          mask_datagen, train_or_test,
                                                                                          decode_method, ANGLE_STEP)
            print("myPolygon.shape:", myPolygon.shape)
            # check there is zero: if there is boundry points

            print("count before next:", count)
            print("range polygon [{}, {}]".format(myPolygon.min(), myPolygon.max()))
            count = (count + 1) % n
            b+=1
            print(count)
            # if np.any(myPolygon==0) or np.any(myPolygon==aug_image.shape[0]-1) or np.any(myPolygon==aug_image.shape[1]-1):  # roll back.
            #
            #     print("boundary image")
            #     count -=1
            #     b-=1
            #     continue
            print("count after next:", count)
            image_data.append(aug_image)
            # box_data.append(box)
            mask_data.append(aug_mask)
            # raw_img_path.append(temp_img_path)
            # raw_mask_path.append(temp_mask_path)
            mypolygon_data.append(myPolygon)
            depolygon_data.append(decoded_polys)
            empty_count_data.append(count_emptys)
                # count = (count + 1) % n
                # image_data = np.array(image_data)
        # print("image_data:", image_data.shape)
        # box_data = np.array(box_data)
        # y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        # yield [image_data, *y_true], np.zeros(batch_size)
        yield  np.array(image_data), np.array(mask_data), np.array(mypolygon_data), np.array(depolygon_data), np.array(empty_count_data)

def find_countours_and_encode(img_path, mask):
    ret, thresh = cv2.threshold(mask, 127, 255, 0)  # this require the numpy array has to be the uint8 type
    # image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # encode contours into annotation lines ---->
    # if len(contours)>1:
    #     cv2.imwrite(contours_compare_root + "batch{}_idx{}".format(i, idx) + 'image.jpg', aug_image)
    #     cv2.imwrite(contours_compare_root + "batch{}_idx{}".format(i, idx) + 'mask.jpg', aug_mask)
    #
    #     cv2.drawContours(copy_mask,contours, 0, (60 , 180 , 75))
    #     cv2.imshow(" ", copy_mask)
    #     cv2.waitKey() # show on line need divided 255 save into folder should remove keep in 0 to 255
    selected_coutours = []
    for x in range(len(contours)):
        print("countour x:", x, contours[x].shape)
        if contours[x].shape[0] > 8:
            selected_coutours.append(contours[x])

    print("# selected_coutours:", len(selected_coutours))
    annotation_line, myPolygon = encode_polygone(img_path, selected_coutours)
    print("myPolygon:", myPolygon.shape)
    if myPolygon.shape[0] > 1000:
        print("number of polygons overr")
    return annotation_line , myPolygon

def get_random_data(img_path, mask_path, input_shape, image_datagen, mask_datagen, train_or_test, decode_method, ANGLE_STEP):
    # load data ------------------------>
    # image_name = os.path.basename(img_path).replace('.JPG', '')
    # mask_name = os.path.basename(mask_path).replace('.JPG', '')
    # print("img name:", image_name)
    # print("mask name:", mask_name)
    resize_factor =  1
    image = krs_image.load_img(img_path, target_size=(input_shape[0]//resize_factor, input_shape[1]//resize_factor))
    mask = krs_image.load_img(mask_path, grayscale=True, target_size=(input_shape[0]//resize_factor, input_shape[1]//resize_factor))


    image = krs_image.img_to_array(image)
    mask = krs_image.img_to_array(mask)
    if train_or_test == "Train":
        # print("train aug")
        seed = np.random.randint(0, 2147483647)

        aug_image = image_datagen.random_transform(image, seed=seed)

        aug_mask = mask_datagen.random_transform(mask, seed=seed)

        copy_mask = aug_mask.copy().astype(np.uint8)
    else:
        # print("Test no aug")
        aug_image = image
        aug_mask = mask
        copy_mask = mask.copy().astype(np.uint8)



    # decode annotation into angle, distance, probability and return the decoded actual polygones
    # decoded_polys, empty_sections = decode_annotationline(annotation_line)
    # decoded_polys, empty_sections = My_bilinear_decode_annotationline1(annotation_line)
    # decoded_polys, empty_sections = My_bilinear_decode_annotationline1dInterpolate(annotation_line)

    if decode_method =="my":
        annotation_line, myPolygon = find_countours_and_encode(img_path, copy_mask)

        print("myPolygon:", myPolygon.shape)
        if myPolygon.shape[0] > 1000:
            print("number of polygons overr")

        decoded_polys, empty_sections = My_bilinear_decode_annotationlineNP_inter(annotation_line, ANGLE_STEP=ANGLE_STEP)
    elif decode_method =="my_pyramid":

        annotation_line, myPolygon = find_countours_and_encode(img_path, copy_mask)

        print("f:", myPolygon.shape)
        if myPolygon.shape[0] > 1000:
            print("number of polygons overr")

        decoded_polys, empty_sections = My_bilinear_decode_annotationlineNP_inter(annotation_line,
                                                                                  ANGLE_STEP=ANGLE_STEP)
    else:
        annotation_line, myPolygon = find_countours_and_encode(img_path, copy_mask)

        print("myPolygon:", myPolygon.shape)
        if myPolygon.shape[0] > 1000:
            print("number of polygons overr")
        decoded_polys, empty_sections = decode_annotationline(annotation_line, ANGLE_STEP=ANGLE_STEP)
    return aug_image , aug_mask, myPolygon, decoded_polys, empty_sections

def encode_polygone(img_path, contours, MAX_VERTICES =1000):
    "give polygons and encode as angle, ditance , probability"
    skipped = 0
    polygons_line = ''
    c = 0
    for obj in contours:
        # print(obj.shape)
        myPolygon = obj.reshape([-1, 2])
        # print("mypolygon:", myPolygon.shape)
        if myPolygon.shape[0] > MAX_VERTICES:
            print()
            print("too many polygons")
            break

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

    return annotation_line, myPolygon

def decode_annotationline(encoded_annotationline, MAX_VERTICES=1000, max_boxes=80, ANGLE_STEP=5):
    """
    :param encoded_annotationline: string for lines of img_path and objects c and its contours
    :return:
    """
    NUM_ANGLES3 = int(360 // ANGLE_STEP * 3)  # 72 = (360/15)*3
    NUM_ANGLES = int(360 // ANGLE_STEP)

    # preprocessing of lines from string  ---> very important otherwise can not well split
    annotation_line = encoded_annotationline.split()
    # print("len annotation_line:", annotation_line)
    # print(lines[i])
    for element in range(1, len(annotation_line)):
        annotation_line = encoded_annotationline.split()
        print("len annotation_line:", annotation_line[element].count(','))
        # print("MAX_VERTICES:", MAX_VERTICES)
        for symbol in range(annotation_line[element].count(',') - 4, MAX_VERTICES * 2, 2):
            # print("symbol:", symbol)
            annotation_line[element] = annotation_line[element] + ',0,0'
    print(annotation_line[1:])
    box = np.array([np.array(list(map(float, box.split(','))))
                    for box in annotation_line[1:]])

    # correct boxes
    box_data = np.zeros((max_boxes, 5 + NUM_ANGLES3))
    print("box:", box)
    print("box.shape:", box.shape)
    print("box_data:", box_data.shape)
    if len(box) > 0:
        np.random.shuffle(box)
        if len(box) > max_boxes:
            box = box[:max_boxes]
        box_data[:len(box), 0:5] = box[:, 0:5]
    # start polygon --->
    for b in range(0, len(box)):
        boxes_xy = (box[b, 0:2] + box[b, 2:4]) // 2
        for i in range(5, MAX_VERTICES * 2, 2):
            if box[b, i] == 0 and box[b, i + 1] == 0:
                break
            dist_x = boxes_xy[0] - box[b, i]
            dist_y = boxes_xy[1] - box[b, i + 1]
            dist = np.sqrt(np.power(dist_x, 2) + np.power(dist_y, 2))
            if (dist < 1): dist = 1

            angle = np.degrees(np.arctan2(dist_y, dist_x))
            if (angle < 0): angle += 360
            # num of section it belongs to
            iangle = int(int(angle) // ANGLE_STEP)
            print("iangle:", iangle)
            if iangle >= NUM_ANGLES: iangle = NUM_ANGLES - 1

            if dist > box_data[b, 5 + iangle * 3]:  # check for vertex existence. only the most distant is taken
                box_data[b, 5 + iangle * 3] = dist
                box_data[b, 5 + iangle * 3 + 1] = (angle - (
                        iangle * int(ANGLE_STEP))) / ANGLE_STEP  # relative angle
                box_data[b, 5 + iangle * 3 + 2] = 1
    polygon_data = box_data[:, 5:]
    # print("polygon_data.shape:", polygon_data.shape)

    # decoded_polygons = np.zeros((len(box), NUM_ANGLES*2))
    empty_section = 0
    decoded_polygons = []
    for b in range(0, len(box)):
        # convert vertex into polygon xy
        cx= (box_data[b, 0] + box_data[b, 2]) // 2  # .....
        cy = (box_data[b, 1] + box_data[b, 3]) // 2  #..............
        for i in range(0, NUM_ANGLES):
            # print("NUM_ANGLES:", NUM_ANGLES)
            # print("i:", i)
            # print("angle index:",i * 3 + 1)
            # print("dist index:",  i * 3,)
            dela_x = math.cos(math.radians((polygon_data[b, i * 3 + 1] + i) / NUM_ANGLES * 360)) * polygon_data[b,
                i * 3]
            dela_y = math.sin(math.radians((polygon_data[b, i * 3 + 1] + i) / NUM_ANGLES * 360)) * polygon_data[b,
                i * 3]
            x1 = cx - dela_x
            y1 = cy - dela_y

            if dela_x == 0 and dela_y ==0:
                print("dist in 0 section:", polygon_data[b, i * 3])
                print("angle in 0 section:", polygon_data[b, i * 3 + 1])
                empty_section += 1

            # print("x, y:", x1, y1)
            decoded_polygons.append([x1, y1])

            print()
    print("total {} empty section:".format(empty_section))
    return decoded_polygons, empty_section

# def My_bilinear_decode_annotationline1(encoded_annotationline, MAX_VERTICES=1000, max_boxes=80):
#     """
#     :param encoded_annotationline: string for lines of img_path and objects c and its contours
#     :return:
#     """
#
#     # preprocessing of lines from string  ---> very important otherwise can not well split
#     annotation_line = encoded_annotationline.split()
#     # print(lines[i])
#     for element in range(1, len(annotation_line)):
#         for symbol in range(annotation_line[element].count(',') - 4, MAX_VERTICES * 2, 2):
#             annotation_line[element] = annotation_line[element] + ',0,0'
#     box = np.array([np.array(list(map(float, box.split(','))))
#                     for box in annotation_line[1:]])
#     print("box:", box[0])
#     # correct boxes
#     box_data = np.zeros((max_boxes, 5 + NUM_ANGLES3))
#     if len(box) > 0:
#         np.random.shuffle(box)
#         if len(box) > max_boxes:
#             box = box[:max_boxes]
#         box_data[:len(box), 0:5] = box[:, 0:5]
#     # start polygon --->
#     for b in range(0, len(box)):
#         boxes_xy = (box[b, 0:2] + box[b, 2:4]) // 2
#         for i in range(5, MAX_VERTICES * 2, 2):
#             if box[b, i] == 0 and box[b, i + 1] == 0:
#                 break
#             dist_x = boxes_xy[0] - box[b, i]
#             dist_y = boxes_xy[1] - box[b, i + 1]
#             dist = np.sqrt(np.power(dist_x, 2) + np.power(dist_y, 2))
#             if (dist < 1): dist = 1
#
#             angle = np.degrees(np.arctan2(dist_y, dist_x))
#             if (angle < 0): angle += 360
#             # num of section it belongs to
#             iangle = int(angle) // ANGLE_STEP
#
#             if iangle >= NUM_ANGLES: iangle = NUM_ANGLES - 1
#
#             if dist > box_data[b, 5 + iangle * 3]:  # check for vertex existence. only the most distant is taken
#                 box_data[b, 5 + iangle * 3] = dist
#                 box_data[b, 5 + iangle * 3 + 1] = (angle - (
#                         iangle * int(ANGLE_STEP))) / ANGLE_STEP  # relative angle
#                 box_data[b, 5 + iangle * 3 + 2] = 1
#     polygon_data = box_data[:, 5: NUM_ANGLES3+5] # left to right
#
#     # plot our polygon vector check our data
#     dist_data =  polygon_data[:, 0:NUM_ANGLES3:3]
#     angle_data = polygon_data[:, 1:NUM_ANGLES3:3]
#     prob_data = polygon_data[:, 2:NUM_ANGLES3:3]
#     plt.subplot(311)
#     plt.plot(range(len(dist_data[0])),dist_data[0], marker ="o")
#     plt.title("dist")
#     plt.subplot(312)
#     plt.plot(range(len(angle_data[0])), angle_data[0], marker="o")
#     plt.title("angle")
#     plt.subplot(313)
#     plt.plot(range(len(prob_data[0])), prob_data[0], marker="o")
#     plt.title("prob")
#
#     plt.show()
#
#
#
#     # start_end_conect3_v_poly_v =  np.tile(polygon_data, 3)
#     # print("start_end_conect3_v_poly_v:", start_end_conect3_v_poly_v.shape)
#     # print("polygon_data.shape:", polygon_data.shape)
#     # check polygon_data zeros inside
#     for b in range(0, len(box)):
#         # count nonzeros
#         nonzeros = np.count_nonzero(dist_data[b])
#         if nonzeros == 0:
#             continue
#         else:
#             # for i in range(NUM_ANGLES):
#             mask_nonzeros = prob_data[b]> 0
#             # X.compressed()  # get normal array with masked values removed
#             # X.mask  # get a boolean array of the mask
#             prob_dataTemp = prob_data[b][mask_nonzeros] # it automatically discards masked values
#             dist_dataTemp = dist_data[b][mask_nonzeros]
#             angle_dataTemp =angle_data[b][mask_nonzeros]
#             print("dist_dataTemp.shape", dist_dataTemp.shape)
#             plt.subplot(311)
#             plt.plot(range(len(dist_dataTemp)), dist_dataTemp, marker="o")
#             plt.title("dist")
#             plt.subplot(312)
#             plt.plot(range(len(angle_dataTemp)), angle_dataTemp, marker="o")
#             plt.title("angle")
#             plt.subplot(313)
#             plt.plot(range(len(prob_dataTemp)), prob_dataTemp, marker="o")
#             plt.title("prob")
#             plt.show()
#
#             # inter inter polate temp to the length that = num of sections :NUM_ANGLES
#             x = np.arange(len(prob_dataTemp))
#             # y_dist = dist_dataTemp
#             dist_f_inter =  interpolate.interp1d(x, dist_dataTemp)  # default mode = "linear" see :https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html
#             angle_f_inter = interpolate.interp1d(x, angle_dataTemp)
#             prob_f_inter = interpolate.interp1d(x, prob_dataTemp)
#             x_new =  np.linspace(0, len(prob_dataTemp)-1, NUM_ANGLES)
#             y_dist_new = dist_f_inter(x_new)
#             y_angle_new = angle_f_inter(x_new)
#             y_prob_new = prob_f_inter(x_new)
#
#             for i in range(NUM_ANGLES):
#                 polygon_data[b, i * 3 ] = y_dist_new[i]
#                 polygon_data[b, i * 3 + 1] =  y_angle_new[i]
#                 polygon_data[b, i * 3 + 2] = y_prob_new[i]
#             # plt.subplot(311)
#             # plt.plot(range(len(y_dist_new)), y_dist_new, marker="o")
#             # plt.title("linear inter. dist")
#             # plt.subplot(312)
#             # plt.plot(range(len(y_angle_new)), y_angle_new, marker="o")
#             # plt.title("linear inter. angle")
#             # plt.subplot(313)
#             # plt.plot(range(len(y_prob_new)), y_prob_new, marker="o")
#             # plt.title("linear inter. prob")
#             # plt.show()
#
#
#             # dist = polygon_data[b, i * 3+2]
#             #
#             # if dist == 0: # probility == 1 there is vertex
#             #     right_angle_nonzero = 0
#             #     left_angle_nonzero = 0
#             #     right_dist_nonzero = 0
#             #     left_dist_nonzero = 0
#             #     # print("selected angle in raw v:", angle)
#             #     # print("current index in raw v：", i * 3 + 1)
#             #
#             #     # print("try to find right nonzero")
#             #     for right_index in range(len(polygon_data) + (i + 1) * 3 +2, len(start_end_conect3_v_poly_v), 3):
#             #         # print("right search value:", start_end_conect3_v_poly_v[b,right_index])
#             #         if start_end_conect3_v_poly_v[b,right_index] != 0:
#             #             # print("searched right nonzero value:", start_end_conect3_v_poly_v[b,right_index])
#             #             right_dist_nonzero = start_end_conect3_v_poly_v[b, right_index-2]
#             #             print("right_dist_nonzero:", right_dist_nonzero)
#             #             right_angle_nonzero = start_end_conect3_v_poly_v[b,right_index-1]
#             #             break
#             #     # print("try to find left nonzero")
#             #     for left_index in range(len(polygon_data) + (i - 1) * 3 + 2, 0, -3):
#             #         # print("left search value:", start_end_conect3_v_poly_v[b,left_index])
#             #         if start_end_conect3_v_poly_v[b,left_index] != 0:
#             #             # print("searched left nonzero value:", start_end_conect3_v_poly_v[b,left_index])
#             #             left_dist_nonzero = start_end_conect3_v_poly_v[b, left_index-2]
#             #             print("left_dist_nonzero:", left_dist_nonzero)
#             #             left_angle_nonzero = start_end_conect3_v_poly_v[b,left_index-1]
#             #             break
#             #
#             #
#             #     polygon_data[b, i * 3 ] = (right_dist_nonzero + left_dist_nonzero) / 2
#             #     polygon_data[b, i * 3 + 1] = (right_angle_nonzero + left_angle_nonzero)/2
#             #     polygon_data[b, i * 3 + 2] = 1
#             # else:
#             #     continue
#     # plt.subplot(212)
#     # plt.plot(range(len(polygon_data[0])), polygon_data[0], "r", marker ="o" )
#     # plt.show()
#     # decoded_polygons = np.zeros((len(box), NUM_ANGLES*2))
#     empty_section = 0
#     decoded_polygons = []
#     distanceone_count =  0
#     for b in range(0, len(box)):
#         # convert vertex into polygon xy
#         cx= (box_data[b, 0] + box_data[b, 2]) // 2  # .....
#         cy = (box_data[b, 1] + box_data[b, 3]) // 2  #..............
#         for i in range(0, NUM_ANGLES):
#             # print("NUM_ANGLES:", NUM_ANGLES)
#             # print("i:", i)
#             print("angle :", polygon_data[b, i * 3 + 1])
#             print("dist :",  polygon_data[b,i * 3])
#             dela_x = math.cos(math.radians((polygon_data[b, i * 3 + 1] + i) / NUM_ANGLES * 360)) * polygon_data[b,
#                 i * 3]
#             dela_y = math.sin(math.radians((polygon_data[b, i * 3 + 1] + i) / NUM_ANGLES * 360)) * polygon_data[b,
#                 i * 3]
#             x1 = cx - dela_x
#             y1 = cy - dela_y
#
#             if dela_x == 0 and dela_y ==0:
#                 print("b:", b)
#                 print("dist in 0 section:", polygon_data[b, i * 3])
#                 print("angle in 0 section:", polygon_data[b, i * 3+1])
#                 empty_section += 1
#             if  polygon_data[b, i * 3] == 1:
#                 distanceone_count +=1
#                 print("current distance =1 has #:", distanceone_count)
#                 # print("x, y:", x1, y1)
#             decoded_polygons.append([x1, y1])
#
#             print()
#     print("total {} empty section:".format(empty_section))
#     return decoded_polygons, empty_section
#
# def My_bilinear_decode_annotationline1dInterpolate(encoded_annotationline, MAX_VERTICES=1000, max_boxes=80):
#     """
#     :param encoded_annotationline: string for lines of img_path and objects c and its contours
#     :return:
#     """
#
#     # preprocessing of lines from string  ---> very important otherwise can not well split
#     annotation_line = encoded_annotationline.split()
#     # print(lines[i])
#     for element in range(1, len(annotation_line)):
#         for symbol in range(annotation_line[element].count(',') - 4, MAX_VERTICES * 2, 2):
#             annotation_line[element] = annotation_line[element] + ',0,0'
#     box = np.array([np.array(list(map(float, box.split(','))))
#                     for box in annotation_line[1:]])
#     print("box:", box[0])
#     # correct boxes
#     box_data = np.zeros((max_boxes, 5 + NUM_ANGLES))
#     if len(box) > 0:
#         np.random.shuffle(box)
#         if len(box) > max_boxes:
#             box = box[:max_boxes]
#         box_data[:len(box), 0:5] = box[:, 0:5]
#     # start polygon --->
#     print("len(b):", len(box))
#     for b in range(0, len(box)):
#         dist_signal = []
#         angle_signal = []
#         x_old = []
#         y_old = []
#         boxes_xy = (box[b, 0:2] + box[b, 2:4]) // 2
#         print(boxes_xy)
#         print("b:", b)
#         for i in range(5, MAX_VERTICES * 2, 2):
#
#             if box[b, i] == 0 and box[b, i + 1] == 0 and i!=5:  # check reading finished with zeros
#                 print("i:", i)
#                 print(box[b, i])
#                 print(box[b, i + 1])
#                 # plt.plot(range(len(box[b])), box[b])
#                 # plt.savefig(save_dis_angle_folder + "checkb.jpg")
#                 # plt.show()
#                 break
#             dist_x = boxes_xy[0] - box[b, i]
#             dist_y = boxes_xy[1] - box[b, i + 1]
#
#             dist = np.sqrt(np.power(dist_x, 2) + np.power(dist_y, 2))
#
#             # if (dist < 1):  # this will cause to connect to center
#             #     print("there is dist< 1")
#             #     dist = 1
#
#             angle = np.degrees(np.arctan2(dist_y, dist_x))
#             if (angle < 0): angle += 360
#
#             dist_signal.append(dist)
#             angle_signal.append(angle)
#             print("dist_signal.len,", len(dist_signal))
#             print("dist_signal.len,", len(angle_signal))
#             pair_signal = sorted(zip(angle_signal, dist_signal))
#             # for signal in pair_signal:
#             #     print(signal)
#             # print(pair_signal)
#             sorted_angle_signal, sorted_distance =  zip(*pair_signal)
#             print("sorted_angle_signal:", len(sorted_angle_signal))
#             print("sorted_distance:", len(sorted_distance))
#             x_old = list(sorted_angle_signal)
#             y_old =  list(sorted_distance)
#         print("x_old:", np.array(x_old).shape)
#         print("y_old:", np.array(y_old).shape)
#
#         dist_angle_f_inter = interpolate.interp1d(x_old,  y_old, fill_value="extrapolate")  # default mode = "linear" see :https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html
#         x_new = np.linspace(0, 360 - 1, NUM_ANGLES)
#         dist_angle_new = dist_angle_f_inter(x_new)
#
#
#         # plt.subplot(411)
#         # plt.plot(range(len(dist_signal)), dist_signal, marker="o")
#         # plt.title("dist")
#         # plt.subplot(412)
#         # plt.plot(range(len(angle_signal)), angle_signal, marker="o")
#         # plt.title("angle")
#         # plt.subplot(413)
#         # plt.plot(range(len(sorted_distance)), sorted_distance, marker="o")
#         # plt.title("sorted dist according to angle")
#         # plt.subplot(414)
#         # plt.plot(range(len(sorted_angle_signal)), sorted_angle_signal, marker="o")
#         # plt.title("sorted angle")
#         # plt.show()
#         # plt.subplot(211)
#         # plt.plot(sorted_angle_signal, sorted_distance, marker="o")
#         # plt.title("dist(angle)")
#         # plt.xlabel("Angle")
#         # plt.ylabel("Dist")
#         # plt.subplot(212)
#         # plt.plot(x_new, dist_angle_new, marker="o")
#         # plt.title("dist(angle) estimated from 0-360 with NUM_ANGLES")
#         # plt.xlabel("Angle(0 to 360 with NUM_ANGLES)")
#         # plt.ylabel("Dist")
#         # plt.show()
#         #     # num of section it belongs to
#         #     iangle = int(angle) // ANGLE_STEP
#         #
#         #     if iangle >= NUM_ANGLES: iangle = NUM_ANGLES - 1
#
#         # if dist > box_data[b, 5 + iangle * 3]:  # check for vertex existence. only the most distant is taken
#         box_data[b, 5 :] = dist_angle_new
#
#     polygon_data = box_data[:, 5: NUM_ANGLES+5] # left to right
#
#     empty_section = 0
#     decoded_polygons = []
#     distanceone_count =  0
#     for b in range(0, len(box)):
#         # convert vertex into polygon xy
#         cx= (box_data[b, 0] + box_data[b, 2]) // 2  # .....
#         cy = (box_data[b, 1] + box_data[b, 3]) // 2  #..............
#         for i in range(0, NUM_ANGLES):
#             # print("NUM_ANGLES:", NUM_ANGLES)
#             # print("i:", i)
#             print("angle :",i)
#             print("dist :",  polygon_data[b,i ])
#             dela_x = math.cos(math.radians(i/ NUM_ANGLES * 360)) * polygon_data[b,i]
#             dela_y = math.sin(math.radians(i/ NUM_ANGLES * 360))* polygon_data[b,i]
#             x1 = cx - dela_x
#             y1 = cy - dela_y
#
#             if dela_x == 0 and dela_y ==0:
#                 print("b:", b)
#                 print("dist in 0 section:", polygon_data[b, i * 3])
#                 print("angle in 0 section:", polygon_data[b, i * 3+1])
#                 empty_section += 1
#             # if  polygon_data[b, i * 3] == 1:
#             #     distanceone_count +=1
#             #     print("current distance =1 has #:", distanceone_count)
#             #     # print("x, y:", x1, y1)
#             decoded_polygons.append([x1, y1])
#
#             print()
#     print("total {} empty section:".format(empty_section))
#     return decoded_polygons, empty_section

def My_bilinear_decode_annotationlineNP_inter(encoded_annotationline, MAX_VERTICES=1000, max_boxes=80, ANGLE_STEP=5):
    """
    :param encoded_annotationline: string for lines of img_path and objects c and its contours
    :return:
    """

    # print(COUNT_F)
    # NUM_ANGLES3 = int(360 // ANGLE_STEP * 3)  # 72 = (360/15)*3
    NUM_ANGLES = int(360 // ANGLE_STEP)

    # preprocessing of lines from string  ---> very important otherwise can not well split
    annotation_line = encoded_annotationline.split()

    # print(lines[i])
    for element in range(1, len(annotation_line)):

        for symbol in range(annotation_line[element].count(',') - 4, MAX_VERTICES * 2, 2):
            print("in")
            annotation_line[element] = annotation_line[element] + ',0,0'
    box = np.array([np.array(list(map(float, box.split(','))))
                    for box in annotation_line[1:]])
    print("box:", box.shape)

    # correct boxes
    box_data = np.zeros((max_boxes, 5 + NUM_ANGLES))
    if len(box) > 0:
        np.random.shuffle(box)
        if len(box) > max_boxes:
            box = box[:max_boxes]
        box_data[:len(box), 0:5] = box[:, 0:5]
    # start polygon --->
    print("len(b):", len(box))
    for b in range(0, len(box)):
        dist_signal = []
        angle_signal = []
        x_old = []
        y_old = []
        boxes_xy = (box[b, 0:2] + box[b, 2:4]) // 2
        print(boxes_xy)
        print("b:", b)
        for i in range(5, MAX_VERTICES * 2, 2):

            if box[b, i] == 0 and box[b, i + 1] == 0 and i!=5:
                print("i:", i)
                print(box[b, i])
                print(box[b, i + 1])
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
            print("dist_signal.len,", len(dist_signal))
            print("dist_signal.len,", len(angle_signal))
            pair_signal = sorted(zip(angle_signal, dist_signal))
            # for signal in pair_signal:
            #     print(signal)
            # print(pair_signal)
            sorted_angle_signal, sorted_distance =  zip(*pair_signal)
            print("sorted_angle_signal:", len(sorted_angle_signal))
            print("sorted_distance:", len(sorted_distance))
            x_old = list(sorted_angle_signal)
            y_old =  list(sorted_distance)
        print("x_old:", np.array(x_old).shape)
        print("y_old:", np.array(y_old).shape)

       # use numpy.interp
        x_new = np.linspace(0, 359, NUM_ANGLES, endpoint=False)
        dist_angle_new = np.interp(x_new, x_old,  y_old)  # default mode = "linear" see :https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html
        # x_new = np.linspace(0, 360 - 1, NUM_ANGLES)
        # dist_angle_new = dist_angle_f_inter(x_new)


        # plt.subplot(411)
        # plt.plot(range(len(dist_signal)), dist_signal, marker="o")
        # plt.title("dist")
        # plt.subplot(412)
        # plt.plot(range(len(angle_signal)), angle_signal, marker="o")
        # plt.title("angle")
        # plt.subplot(413)
        # plt.plot(range(len(sorted_distance)), sorted_distance, marker="o")
        # plt.title("sorted dist according to angle")
        # plt.subplot(414)
        # plt.plot(range(len(sorted_angle_signal)), sorted_angle_signal, marker="o")
        # plt.title("sorted angle")
        # plt.show()

        # plt.subplot(211)
        # plt.plot(sorted_angle_signal, sorted_distance, marker="o")
        # #
        # plt.title("dist(angle)")
        # plt.xlabel("Angle")
        # plt.ylabel("Dist")
        # plt.xlim(0, 359)
        # plt.ylim(0, 200)
        # plt.subplot(212)
        # plt.plot(x_new, dist_angle_new, marker="o")
        # plt.title("dist(angle) estimated from 0-360 with NUM_ANGLES")
        # plt.xlabel("Angle(0 to 360 with NUM_ANGLES)")
        # plt.ylabel("Dist")
        # plt.xlim(0, 359)
        # plt.ylim(0, 200)
        # plt.savefig(save_dis_angle_folder + "distAngle_BATCH{}".format(COUNT_F+1) + ".jpg")
        #
        # plt.show()
        #     # num of section it belongs to
        #     iangle = int(angle) // ANGLE_STEP
        #
        #     if iangle >= NUM_ANGLES: iangle = NUM_ANGLES - 1

        # if dist > box_data[b, 5 + iangle * 3]:  # check for vertex existence. only the most distant is taken
        box_data[b, 5 :] = dist_angle_new

    polygon_data = box_data[:, 5: NUM_ANGLES+5] # left to right

    empty_section = 0
    decoded_polygons = []
    distanceone_count =  0
    for b in range(0, len(box)):
        # convert vertex into polygon xy
        cx= (box_data[b, 0] + box_data[b, 2]) // 2  # .....
        cy = (box_data[b, 1] + box_data[b, 3]) // 2  #..............
        for i in range(0, NUM_ANGLES):
            # print("NUM_ANGLES:", NUM_ANGLES)
            # print("i:", i)
            print("angle :",i)
            print("dist :",  polygon_data[b,i ])
            dela_x = math.cos(math.radians(i/ NUM_ANGLES * 360)) * polygon_data[b,i]
            dela_y = math.sin(math.radians(i/ NUM_ANGLES * 360))* polygon_data[b,i]
            x1 = cx - dela_x
            y1 = cy - dela_y

            if dela_x == 0 and dela_y ==0:
                print("b:", b)
                print("dist in 0 section:", polygon_data[b, i * 3])
                print("angle in 0 section:", polygon_data[b, i * 3+1])
                empty_section += 1
            # if  polygon_data[b, i * 3] == 1:
            #     distanceone_count +=1
            #     print("current distance =1 has #:", distanceone_count)
            #     # print("x, y:", x1, y1)
            decoded_polygons.append([x1, y1])

            print()
    print("total {} empty section:".format(empty_section))
    return decoded_polygons, empty_section


def plot_aug_compare(image_list,name_list, batch_idx, img_idx):
    num_images_per_raw =  int(len(image_list)/2)
    fig, ax = plt.subplots(num_images_per_raw, 2)
    print("num_images_per_raw", num_images_per_raw)
    range_img_select =  range(0, len(image_list), 2)
    for x in range_img_select:
        print(x)
    print("range_img_select",range_img_select)
    for i in range(num_images_per_raw):

        print("row index:", i)
        if "Aug" in name_list[range_img_select[i]]:
            print(name_list[range_img_select[i]])
            ax[i, 0].imshow(image_list[range_img_select[i]]/255)
        else:
            ax[i, 0].imshow(image_list[range_img_select[i]])
        ax[i, 0].set_title(name_list[range_img_select[i]] + "; range[{:.2f}, {:.2f}]".format(np.min(image_list[range_img_select[i]]), np.max(image_list[range_img_select[i]])))

        if "Polygons" in name_list[range_img_select[i]+1]:
            print(image_list[range_img_select[i]+1].shape)
            ax[i, 1].imshow(image_list[range_img_select[i]])
            patch = Polygon(image_list[range_img_select[i]+1], facecolor=None, fill=False, color='r')
            ax[i, 1].add_patch(patch)

        else:
            ax[i, 1].imshow(np.squeeze(image_list[range_img_select[i]+1] ), cmap="gray")
        ax[i, 1].set_title(name_list[range_img_select[i]+1]+ "; range[{:.2f}, {:.2f}]".format(np.min(image_list[range_img_select[i]+1]), np.max(image_list[range_img_select[i]+1])))


    plt.tight_layout()
    plt.savefig(save_aug_compare_folder + "/{}_{}.jpg".format(batch_idx, img_idx))
    plt.close()


def get_iou_vector(A, B):
    # Numpy version

    batch_size = A.shape[0]
    metric = 0.0
    for batch in range(batch_size):
        t, p = A[batch], B[batch]
        true = np.sum(t)
        pred = np.sum(p)

        # deal with empty mask first
        if true == 0:
            metric += (pred == 0)
            continue

        # non empty mask case.  Union is never empty
        # hence it is safe to divide by its number of pixels
        intersection = np.sum(t * p)
        union = true + pred - intersection
        iou = intersection / union

        # iou metrric is a stepwise approximation of the real iou over 0.5
        # iou = np.floor(max(0, (iou - 0.45) * 20)) / 10

        metric += iou

    # teake the average over all images in batch
    metric /= batch_size
    return metric

if __name__ == '__main__':
    my_IoU_list= []
    their_IoU_list = []

    # check train
    # raw_input_paths = glob('E:\\dataset\\Tongue\\tongue_dataset_tang_plus\\backup\\inputs\\Tongue/*')
    # raw_binary_paths = glob('E:\\dataset\\Tongue\\tongue_dataset_tang_plus\\backup\\binary_labels\\Tongue/*.jpg')
    # print("len of imgs:", len(raw_input_paths))

    # check test
    raw_input_paths = glob('E:\\dataset\\Tongue\\fromTangNewDataset\\testImg\data/*')
    raw_binary_paths = glob('E:\\dataset\\Tongue\\fromTangNewDataset\\testLabel\\datamadebymyself\label512640/*.jpg')
    print("len of imgs:", len(raw_input_paths))
    # path  to compare countour
    # contours_compare_root = "E:\\MyWritings\\Tongue\\2020IEEE\\TMI\\imgs\\annoationRaw"
    contours_compare_root = "E:\\MyWritings\\Tongue\\2020IEEE\\TMI\\imgs\\annoationMine"
    # decode_choice=["their"]

    # decode_choice = ["my_pyramid"]
    decode_choice = ["my"]
    # my_pyramid
    my_IoU_data = []
    their_IoU_data= []

    # angle_steps = np.linspace(0.1, 20, 10)
    # print("angle_steps", angle_steps)

    for decode in decode_choice:
        for angle_step in np.linspace(1, 1, 1):
            print("angle_step:", angle_step)
        # for angle_step in np.linspace(30, 30 , 1):
            IoU_list =[]
            my_data =  my_Gnearator(raw_input_paths, raw_binary_paths, batch_size=4, input_shape=[256, 256],
                                    train_or_test="Test",decode_method=decode, ANGLE_STEP=angle_step)
            # my_data = my_Gnearator(raw_input_paths, raw_binary_paths, batch_size=4, input_shape=[256, 256],
            #                        train_or_test="Train", decode_method=decode, ANGLE_STEP=angle_step)

            print(my_data)
            empty_section_list = []
            save_aug_compare_folder ="E:\\dataset\\Tongue\\tongue_dataset_tang_plus\\backup\\AugCompare"
            i=1
            for aug_image , aug_mask, myPolygon, decoded_polys, empty_sections in my_data:
                print("empty_sections:", empty_sections)
                overlay = aug_image.copy()
                print("decoded poly shape:", decoded_polys.shape)
                for idx in range(aug_image.shape[0]):


                    # calculate IoUs for the countours
                    GT_MASK = np.zeros([aug_image[idx].shape[0], aug_image[idx].shape[1]])
                    encoded_mask = np.zeros([aug_image[idx].shape[0], aug_image[idx].shape[1]])
                    cv2.fillPoly(GT_MASK, np.int32([myPolygon[idx]]), color=(1))
                    cv2.fillPoly(encoded_mask, np.int32([decoded_polys[idx]]), color=(1))
                    # print("type(GT_MASK):", type(GT_MASK))
                    # cv2.imshow("GT_MASK ", GT_MASK)
                    # cv2.imshow("encoded_mask ", encoded_mask)
                    # cv2.waitKey()
                    # calculate iou
                    GT_MASK =  np.expand_dims(GT_MASK, 0)
                    encoded_mask = np.expand_dims(encoded_mask, 0)
                    one_iou = get_iou_vector(GT_MASK,encoded_mask)
                    print("temp iou:", one_iou)
                    IoU_list.append(one_iou)

                    # for saving
                    # print("range{}, {}".format(overlay[idx].min(), overlay[idx].max()))

                    # background_img = overlay[idx]/255.0 # CV2 ONLY SHOW IMAGES WHITH 0 TO 1 OTHERWISE IT WILL BE ALL WHITE

                    background_img = overlay[idx]  # CV2 ONLY SHOW IMAGES WHITH 0 TO 1 OTHERWISE IT WILL BE ALL WHITE

                    background_img_rgb = cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB)  # BRG---PIL read to RGB
                    cv2.polylines(background_img_rgb, np.int32([myPolygon[idx]]), True, color=(230, 25 , 75 ),
                                  thickness=2)  # 加中括号 ，否则报错
                    layer1 =  background_img_rgb.copy()
                    cv2.polylines(background_img_rgb, np.int32([decoded_polys[idx]]), True, color=(60 , 180 , 75 ),
                                  thickness=2)  # 加中括号 ，否则报错
                    layer2 = background_img_rgb.copy()
                    cv2.addWeighted(layer1, 0.5, layer2, 1 - 0.5,
                                    0, layer2) # for Transparent

                    cv2.putText(background_img_rgb, "IoU: {:.2f}".format(one_iou), (30, 30 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255 ), 1)

                    # cv2.polylines(background_white, np.int32([myPolygon[0]]), True, color=(230 / 255.0, 25 / 255.0, 75 / 255.0),
                    #               thickness=2)  # 加中括号 ，否则报错
                    # cv2.fillPoly(background_img_rgb, np.int32([myPolygon[0]]), color=(230, 25, 75))
                    # cv2.imshow(" ", background_img_rgb)
                    # cv2.waitKey() # show on line need divided 255 save into folder should remove keep in 0 to 255
                    root = os.path.join(contours_compare_root, "{}".format(angle_step))
                    if not os.path.exists(root):
                        os.makedirs(root)
                    cv2.imwrite(root + "/batch{}_idx{}".format(i, idx) + '.jpg', background_img_rgb)

                    print()


                i += 1
                if i >  math.ceil(len(raw_input_paths)/batchsize):
                    break
                plt.close()

            print("mean IoU_list:", np.mean(IoU_list))
            print("std IoU_list", np.std(IoU_list))
            if decode =="my":
                my_IoU_data.append((np.mean(IoU_list), np.std(IoU_list)))

            else:
                their_IoU_data.append((np.mean(IoU_list), np.std(IoU_list)))
        #
        # if decode == "my":
        #     with open('my_IoU_data_polydeocde', 'wb') as fp:
        #         pickle.dump(my_IoU_data, fp)
        # else:
        #     with open('their_IoU_data_polydeocde', 'wb') as fp:
        #         pickle.dump(their_IoU_data, fp)

