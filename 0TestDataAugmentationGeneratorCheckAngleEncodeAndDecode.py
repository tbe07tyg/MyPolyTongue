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
ANGLE_STEP = 20
NUM_ANGLES3 = int(360 // ANGLE_STEP * 3)  # 72 = (360/15)*3
NUM_ANGLES = int(360 // ANGLE_STEP)
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




def my_Gnearator(images_list, masks_list, batch_size, input_shape):
    n = len(images_list)
    i = 0
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
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(ziped_img_mask_list)
            images_list, masks_list = zip(*ziped_img_mask_list)

            temp_img_path = images_list[i]
            temp_mask_path =  masks_list[i]
            aug_image , aug_mask, myPolygon, decoded_polys, count_emptys= get_random_data(temp_img_path, temp_mask_path, input_shape, image_datagen, mask_datagen)
            image_data.append(aug_image)
            # box_data.append(box)
            mask_data.append(aug_mask)
            raw_img_path.append(temp_img_path)
            raw_mask_path.append(temp_mask_path)
            mypolygon_data.append(myPolygon)
            depolygon_data.append(decoded_polys)
            empty_count_data.append(count_emptys)
            i = (i + 1) % n
        # image_data = np.array(image_data)
        # print("image_data:", image_data.shape)
        # box_data = np.array(box_data)
        # y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        # yield [image_data, *y_true], np.zeros(batch_size)
        yield  np.array(image_data), np.array(mask_data), np.array(mypolygon_data), np.array(depolygon_data), np.array(empty_count_data)

def get_random_data(img_path, mask_path, input_shape, image_datagen, mask_datagen, random=True, max_boxes=80, hue_alter=20, sat_alter=30, val_alter=30, proc_img=True):
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
    print("sys.maxsize", sys.maxsize)
    seed =  np.random.randint(0, 2147483647)
    # aug_image = image_datagen.random_transform(image, seed=seed)
    #
    # # print("img shape after aug:", image.shape)
    # aug_mask = mask_datagen.random_transform(mask, seed=seed)
    aug_image = image_datagen.random_transform(image, seed=seed)

    # print("img shape after aug:", image.shape)
    aug_mask = mask_datagen.random_transform(mask, seed=seed)
    print("mask shape after aug:", aug_mask.shape)
    copy_mask  = aug_mask.copy().astype(np.uint8)

    print("mask shape after aug:", np.squeeze(aug_mask).shape)
    # aug_image = krs_image.img_to_array(aug_image)
    # aug_mask = krs_image.img_to_array(aug_mask)
    # find polygons with augmask ------------------------------------>
    # imgray = cv2.cvtColor(np.squeeze(copy_mask), cv2.COLOR_BGR2GRAY)
    # print(copy_mask)
    ret, thresh = cv2.threshold(copy_mask, 127, 255, 0) # this require the numpy array has to be the uint8 type
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # encode contours into annotation lines ---->
    annotation_line, myPolygon = encode_polygone(img_path, contours)
    # decode annotation into angle, distance, probability and return the decoded actual polygones
    decoded_polys, empty_sections = decode_annotationline(annotation_line)

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

def decode_annotationline(encoded_annotationline, MAX_VERTICES=1000, max_boxes=80):
    """
    :param encoded_annotationline: string for lines of img_path and objects c and its contours
    :return:
    """

    # preprocessing of lines from string  ---> very important otherwise can not well split
    annotation_line = encoded_annotationline.split()
    # print(lines[i])
    for element in range(1, len(annotation_line)):
        for symbol in range(annotation_line[element].count(',') - 4, MAX_VERTICES * 2, 2):
            annotation_line[element] = annotation_line[element] + ',0,0'
    box = np.array([np.array(list(map(float, box.split(','))))
                    for box in annotation_line[1:]])

    # correct boxes
    box_data = np.zeros((max_boxes, 5 + NUM_ANGLES3))
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
            iangle = int(angle) // ANGLE_STEP

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
                empty_section += 1

            # print("x, y:", x1, y1)
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

if __name__ == '__main__':
    raw_input_paths = glob('E:\\dataset\\Tongue\\tongue_dataset_tang_plus\\backup\\inputs\\Tongue/*')
    raw_binary_paths = glob('E:\\dataset\\Tongue\\tongue_dataset_tang_plus\\backup\\binary_labels\\Tongue/*.jpg')
    print("len of imgs:", len(raw_input_paths))





    my_data =  my_Gnearator(raw_input_paths, raw_binary_paths, batch_size=4, input_shape=[256, 256])
    print(my_data)
    empty_section_list = []
    save_aug_compare_folder ="E:\\dataset\\Tongue\\tongue_dataset_tang_plus\\backup\\AugCompare"
    i=1
    for aug_image , aug_mask, myPolygon, decoded_polys, empty_sections in my_data:
        print("empty_sections:", empty_sections)
        # # fig, ax = plt.subplots(2, 2)
        # print("fetched myPolygon shape:", myPolygon.shape)
        # print("fetched decoded_polys shape:", decoded_polys.shape)
        # raw_im = Image.open(b_raw_input_paths[0])
        # raw_mask = Image.open(b_raw_mask_paths[0])
        for idx in range(aug_image.shape[0]):
            empty_section_list.append(empty_sections[idx])
            # print("fetched myPolygon shape:", myPolygon[idx].shape)
            # print("fetched decoded_polys shape:", decoded_polys[idx].shape)
            aug_i = aug_image[idx]
            aug_m =  aug_mask[idx]
            # raw_im = Image.open(b_raw_input_paths[idx])
            # raw_mask = Image.open(b_raw_mask_paths[idx])
            plot_aug_compare([aug_i, aug_m, aug_i, myPolygon[idx], aug_i, decoded_polys[idx]], ["Aug_img", "Aug_mask", "Aug_img","rawPolygons", "Aug_img","DePolygons"], i, idx)




        i+=1
        if i >20:
            break

        plt.close()
    print(" emptys list:", empty_section_list)
    # print("std emptys", np.std(empty_section_list))
    print("mean emptys:", np.mean(empty_section_list))
    print("std emptys", np.std(empty_section_list))
