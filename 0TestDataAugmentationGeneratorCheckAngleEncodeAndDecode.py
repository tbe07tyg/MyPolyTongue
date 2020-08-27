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
batchsize = 32

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
            image, mask, annoation_line = get_random_data(temp_img_path, temp_mask_path, input_shape, image_datagen, mask_datagen)
            image_data.append(image)
            # box_data.append(box)
            mask_data.append(mask)
            raw_img_path.append(temp_img_path)
            raw_mask_path.append(temp_mask_path)
            mypolygon_data.append(annoation_line)
            i = (i + 1) % n
        # image_data = np.array(image_data)
        # print("image_data:", image_data.shape)
        # box_data = np.array(box_data)
        # y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        # yield [image_data, *y_true], np.zeros(batch_size)
        yield  np.array(image_data), np.array(mask_data), np.array(raw_img_path), np.array(raw_mask_path), np.array(mypolygon_data)

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
    # myPolygon= None
    skipped = 0
    polygons_line = ''
    c=0
    for obj in contours:
        print(obj.shape)
        myPolygon = obj.reshape([-1, 2])
        print("mypolygon:", myPolygon.shape)
        if myPolygon.shape[0] > max_v:
            print()
            print("too many polygons")
            break


    return aug_image , aug_mask, myPolygon

def encode_polygone():
    "give polygons and encode as angle, ditance , probability"


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

    save_aug_compare_folder ="E:\\dataset\\Tongue\\tongue_dataset_tang_plus\\backup\\AugCompare"
    i=1
    for aug_image, aug_mask, b_raw_input_paths, b_raw_mask_paths, b_polygons in my_data:

        # # fig, ax = plt.subplots(2, 2)
        # print("fetched image shape:", aug_image.shape)
        # print("fetched mask shape:", aug_mask.shape)
        # raw_im = Image.open(b_raw_input_paths[0])
        # raw_mask = Image.open(b_raw_mask_paths[0])
        for idx in range(aug_image.shape[0]):
            aug_i = aug_image[idx]
            aug_m =  aug_mask[idx]
            raw_im = Image.open(b_raw_input_paths[idx])
            raw_mask = Image.open(b_raw_mask_paths[idx])
            plot_aug_compare([ raw_im, raw_mask, aug_i, aug_m, aug_i,b_polygons[idx]], ["raw_img", "raw_mask", "Aug_img", "Aug_mask", "Aug_img","Polygons"], i, idx)




        i+=1
        plt.close()
