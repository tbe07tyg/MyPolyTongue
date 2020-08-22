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

        # ax[0, 0].imshow(raw_im)
        # ax[0, 0].set_title("raw image; range[{:.2f}, {:.2f}]".format(np.min(raw_im), np.max(raw_im)))
        # ax[0, 1].imshow(raw_mask, cmap="gray")
        # ax[0, 1].set_title("raw mask; range[{:.2f}, {:.2f}]".format(np.min(raw_mask), np.max(raw_mask)))
        #
        # ax[1, 0].imshow(aug_image[0] / 255.0)
        # ax[1, 0].set_title("Aug image; range[{:.2f}, {:.2f}]".format(np.min(aug_image), np.max(aug_image)))
        # ax[1, 1].imshow(np.squeeze(aug_mask[0]), cmap="gray")
        # ax[1, 1].set_title("Aug mask; range[{:.2f}, {:.2f}]".format(np.min(aug_mask), np.max(aug_mask)))
        # plt.tight_layout()
        # plt.savefig(save_aug_compare_folder + "/{}_1.jpg".format(i))
        #
        # # ..................
        #
        # fig, ax = plt.subplots(2, 2)
        # print("fetched image shape:", aug_image.shape)
        # print("fetched mask shape:", aug_mask.shape)
        # raw_im = Image.open(b_raw_input_paths[1])
        # raw_mask = Image.open(b_raw_mask_paths[1])
        # ax[0, 0].imshow(raw_im)
        # ax[0, 0].set_title("raw image; range[{:.2f}, {:.2f}]".format(np.min(raw_im), np.max(raw_im)))
        # ax[0, 1].imshow(raw_mask, cmap="gray")
        # ax[0, 1].set_title("raw mask; range[{:.2f}, {:.2f}]".format(np.min(raw_mask), np.max(raw_mask)))
        #
        # ax[1, 0].imshow(aug_image[1] / 255.0)
        # ax[1, 0].set_title("Aug image; range[{:.2f}, {:.2f}]".format(np.min(aug_image), np.max(aug_image)))
        # ax[1, 1].imshow(np.squeeze(aug_mask[1]), cmap="gray")
        # ax[1, 1].set_title("Aug mask; range[{:.2f}, {:.2f}]".format(np.min(aug_mask), np.max(aug_mask)))
        # plt.tight_layout()
        # plt.savefig(save_aug_compare_folder + "/{}_2.jpg".format(i))
        #
        # # ..................
        #
        # fig, ax = plt.subplots(2, 2)
        # print("fetched image shape:", aug_image.shape)
        # print("fetched mask shape:", aug_mask.shape)
        # raw_im = Image.open(b_raw_input_paths[2])
        # raw_mask = Image.open(b_raw_mask_paths[2])
        # ax[0, 0].imshow(raw_im)
        # ax[0, 0].set_title("raw image; range[{:.2f}, {:.2f}]".format(np.min(raw_im), np.max(raw_im)))
        # ax[0, 1].imshow(raw_mask, cmap="gray")
        # ax[0, 1].set_title("raw mask; range[{:.2f}, {:.2f}]".format(np.min(raw_mask), np.max(raw_mask)))
        #
        # ax[1, 0].imshow(aug_image[2] / 255.0)
        # ax[1, 0].set_title("Aug image; range[{:.2f}, {:.2f}]".format(np.min(aug_image), np.max(aug_image)))
        # ax[1, 1].imshow(np.squeeze(aug_mask[2]), cmap="gray")
        # ax[1, 1].set_title("Aug mask; range[{:.2f}, {:.2f}]".format(np.min(aug_mask), np.max(aug_mask)))
        # plt.tight_layout()
        # plt.savefig(save_aug_compare_folder + "/{}_3.jpg".format(i))
        #
        # # ..................
        #
        # fig, ax = plt.subplots(2, 2)
        # print("fetched image shape:", aug_image.shape)
        # print("fetched mask shape:", aug_mask.shape)
        # raw_im = Image.open(b_raw_input_paths[3])
        # raw_mask = Image.open(b_raw_mask_paths[3])
        # ax[0, 0].imshow(raw_im)
        # ax[0, 0].set_title("raw image; range[{:.2f}, {:.2f}]".format(np.min(raw_im), np.max(raw_im)))
        # ax[0, 1].imshow(raw_mask, cmap="gray")
        # ax[0, 1].set_title("raw mask; range[{:.2f}, {:.2f}]".format(np.min(raw_mask), np.max(raw_mask)))
        #
        # ax[1, 0].imshow(aug_image[3] / 255.0)
        # ax[1, 0].set_title("Aug image; range[{:.2f}, {:.2f}]".format(np.min(aug_image), np.max(aug_image)))
        # ax[1, 1].imshow(np.squeeze(aug_mask[3]), cmap="gray")
        # ax[1, 1].set_title("Aug mask; range[{:.2f}, {:.2f}]".format(np.min(aug_mask), np.max(aug_mask)))
        # plt.tight_layout()
        # plt.savefig(save_aug_compare_folder + "/{}_4.jpg".format(i))


        i+=1
        plt.close()
        # plt.show()
    # load data
    # the color conversion is later. it is not necessary to realize bgr->rgb->hsv->rgb
    # print("get_random_data line[0]:", line[0])
    # print(os.getcwd())

    # image = cv.imread(line[0])
    # # print("get_random_data image:", image)
    #
    # iw = image.shape[1]
    # ih = image.shape[0]
    # h, w = input_shape
    # box = np.array([np.array(list(map(float, box.split(','))))
    #                 for box in line[1:]])
    #
    # if not random:
    #     # resize image
    #     scale = min(w / iw, h / ih)
    #     nw = int(iw * scale)
    #     nh = int(ih * scale)
    #     dx = (w - nw) // 2
    #     dy = (h - nh) // 2
    #     image_data = 0
    #     if proc_img:
    #         # image = image.resize((nw, nh), Image.BICUBIC)
    #         image = cv.cvtColor(
    #             cv.resize(image, (nw, nh), interpolation=cv.INTER_CUBIC), cv.COLOR_BGR2RGB)
    #         image = Image.fromarray(image)
    #         new_image = Image.new('RGB', (w, h), (128, 128, 128))
    #         new_image.paste(image, (dx, dy))
    #         image_data = np.array(new_image) / 255.
    #     # correct boxes
    #     box_data = np.zeros((max_boxes, 5 + NUM_ANGLES3))
    #     if len(box) > 0:
    #         np.random.shuffle(box)
    #         if len(box) > max_boxes:
    #             box = box[:max_boxes]
    #         box[:, [0, 2]] = box[:, [0, 2]] * scale + dx
    #         box[:, [1, 3]] = box[:, [1, 3]] * scale + dy
    #         box_data[:len(box), 0:5] = box[:, 0:5]
    #         for b in range(0, len(box)):
    #             for i in range(5, MAX_VERTICES * 2, 2):
    #                 if box[b,i] == 0 and box[b, i + 1] == 0:
    #                     continue
    #                 box[b, i] = box[b, i] * scale + dx
    #                 box[b, i + 1] = box[b, i + 1] * scale + dy
    #
    #         # box_data[:, i:NUM_ANGLES3 + 5] = 0
    #
    #         for i in range(0, len(box)):
    #             boxes_xy = (box[i, 0:2] + box[i, 2:4]) // 2
    #
    #             for ver in range(5, MAX_VERTICES * 2, 2):
    #                 if box[i, ver] == 0 and box[i, ver + 1] == 0:
    #                     break
    #                 dist_x = boxes_xy[0] - box[i, ver]
    #                 dist_y = boxes_xy[1] - box[i, ver + 1]
    #                 dist = np.sqrt(np.power(dist_x, 2) + np.power(dist_y, 2))
    #                 if (dist < 1): dist = 1 #to avoid inf or nan in log in loss
    #
    #                 angle = np.degrees(np.arctan2(dist_y, dist_x))
    #                 if (angle < 0): angle += 360
    #                 iangle = int(angle) // ANGLE_STEP
    #                 relative_angle = (angle - (iangle * int(ANGLE_STEP))) / ANGLE_STEP
    #
    #                 if dist > box_data[i, 5 + iangle * 3]:  # check for vertex existence. only the most distant is taken
    #                     box_data[i, 5 + iangle * 3] = dist
    #                     box_data[i, 5 + iangle * 3 + 1] = relative_angle
    #                     box_data[i, 5 + iangle * 3 + 2] = 1 # problbility  mask to be 1 for the exsitance of the vertex otherwise =0
    #     return image_data, box_data


    # # resize image
    # random_scale = rd.uniform(.6, 1.4)
    # scale = min(w / iw, h / ih)
    # nw = int(iw * scale * random_scale)
    # nh = int(ih * scale * random_scale)
    #
    # # force nw a nh to be an even
    # if (nw % 2) == 1:
    #     nw = nw + 1
    # if (nh % 2) == 1:
    #     nh = nh + 1
    #
    # # jitter for slight distort of aspect ratio
    # if np.random.rand() < 0.3:
    #     if np.random.rand() < 0.5:
    #         nw = int(nw*rd.uniform(.8, 1.0))
    #     else:
    #         nh = int(nh*rd.uniform(.8, 1.0))
    #
    # image = cv.resize(image, (nw, nh), interpolation=cv.INTER_CUBIC)
    # nwiw = nw/iw
    # nhih = nh/ih
    #
    # # clahe. applied on resized image to save time. but before placing to avoid
    # # the influence of homogenous background
    # clahe = cv.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    # lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    # l, a, b = cv.split(lab)
    # cl = clahe.apply(l)
    # limg = cv.merge((cl, a, b))
    # image = cv.cvtColor(limg, cv.COLOR_LAB2BGR)
    #
    # # place image
    # dx = rd.randint(0, max(w - nw, 0))
    # dy = rd.randint(0, max(h - nh, 0))
    #
    # new_image = np.full((h, w, 3), 128, dtype='uint8')
    # new_image, crop_coords, new_img_coords = random_crop(
    #     image, new_image)
    #
    # # flip image or not
    # flip = rd.random() < .5
    # if flip:
    #     new_image = cv.flip(new_image, 1)
    #
    # # distort image
    # hsv = np.int32(cv.cvtColor(new_image, cv.COLOR_BGR2HSV))
    #
    # # linear hsv distortion
    # hsv[..., 0] += rd.randint(-hue_alter, hue_alter)
    # hsv[..., 1] += rd.randint(-sat_alter, sat_alter)
    # hsv[..., 2] += rd.randint(-val_alter, val_alter)
    #
    # # additional non-linear distortion of saturation and value
    # if np.random.rand() < 0.5:
    #     hsv[..., 1] = hsv[..., 1]*rd.uniform(.7, 1.3)
    #     hsv[..., 2] = hsv[..., 2]*rd.uniform(.7, 1.3)
    #
    # hsv[..., 0][hsv[..., 0] > 179] = 179
    # hsv[..., 0][hsv[..., 0] < 0] = 0
    # hsv[..., 1][hsv[..., 1] > 255] = 255
    # hsv[..., 1][hsv[..., 1] < 0] = 0
    # hsv[..., 2][hsv[..., 2] > 255] = 255
    # hsv[..., 2][hsv[..., 2] < 0] = 0
    #
    # image_data = cv.cvtColor(
    #     np.uint8(hsv), cv.COLOR_HSV2RGB).astype('float32') / 255.0
    #
    # # add noise
    # if np.random.rand() < 0.15:
    #     image_data = np.clip(image_data + np.random.rand() *
    #                          image_data.std() * np.random.random(image_data.shape), 0, 1)
    #
    # # correct boxes
    # box_data = np.zeros((max_boxes, 5 + NUM_ANGLES3))
    #
    # if len(box) > 0:
    #     np.random.shuffle(box)
    #     # rescaling separately because 5-th element is class
    #     box[:, [0, 2]] = box[:, [0, 2]] * nwiw  # for x
    #     # rescale polygon vertices
    #     box[:, 5::2] = box[:, 5::2] * nwiw
    #     # rescale polygon vertices
    #     box[:, [1, 3]] = box[:, [1, 3]] * nhih  # for y
    #     box[:, 6::2] = box[:, 6::2] * nhih
    #
    #     # # mask out boxes that lies outside of croping window ## new commit deleted
    #     # mask = (box[:, 1] >= crop_coords[0]) & (box[:, 3] < crop_coords[1]) & (
    #     #     box[:, 0] >= crop_coords[2]) & (box[:, 2] < crop_coords[3])
    #     # box = box[mask]
    #
    #     # transform boxes to new coordinate system w.r.t new_image
    #     box[:, :2] = box[:, :2] - [crop_coords[2], crop_coords[0]] + [new_img_coords[2], new_img_coords[0]]
    #     box[:, 2:4] = box[:, 2:4] - [crop_coords[2], crop_coords[0]] + [new_img_coords[2], new_img_coords[0]]
    #     if flip:
    #         box[:, [0, 2]] = (w-1) - box[:, [2, 0]]
    #
    #     box[:, 0:2][box[:, 0:2] < 0] = 0
    #     box[:, 2][box[:, 2] >= w] = w-1
    #     box[:, 3][box[:, 3] >= h] = h-1
    #     box_w = box[:, 2] - box[:, 0]
    #     box_h = box[:, 3] - box[:, 1]
    #     box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
    #
    #     if len(box) > max_boxes:
    #         box = box[:max_boxes]
    #
    #     box_data[:len(box), 0:5] = box[:, 0:5]
    #
    # #-------------------------------start polygon vertices processing-------------------------------#
    # for b in range(0, len(box)):
    #     boxes_xy = (box[b, 0:2] + box[b, 2:4]) // 2
    #     for i in range(5, MAX_VERTICES * 2, 2):
    #         if box[b, i] == 0 and box[b, i + 1] == 0:
    #             break
    #         box[b, i:i+2] = box[b, i:i+2] - [crop_coords[2], crop_coords[0]] + [new_img_coords[2], new_img_coords[0]] # transform
    #         if flip: box[b, i] = (w - 1) - box[b, i]
    #         dist_x = boxes_xy[0] - box[b, i]
    #         dist_y = boxes_xy[1] - box[b, i + 1]
    #         dist = np.sqrt(np.power(dist_x, 2) + np.power(dist_y, 2))
    #         if (dist < 1): dist = 1
    #
    #         angle = np.degrees(np.arctan2(dist_y, dist_x))
    #         if (angle < 0): angle += 360
    #         # num of section it belongs to
    #         iangle = int(angle) // ANGLE_STEP
    #
    #         if iangle>=NUM_ANGLES: iangle = NUM_ANGLES-1
    #
    #         if dist > box_data[b, 5 + iangle * 3]: # check for vertex existence. only the most distant is taken
    #             box_data[b, 5 + iangle * 3]     = dist
    #             box_data[b, 5 + iangle * 3 + 1] = (angle - (iangle * int(ANGLE_STEP))) / ANGLE_STEP #relative angle
    #             box_data[b, 5 + iangle * 3 + 2] = 1
    # #---------------------------------end polygon vertices processing-------------------------------#
    # return image_data, box_data