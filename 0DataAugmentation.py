import cv2
import os
from glob import glob
import sys
import numpy as np
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from random import randint
from PIL import Image
# plt.figure()
import math
raw_input_paths =  glob('E:\\dataset\\Tongue\\tongue_dataset_tang_plus\\backup\\inputs\\Tongue/*.jpg')
raw_binary_paths =  glob('E:\\dataset\\Tongue\\tongue_dataset_tang_plus\\backup\\binary_labels/*.jpg')
# we create two instances with the same arguments
# data_gen_args = dict(featurewise_center=True,
#                      featurewise_std_normalization=True,
#                      rotation_range=90,
#                      width_shift_range=0.1,
#                      height_shift_range=0.1,
#                      zoom_range=0.2)
# imgs =  input = Image.fromarray(input)
#     input.save("your_file.jpeg")
# number of images want to genearte
MAX_num_epochs = 20# 10 *443 =  4430 image
seed_list= range(MAX_num_epochs)
print(seed_list)
rotation_range = 45
width_shift_range = 0.1
height_shift_range = 0.1
zoom_range = 0.2
root= "E:\\dataset\\Tongue\\tongue_dataset_tang_plus\\backup\\Angumented_dataset/" + "rotate{}_widthshift{}_heightshift{}_zoom{}_num{}".format(rotation_range,width_shift_range, height_shift_range, zoom_range,MAX_num_epochs*443)
augmentation_input_folder =root + "/input"
augmentation_mask_folder =root + "/label"

if not os.path.exists(augmentation_input_folder):
    os.makedirs(augmentation_input_folder)

if not os.path.exists(augmentation_mask_folder):
    os.makedirs(augmentation_mask_folder)
seed =0
for i in range(MAX_num_epochs):
    seed = seed_list[i]



    data_gen_args = dict(
                         rotation_range=rotation_range,
                         width_shift_range=width_shift_range,
                         height_shift_range=height_shift_range,
                         zoom_range=zoom_range)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    # Provide the same seed and keyword arguments to the fit and flow methods

    print("seed:", seed)
    # image_datagen.fit(images, augment=True, seed=seed)
    # mask_datagen.fit(masks, augment=True, seed=seed)
    image_generator = image_datagen.flow_from_directory(
        'E:/dataset/Tongue/tongue_dataset_tang_plus//backup/inputs',
        class_mode=None,
        color_mode="rgb",
        shuffle=False,
        save_to_dir=augmentation_input_folder,
        save_format="jpg",
        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(
        'E:/dataset/Tongue/tongue_dataset_tang_plus/backup/binary_labels',
        class_mode=None,
        shuffle=False,
        color_mode="grayscale",
        save_format="jpg",
        save_to_dir=augmentation_mask_folder,
        seed=seed)
    # combine generators into one which yields image and masks
    Augmentation_train_generator = zip(image_generator, mask_generator)
    # model.fit_generator(
    #     train_generator,
    #     steps_per_epoch=2000,
    #     epochs=50)

    # numble of sumples
    i=1
    # number of batches needd
    batchsize = 32
    print ("we need batches:", math.ceil(443/batchsize))
    num_batches = math.ceil(443/batchsize)
    for input, mask in Augmentation_train_generator :
        print("batch {}".format(i))


        print(image_generator.filenames[i])
        print("input shape:", input[0].shape)
        print(type(input))
        print("mask shape:", mask[0].shape)
        print(type(mask))
        # input = Image.fromarray(input)
        # input.save("your_file.jpeg")
        # input = Image.fromarray(input)
        # input.save("your_file.jpeg")
        # plt.subplot(121)
        # plt.imshow(input[0]/255)
        # plt.subplot(122)
        # plt.imshow(np.squeeze(mask[0]),cmap="gray")
        # plt.show()
        i+=1
        if i > num_batches:
            break

