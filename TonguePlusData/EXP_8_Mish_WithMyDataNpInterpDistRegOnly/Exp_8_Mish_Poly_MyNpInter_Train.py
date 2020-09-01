#that is fork from https://github.com/qqwweee/keras-yolo3
#the modifications are as follows:
# - backbone uses squeeze-and-excitation blocks and has fewer parameters
# - the neck and head is new, it has single output scale
# - it is extended by the bounding polygon functionality

from datetime import datetime
import colorsys
import os
import sys
from functools import reduce
from functools import wraps

import math
import random as rd
import cv2 as cv
import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, Callback
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Reshape, Dense, Permute, multiply, Activation, add, Lambda, concatenate, MaxPooling2D
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adadelta, Adagrad
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import multi_gpu_model
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from tensorflow.keras.utils import plot_model
from TonguePlusData.MyCustomCallbacks import TrainingPlotCallback, DeleteEarlySavedH5models
# os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\Graphviz2.38\\bin'
from tensorflow.python.keras.utils.data_utils import Sequence, get_file
from keras_preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import random
import pandas as pd
import numpy as np
from glob import glob
from tensorflow.keras.preprocessing import image as krs_image
import cv2



np.set_printoptions(precision=3, suppress=True)
MAX_VERTICES = 1000 #that allows the labels to have 1000 vertices per polygon at max. They are reduced for training
ANGLE_STEP  = 5 #that means Poly-YOLO will detect 360/15=24 vertices per polygon at max
# NUM_ANGLES3  = int(360 // ANGLE_STEP * 3) #72 = (360/15)*3
# print("NUM_ANGLES3:", NUM_ANGLES3)
NUM_ANGLES  = int(360 // ANGLE_STEP) # 24
print("NUM_ANGLES:", NUM_ANGLES)
# for mydatagenerator aug
rotation_range = 90
width_shift_range = 0.3
height_shift_range = 0.3
zoom_range = 0.2
shear_range=0.35
horizontal_flip=True
brightness_range=[0.5, 1.3]


grid_size_multiplier = 4 #that is resolution of the output scale compared with input. So it is 1/4
anchor_mask = [[0,1,2,3,4,5,6,7,8], [0,1,2,3,4,5,6,7,8], [0,1,2,3,4,5,6,7,8]] #that should be optimized
anchors_per_level = 9 #single scale and nine anchors
# for running the script
model_index =  sys.argv[1]

# def mish(x):
#     return x * tf.math.tanh(tf.math.softplus(x))
class Mish(Layer):
    '''
    Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Examples:
        >>> X_input = Input(input_shape)
        >>> X = Mish()(X_input)
    '''

    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))

    def get_config(self):
        config = super(Mish, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

# class MyGenerator(Sequence):
#     def __init__(self, annotation_lines, batch_size, input_shape, anchors, num_classes, is_random) :
#         self.annotation_lines = annotation_lines
#         self.batch_size = batch_size
#         self.input_shape = input_shape
#         self.anchors =  anchors
#         self.num_classes =  num_classes
#         self.is_random = is_random
#
#
#     def __len__(self) :
#         return (np.ceil(len(self.annotation_lines) / float(self.batch_size))).astype(np.int)
#
#
#     def __getitem__(self, idx) :
#         return self.data_gen()
#
#     def data_gen(self):
#         """data generator for fit_generator"""
#         n = len(self.annotation_lines)
#         i = 0
#         while True:
#             image_data = []
#             box_data = []
#             for b in range(self.batch_size):
#                 if i == 0:
#                     np.random.shuffle(self.annotation_lines)
#                 image, box = self.get_random_data(self.annotation_lines[i], self.input_shape, random=self.is_random)
#                 image_data.append(image)
#                 box_data.append(box)
#                 i = (i + 1) % n
#             image_data = np.array(image_data)
#             # print("image_data:", image_data.shape)
#             box_data = np.array(box_data)
#             y_true = preprocess_true_boxes(box_data, self.input_shape, self.anchors, self.num_classes)
#             return [image_data, *y_true], np.zeros(self.batch_size)
#
#     def get_random_data(self, line, input_shape, random=True, max_boxes=80, hue_alter=20, sat_alter=30, val_alter=30,
#                         proc_img=True):
#         # load data
#         # the color conversion is later. it is not necessary to realize bgr->rgb->hsv->rgb
#         # print("get_random_data line[0]:", line[0])
#         # print(os.getcwd())
#
#         image = cv.imread(line[0])
#         # print("get_random_data image:", image)
#
#         iw = image.shape[1]
#         ih = image.shape[0]
#         h, w = input_shape
#         box = np.array([np.array(list(map(float, box.split(','))))
#                         for box in line[1:]])
#         box_list = []
#         for box in line[1:]:
#             # print(box)
#             box_list.append(box.split(','))
#         print(box_list)
#         if not random:
#             # resize image
#             scale = min(w / iw, h / ih)
#             nw = int(iw * scale)
#             nh = int(ih * scale)
#             dx = (w - nw) // 2
#             dy = (h - nh) // 2
#             image_data = 0
#             if proc_img:
#                 # image = image.resize((nw, nh), Image.BICUBIC)
#                 image = cv.cvtColor(
#                     cv.resize(image, (nw, nh), interpolation=cv.INTER_CUBIC), cv.COLOR_BGR2RGB)
#                 image = Image.fromarray(image)
#                 new_image = Image.new('RGB', (w, h), (128, 128, 128))
#                 new_image.paste(image, (dx, dy))
#                 image_data = np.array(new_image) / 255.
#             # correct boxes
#             box_data = np.zeros((max_boxes, 5 + NUM_ANGLES3))
#             if len(box) > 0:
#                 np.random.shuffle(box)
#                 if len(box) > max_boxes:
#                     box = box[:max_boxes]
#                 box[:, [0, 2]] = box[:, [0, 2]] * scale + dx
#                 box[:, [1, 3]] = box[:, [1, 3]] * scale + dy
#                 box_data[:len(box), 0:5] = box[:, 0:5]
#                 for b in range(0, len(box)):
#                     for i in range(5, MAX_VERTICES * 2, 2):
#                         if box[b, i] == 0 and box[b, i + 1] == 0:
#                             continue
#                         box[b, i] = box[b, i] * scale + dx
#                         box[b, i + 1] = box[b, i + 1] * scale + dy
#
#                 # box_data[:, i:NUM_ANGLES3 + 5] = 0
#
#                 for i in range(0, len(box)):
#                     boxes_xy = (box[i, 0:2] + box[i, 2:4]) // 2
#
#                     for ver in range(5, MAX_VERTICES * 2, 2):
#                         if box[i, ver] == 0 and box[i, ver + 1] == 0:
#                             break
#                         dist_x = boxes_xy[0] - box[i, ver]
#                         dist_y = boxes_xy[1] - box[i, ver + 1]
#                         dist = np.sqrt(np.power(dist_x, 2) + np.power(dist_y, 2))
#                         if (dist < 1): dist = 1  # to avoid inf or nan in log in loss
#
#                         angle = np.degrees(np.arctan2(dist_y, dist_x))
#                         if (angle < 0): angle += 360
#                         iangle = int(angle) // ANGLE_STEP
#                         relative_angle = (angle - (iangle * int(ANGLE_STEP))) / ANGLE_STEP
#
#                         if dist > box_data[
#                             i, 5 + iangle * 3]:  # check for vertex existence. only the most distant is taken
#                             box_data[i, 5 + iangle * 3] = dist
#                             box_data[i, 5 + iangle * 3 + 1] = relative_angle
#                             box_data[
#                                 i, 5 + iangle * 3 + 2] = 1  # problbility  mask to be 1 for the exsitance of the vertex otherwise =0
#             return image_data, box_data
#
#         # resize image
#         random_scale = rd.uniform(.6, 1.4)
#         scale = min(w / iw, h / ih)
#         nw = int(iw * scale * random_scale)
#         nh = int(ih * scale * random_scale)
#
#         # force nw a nh to be an even
#         if (nw % 2) == 1:
#             nw = nw + 1
#         if (nh % 2) == 1:
#             nh = nh + 1
#
#         # jitter for slight distort of aspect ratio
#         if np.random.rand() < 0.3:
#             if np.random.rand() < 0.5:
#                 nw = int(nw * rd.uniform(.8, 1.0))
#             else:
#                 nh = int(nh * rd.uniform(.8, 1.0))
#
#         image = cv.resize(image, (nw, nh), interpolation=cv.INTER_CUBIC)
#         nwiw = nw / iw
#         nhih = nh / ih
#
#         # clahe. applied on resized image to save time. but before placing to avoid
#         # the influence of homogenous background
#         clahe = cv.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
#         lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
#         l, a, b = cv.split(lab)
#         cl = clahe.apply(l)
#         limg = cv.merge((cl, a, b))
#         image = cv.cvtColor(limg, cv.COLOR_LAB2BGR)
#
#         # place image
#         dx = rd.randint(0, max(w - nw, 0))
#         dy = rd.randint(0, max(h - nh, 0))
#
#         new_image = np.full((h, w, 3), 128, dtype='uint8')
#         new_image, crop_coords, new_img_coords = random_crop(
#             image, new_image)
#
#         # flip image or not
#         flip = rd.random() < .5
#         if flip:
#             new_image = cv.flip(new_image, 1)
#
#         # distort image
#         hsv = np.int32(cv.cvtColor(new_image, cv.COLOR_BGR2HSV))
#
#         # linear hsv distortion
#         hsv[..., 0] += rd.randint(-hue_alter, hue_alter)
#         hsv[..., 1] += rd.randint(-sat_alter, sat_alter)
#         hsv[..., 2] += rd.randint(-val_alter, val_alter)
#
#         # additional non-linear distortion of saturation and value
#         if np.random.rand() < 0.5:
#             hsv[..., 1] = hsv[..., 1] * rd.uniform(.7, 1.3)
#             hsv[..., 2] = hsv[..., 2] * rd.uniform(.7, 1.3)
#
#         hsv[..., 0][hsv[..., 0] > 179] = 179
#         hsv[..., 0][hsv[..., 0] < 0] = 0
#         hsv[..., 1][hsv[..., 1] > 255] = 255
#         hsv[..., 1][hsv[..., 1] < 0] = 0
#         hsv[..., 2][hsv[..., 2] > 255] = 255
#         hsv[..., 2][hsv[..., 2] < 0] = 0
#
#         image_data = cv.cvtColor(
#             np.uint8(hsv), cv.COLOR_HSV2RGB).astype('float32') / 255.0
#
#         # add noise
#         if np.random.rand() < 0.15:
#             image_data = np.clip(image_data + np.random.rand() *
#                                  image_data.std() * np.random.random(image_data.shape), 0, 1)
#
#         # correct boxes
#         box_data = np.zeros((max_boxes, 5 + NUM_ANGLES3))
#
#         if len(box) > 0:
#             np.random.shuffle(box)
#             # rescaling separately because 5-th element is class
#             box[:, [0, 2]] = box[:, [0, 2]] * nwiw  # for x
#             # rescale polygon vertices
#             box[:, 5::2] = box[:, 5::2] * nwiw
#             # rescale polygon vertices
#             box[:, [1, 3]] = box[:, [1, 3]] * nhih  # for y
#             box[:, 6::2] = box[:, 6::2] * nhih
#
#             # # mask out boxes that lies outside of croping window ## new commit deleted
#             # mask = (box[:, 1] >= crop_coords[0]) & (box[:, 3] < crop_coords[1]) & (
#             #     box[:, 0] >= crop_coords[2]) & (box[:, 2] < crop_coords[3])
#             # box = box[mask]
#
#             # transform boxes to new coordinate system w.r.t new_image
#             box[:, :2] = box[:, :2] - [crop_coords[2], crop_coords[0]] + [new_img_coords[2], new_img_coords[0]]
#             box[:, 2:4] = box[:, 2:4] - [crop_coords[2], crop_coords[0]] + [new_img_coords[2], new_img_coords[0]]
#             if flip:
#                 box[:, [0, 2]] = (w - 1) - box[:, [2, 0]]
#
#             box[:, 0:2][box[:, 0:2] < 0] = 0
#             box[:, 2][box[:, 2] >= w] = w - 1
#             box[:, 3][box[:, 3] >= h] = h - 1
#             box_w = box[:, 2] - box[:, 0]
#             box_h = box[:, 3] - box[:, 1]
#             box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
#
#             if len(box) > max_boxes:
#                 box = box[:max_boxes]
#
#             box_data[:len(box), 0:5] = box[:, 0:5]
#
#         # -------------------------------start polygon vertices processing-------------------------------#
#         for b in range(0, len(box)):
#             boxes_xy = (box[b, 0:2] + box[b, 2:4]) // 2
#             for i in range(5, MAX_VERTICES * 2, 2):
#                 if box[b, i] == 0 and box[b, i + 1] == 0:
#                     break
#                 box[b, i:i + 2] = box[b, i:i + 2] - [crop_coords[2], crop_coords[0]] + [new_img_coords[2],
#                                                                                         new_img_coords[0]]  # transform
#                 if flip: box[b, i] = (w - 1) - box[b, i]
#                 dist_x = boxes_xy[0] - box[b, i]
#                 dist_y = boxes_xy[1] - box[b, i + 1]
#                 dist = np.sqrt(np.power(dist_x, 2) + np.power(dist_y, 2))
#                 if (dist < 1): dist = 1
#
#                 angle = np.degrees(np.arctan2(dist_y, dist_x))
#                 if (angle < 0): angle += 360
#                 # num of section it belongs to
#                 iangle = int(angle) // ANGLE_STEP
#
#                 if iangle >= NUM_ANGLES: iangle = NUM_ANGLES - 1
#
#                 if dist > box_data[b, 5 + iangle * 3]:  # check for vertex existence. only the most distant is taken
#                     box_data[b, 5 + iangle * 3] = dist
#                     box_data[b, 5 + iangle * 3 + 1] = (angle - (
#                                 iangle * int(ANGLE_STEP))) / ANGLE_STEP  # relative angle
#                     box_data[b, 5 + iangle * 3 + 2] = 1
#         # ---------------------------------end polygon vertices processing-------------------------------#
#         return image_data, box_data
#
#     def preprocess_true_boxes(self, true_boxes, input_shape, anchors, num_classes):
#         '''Preprocess true boxes to training input format
#
#         Parameters
#         ----------
#         true_boxes: array, shape=(m, T, 5+69)
#             Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape
#         input_shape: array-like, hw, multiples of 32
#         anchors: array, shape=(N, 2), wh
#         num_classes: integer
#
#         Returns
#         -------
#         y_true: list of array, shape like yolo_outputs, xywh are reletive value
#
#         '''
#         assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
#         true_boxes = np.array(true_boxes, dtype='float32')
#         input_shape = np.array(input_shape, dtype='int32')
#         boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
#         boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
#
#         true_boxes[:, :, 5:NUM_ANGLES3 + 5:3] /= np.clip(
#             np.expand_dims(np.sqrt(np.power(boxes_wh[:, :, 0], 2) + np.power(boxes_wh[:, :, 1], 2)), -1), 0.0001,
#             9999999)
#         true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
#         true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]
#
#         m = true_boxes.shape[0]
#         grid_shapes = [input_shape // {0: grid_size_multiplier}[l] for l in range(1)]
#         y_true = [
#             np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes + NUM_ANGLES3),
#                      dtype='float32') for l in range(1)]
#
#         # Expand dim to apply broadcasting.
#         anchors = np.expand_dims(anchors, 0)
#         anchor_maxes = anchors / 2.
#         anchor_mins = -anchor_maxes
#         valid_mask = boxes_wh[..., 0] > 0
#
#         for b in range(m):
#             # Discard zero rows.
#             wh = boxes_wh[b, valid_mask[b]]
#             if len(wh) == 0: continue
#             # Expand dim to apply broadcasting.
#             wh = np.expand_dims(wh, -2)
#             box_maxes = wh / 2.
#             box_mins = -box_maxes
#
#             intersect_mins = np.maximum(box_mins, anchor_mins)
#             intersect_maxes = np.minimum(box_maxes, anchor_maxes)
#             intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
#             intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
#             box_area = wh[..., 0] * wh[..., 1]
#             anchor_area = anchors[..., 0] * anchors[..., 1]
#             iou = intersect_area / (box_area + anchor_area - intersect_area)
#
#             # Find best anchor for each true box
#             best_anchor = np.argmax(iou, axis=-1)
#             for t, n in enumerate(best_anchor):
#                 l = 0
#                 if n in anchor_mask[l]:
#                     i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
#                     j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
#                     k = anchor_mask[l].index(n)
#                     c = true_boxes[b, t, 4].astype('int32')
#
#                     y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
#                     y_true[l][b, j, i, k, 4] = 1
#                     y_true[l][b, j, i, k, 5 + c] = 1
#                     y_true[l][b, j, i, k, 5 + num_classes:5 + num_classes + NUM_ANGLES3] = true_boxes[b, t,
#                                                                                            5: 5 + NUM_ANGLES3]
#         return y_true

def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw = image.shape[1]
    ih = image.shape[0]
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    cvi = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    cvi = cv.resize(cvi, (nw, nh), interpolation=cv.INTER_CUBIC)
    dx = int((w - nw) // 2)
    dy = int((h - nh) // 2)
    new_image = np.zeros((h, w, 3), dtype='uint8')
    new_image[...] = 128
    if nw <= w and nh <= h:
        new_image[dy:dy + nh, dx:dx + nw, :] = cvi
    else:
        new_image = cvi[-dy:-dy + h, -dx:-dx + w, :]

    return new_image.astype('float32') / 255.0


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


# def get_random_data(line, input_shape, random=True, max_boxes=80, hue_alter=20, sat_alter=30, val_alter=30, proc_img=True):
#     # load data
#     # the color conversion is later. it is not necessary to realize bgr->rgb->hsv->rgb
#     # print("get_random_data line[0]:", line[0])
#     # print(os.getcwd())
#
#     image = cv.imread(line[0])
#     # print("get_random_data image:", image)
#
#     iw = image.shape[1]
#     ih = image.shape[0]
#     h, w = input_shape
#     box = np.array([np.array(list(map(float, box.split(','))))
#                     for box in line[1:]])
#
#     if not random:
#         # resize image
#         scale = min(w / iw, h / ih)
#         nw = int(iw * scale)
#         nh = int(ih * scale)
#         dx = (w - nw) // 2
#         dy = (h - nh) // 2
#         image_data = 0
#         if proc_img:
#             # image = image.resize((nw, nh), Image.BICUBIC)
#             image = cv.cvtColor(
#                 cv.resize(image, (nw, nh), interpolation=cv.INTER_CUBIC), cv.COLOR_BGR2RGB)
#             image = Image.fromarray(image)
#             new_image = Image.new('RGB', (w, h), (128, 128, 128))
#             new_image.paste(image, (dx, dy))
#             image_data = np.array(new_image) / 255.
#         # correct boxes
#         box_data = np.zeros((max_boxes, 5 + NUM_ANGLES3))
#         if len(box) > 0:
#             np.random.shuffle(box)
#             if len(box) > max_boxes:
#                 box = box[:max_boxes]
#             box[:, [0, 2]] = box[:, [0, 2]] * scale + dx
#             box[:, [1, 3]] = box[:, [1, 3]] * scale + dy
#             box_data[:len(box), 0:5] = box[:, 0:5]
#             for b in range(0, len(box)):
#                 for i in range(5, MAX_VERTICES * 2, 2):
#                     if box[b,i] == 0 and box[b, i + 1] == 0:
#                         continue
#                     box[b, i] = box[b, i] * scale + dx
#                     box[b, i + 1] = box[b, i + 1] * scale + dy
#
#             box_data[:, i:NUM_ANGLES3 + 5] = 0
#
#             for i in range(0, len(box)):
#                 boxes_xy = (box[i, 0:2] + box[i, 2:4]) // 2
#
#                 for ver in range(5, MAX_VERTICES * 2, 2):
#                     if box[i, ver] == 0 and box[i, ver + 1] == 0:
#                         break
#                     dist_x = boxes_xy[0] - box[i, ver]
#                     dist_y = boxes_xy[1] - box[i, ver + 1]
#                     dist = np.sqrt(np.power(dist_x, 2) + np.power(dist_y, 2))
#                     if (dist < 1): dist = 1 #to avoid inf or nan in log in loss
#
#                     angle = np.degrees(np.arctan2(dist_y, dist_x))
#                     if (angle < 0): angle += 360
#                     iangle = int(angle) // ANGLE_STEP
#                     relative_angle = (angle - (iangle * int(ANGLE_STEP))) / ANGLE_STEP
#
#                     if dist > box_data[i, 5 + iangle * 3]:  # check for vertex existence. only the most distant is taken
#                         box_data[i, 5 + iangle * 3] = dist
#                         box_data[i, 5 + iangle * 3 + 1] = relative_angle
#                         box_data[i, 5 + iangle * 3 + 2] = 1 # problbility  mask to be 1 for the exsitance of the vertex otherwise =0
#         return image_data, box_data
#
#
#     # resize image
#     random_scale = rd.uniform(.6, 1.4)
#     scale = min(w / iw, h / ih)
#     nw = int(iw * scale * random_scale)
#     nh = int(ih * scale * random_scale)
#
#     # force nw a nh to be an even
#     if (nw % 2) == 1:
#         nw = nw + 1
#     if (nh % 2) == 1:
#         nh = nh + 1
#
#     # jitter for slight distort of aspect ratio
#     if np.random.rand() < 0.3:
#         if np.random.rand() < 0.5:
#             nw = int(nw*rd.uniform(.8, 1.0))
#         else:
#             nh = int(nh*rd.uniform(.8, 1.0))
#
#     image = cv.resize(image, (nw, nh), interpolation=cv.INTER_CUBIC)
#     nwiw = nw/iw
#     nhih = nh/ih
#
#     # clahe. applied on resized image to save time. but before placing to avoid
#     # the influence of homogenous background
#     clahe = cv.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
#     lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
#     l, a, b = cv.split(lab)
#     cl = clahe.apply(l)
#     limg = cv.merge((cl, a, b))
#     image = cv.cvtColor(limg, cv.COLOR_LAB2BGR)
#
#     # place image
#     dx = rd.randint(0, max(w - nw, 0))
#     dy = rd.randint(0, max(h - nh, 0))
#
#     new_image = np.full((h, w, 3), 128, dtype='uint8')
#     new_image, crop_coords, new_img_coords = random_crop(
#         image, new_image)
#
#     # flip image or not
#     flip = rd.random() < .5
#     if flip:
#         new_image = cv.flip(new_image, 1)
#
#     # distort image
#     hsv = np.int32(cv.cvtColor(new_image, cv.COLOR_BGR2HSV))
#
#     # linear hsv distortion
#     hsv[..., 0] += rd.randint(-hue_alter, hue_alter)
#     hsv[..., 1] += rd.randint(-sat_alter, sat_alter)
#     hsv[..., 2] += rd.randint(-val_alter, val_alter)
#
#     # additional non-linear distortion of saturation and value
#     if np.random.rand() < 0.5:
#         hsv[..., 1] = hsv[..., 1]*rd.uniform(.7, 1.3)
#         hsv[..., 2] = hsv[..., 2]*rd.uniform(.7, 1.3)
#
#     hsv[..., 0][hsv[..., 0] > 179] = 179
#     hsv[..., 0][hsv[..., 0] < 0] = 0
#     hsv[..., 1][hsv[..., 1] > 255] = 255
#     hsv[..., 1][hsv[..., 1] < 0] = 0
#     hsv[..., 2][hsv[..., 2] > 255] = 255
#     hsv[..., 2][hsv[..., 2] < 0] = 0
#
#     image_data = cv.cvtColor(
#         np.uint8(hsv), cv.COLOR_HSV2RGB).astype('float32') / 255.0
#
#     # add noise
#     if np.random.rand() < 0.15:
#         image_data = np.clip(image_data + np.random.rand() *
#                              image_data.std() * np.random.random(image_data.shape), 0, 1)
#
#     # correct boxes
#     box_data = np.zeros((max_boxes, 5 + NUM_ANGLES3))
#
#     if len(box) > 0:
#         np.random.shuffle(box)
#         # rescaling separately because 5-th element is class
#         box[:, [0, 2]] = box[:, [0, 2]] * nwiw  # for x
#         # rescale polygon vertices
#         box[:, 5::2] = box[:, 5::2] * nwiw
#         # rescale polygon vertices
#         box[:, [1, 3]] = box[:, [1, 3]] * nhih  # for y
#         box[:, 6::2] = box[:, 6::2] * nhih
#
#         # # mask out boxes that lies outside of croping window ## new commit deleted
#         # mask = (box[:, 1] >= crop_coords[0]) & (box[:, 3] < crop_coords[1]) & (
#         #     box[:, 0] >= crop_coords[2]) & (box[:, 2] < crop_coords[3])
#         # box = box[mask]
#
#         # transform boxes to new coordinate system w.r.t new_image
#         box[:, :2] = box[:, :2] - [crop_coords[2], crop_coords[0]] + [new_img_coords[2], new_img_coords[0]]
#         box[:, 2:4] = box[:, 2:4] - [crop_coords[2], crop_coords[0]] + [new_img_coords[2], new_img_coords[0]]
#         if flip:
#             box[:, [0, 2]] = (w-1) - box[:, [2, 0]]
#
#         box[:, 0:2][box[:, 0:2] < 0] = 0
#         box[:, 2][box[:, 2] >= w] = w-1
#         box[:, 3][box[:, 3] >= h] = h-1
#         box_w = box[:, 2] - box[:, 0]
#         box_h = box[:, 3] - box[:, 1]
#         box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
#
#         if len(box) > max_boxes:
#             box = box[:max_boxes]
#
#         box_data[:len(box), 0:5] = box[:, 0:5]
#
#     #-------------------------------start polygon vertices processing-------------------------------#
#     for b in range(0, len(box)):
#         boxes_xy = (box[b, 0:2] + box[b, 2:4]) // 2
#         for i in range(5, MAX_VERTICES * 2, 2):
#             if box[b, i] == 0 and box[b, i + 1] == 0:
#                 break
#             box[b, i:i+2] = box[b, i:i+2] - [crop_coords[2], crop_coords[0]] + [new_img_coords[2], new_img_coords[0]] # transform
#             if flip: box[b, i] = (w - 1) - box[b, i]
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
#             if iangle>=NUM_ANGLES: iangle = NUM_ANGLES-1
#
#             if dist > box_data[b, 5 + iangle * 3]: # check for vertex existence. only the most distant is taken
#                 box_data[b, 5 + iangle * 3]     = dist
#                 box_data[b, 5 + iangle * 3 + 1] = (angle - (iangle * int(ANGLE_STEP))) / ANGLE_STEP #relative angle
#                 box_data[b, 5 + iangle * 3 + 2] = 1
#     #---------------------------------end polygon vertices processing-------------------------------#
#     return image_data, box_data


def random_crop(img, new_img):
    """Creates random crop from img and insert it into new_img

    Args:
        img (numpy array): Image to be cropped
        new_img (numpy array): Image to which the crop will be inserted into.

    Returns:
        tuple: Tuple of image containing the crop, list of coordinates used to crop img and list of coordinates where the crop
        has been inserted into in new_img
    """
    h, w = img.shape[:2]
    crop_shape = new_img.shape[:2]
    crop_coords = [0, 0, 0, 0]
    new_pos = [0, 0, 0, 0]
    # if image height is smaller than cropping window
    if h < crop_shape[0]:
        # cropping whole image [0,h]
        crop_coords[1] = h
        # randomly position whole img along height dimension
        val = rd.randint(0, crop_shape[0]-h)
        new_pos[0:2] = [val, val + h]
    else:
        # if image height is bigger than cropping window
        # randomly position cropping window on image
        crop_h_shift = rd.randint(crop_shape[0], h)
        crop_coords[0:2] = [crop_h_shift - crop_shape[0], crop_h_shift]
        new_pos[0:2] = [0, crop_shape[0]]

    # same as above for image width
    if w < crop_shape[1]:
        crop_coords[3] = w
        val = rd.randint(0, crop_shape[1] - w)
        new_pos[2:4] = [val, val + w]
    else:
        crop_w_shift = rd.randint(crop_shape[1], w)
        crop_coords[2:4] = [crop_w_shift - crop_shape[1], crop_w_shift]
        new_pos[2:4] = [0, crop_shape[1]]

    # slice, insert and return image including crop and coordinates used for cropping and inserting
    # coordinates are later used for boxes adjustments.
    new_img[new_pos[0]:new_pos[1], new_pos[2]:new_pos[3],
            :] = img[crop_coords[0]:crop_coords[1], crop_coords[2]:crop_coords[3], :]
    return new_img, crop_coords, new_pos


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


# def DarknetConv2D_BN_Leaky(*args, **kwargs):
#     """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
#     no_bias_kwargs = {'use_bias': False}
#     no_bias_kwargs.update(kwargs)
#     return compose(
#         DarknetConv2D(*args, **no_bias_kwargs),
#         BatchNormalization(),
#         LeakyReLU(alpha=0.1))

def DarknetConv2D_BN_Mish(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        Mish())


def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Mish(num_filters, (3, 3), strides=(2, 2))(x)
    for i in range(num_blocks):
        y = compose(
            DarknetConv2D_BN_Mish(num_filters // 2, (1, 1)),
            DarknetConv2D_BN_Mish(num_filters, (3, 3)))(x)
        y = squeeze_excite_block(y)
        x = Add()([x, y])
    return x


#taken from https://github.com/titu1994/keras-squeeze-excite-network/blob/master/keras_squeeze_excite_network/se_resnet.py
def squeeze_excite_block(tensor, ratio=16):
    init = tensor
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, kernel_initializer='he_normal', use_bias=False)(se)
    se = Mish()(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x


def _tensor_shape(tensor):
    return getattr(tensor, '_keras_shape')



def darknet_body(x):
    '''Darknent body having 52 Convolution2D layers'''
    base = 6  # YOLOv3 has base=8, we have less parameters
    x = DarknetConv2D_BN_Mish(base * 4, (3, 3))(x)
    x = resblock_body(x, base * 8, 1)
    x = resblock_body(x, base * 16, 2)
    tiny = x
    x = resblock_body(x, base * 32, 8)
    small = x
    x = resblock_body(x, base * 64, 8)
    medium = x
    x = resblock_body(x, base * 128, 8)
    big = x
    return tiny, small, medium, big



def yolo_body(inputs, num_anchors, num_classes):
    """Create Poly-YOLO model CNN body in Keras."""
    tiny, small, medium, big = darknet_body(inputs)

    base = 6
    tiny   = DarknetConv2D_BN_Mish(base*32, (1, 1))(tiny)
    small  = DarknetConv2D_BN_Mish(base*32, (1, 1))(small)
    medium = DarknetConv2D_BN_Mish(base*32, (1, 1))(medium)
    big    = DarknetConv2D_BN_Mish(base*32, (1, 1))(big)

    #stairstep upsamplig
    all = Add()([medium, UpSampling2D(2,interpolation='bilinear')(big)])
    all = Add()([small, UpSampling2D(2,interpolation='bilinear')(all)])
    all = Add()([tiny, UpSampling2D(2,interpolation='bilinear')(all)])



    num_filters = base*32

    x = compose(
        DarknetConv2D_BN_Mish(num_filters, (1, 1)),
        DarknetConv2D_BN_Mish(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Mish(num_filters, (1, 1)))(all)
    print("x.shape:", x.shape)
    print()
    print("num_classes:", num_classes)
    print("NUM_ANGLES:", NUM_ANGLES)
    print("num_anchors:", num_anchors)
    print()
    all = compose(
        DarknetConv2D_BN_Mish(num_filters * 2, (3, 3)),

        DarknetConv2D(num_anchors * (num_classes + 5 + NUM_ANGLES), (1, 1)))(x)
    print("all.shape:", all.shape)
    return Model(inputs, all)



def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    print("feats in yolo head:", feats)
    # feats in yolo head: Tensor("conv2d_68/Identity:0", shape=(None, None, None, 702), dtype=float32)
    num_anchors = anchors_per_level
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])  # provided reshaped for the output of feature map

    grid_shape = K.shape(feats)[1:3]  # height, width
    grid_y = K.tile(tf.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1], name='yolo_head/tile/reshape/grid_y'),
                    [1, grid_shape[1], 1, 1])
    grid_x = K.tile(tf.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1], name='yolo_head/tile/reshape/grid_x'),
                    [grid_shape[0], 1, 1, 1])
    grid = tf.concat([grid_x, grid_y], axis=-1, name='yolo_head/concatenate/grid')
    print("grid.shape:", grid)
    grid = K.cast(grid, K.dtype(feats))
    feats = tf.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5 + NUM_ANGLES], name='yolo_head/reshape/feats')
    print("reshaped predictions:", feats)
    # Adjust predictions to each spatial grid point and anchor size.
    print("grid_shape[...,::-1]", grid_shape[...,::-1])
    # left-up corner +  grid possition = the center cordinates.
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[...,::-1], K.dtype(feats))
    print("box_xy from prediction +grid:", box_xy)
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[...,::-1], K.dtype(feats))

    box_confidence      = K.sigmoid(feats[..., 4:5])
    box_class_probs     = K.sigmoid(feats[..., 5:5 + num_classes])
    # polygons_confidence = K.sigmoid(feats[..., 5+num_classes+2:5+num_classes+NUM_ANGLES3:3])
    # print("polygons_confidence:", polygons_confidence)

    polygons_dist = K.exp(feats[..., 5 + num_classes:num_classes + 5 + NUM_ANGLES]) # distance
    print("polygons_dist:", polygons_dist) # shape=(None, None, None, 9, 24) , distance predictions from vertix to center (px)? (px, py)

    dx = K.square(anchors_tensor[..., 0:1] / 2)
    dy = K.square(anchors_tensor[..., 1:2] / 2)
    d = K.cast(K.sqrt(dx + dy), K.dtype(polygons_dist)) # provided anchor diagonal as the type of our prediction of dists
    print("input_shape[::-1]:", input_shape[::-1])
    a = K.pow(input_shape[::-1], 2)  # elementwise exponential of feature maps' size? (h^2, w^2)
    a = K.cast(a, K.dtype(feats))
    b= K.sum(a) # b = h^2 + w^2  # diagonal of length fo feature map
    diagonal =  K.cast(K.sqrt(b), K.dtype(feats)) # calculate the diagonal distance according to the feature size
    polygons_dist = polygons_dist * d / diagonal  #  normalize the anchor diagonal/ feature  map diagnal


    # # polar position (px, py), the following is py which is the angle prediction
    # polygons_y = feats[..., 5 + num_classes + 1:num_classes + 5 + NUM_ANGLES3:3]
    # polygons_y = K.sigmoid(polygons_y)

    if calc_loss == True: # for calculating the loss use this other wise output the output
        return grid, feats, box_xy, box_wh
    # return box_xy, box_wh, box_confidence, box_class_probs, polygons_dist, polygons_y, polygons_confidence
    return box_xy, box_wh, box_confidence, box_class_probs, polygons_dist

def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''Get corrected boxes (xmin, ymin, xmax, ymax) to (ymax, xmax,ymin, xmin)'''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def yolo_correct_polygons(polygons_x, boxes, input_shape, image_shape):
    polygons = K.concatenate([polygons_x])
    return polygons


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    box_xy, box_wh, box_confidence, box_class_probs, polygons_dist = yolo_head(feats, anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    polygons = yolo_correct_polygons(polygons_dist, boxes, input_shape, image_shape)
    polygons = K.reshape(polygons, [-1, NUM_ANGLES]) # [dis1, dist2, dist 3...]
    return boxes, box_scores, polygons


def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=80,
              score_threshold=.5,
              iou_threshold=.5):
    """Evaluate YOLO model on given input and return filtered boxes."""
    input_shape = K.shape(yolo_outputs)[1:3] * grid_size_multiplier
    boxes = []
    box_scores = []
    polygons = []

    for l in range(1):
        _boxes, _box_scores, _polygons = yolo_boxes_and_scores(yolo_outputs,
                                                               anchors[anchor_mask[l]], num_classes, input_shape,
                                                               image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
        polygons.append(_polygons)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)
    polygons = K.concatenate(polygons, axis=0)

    mask = box_scores >= score_threshold
    # box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    polygons_ = []
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_polygons = tf.boolean_mask(polygons, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        class_polygons = K.gather(class_polygons, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
        polygons_.append(class_polygons)
    polygons_ = K.concatenate(polygons_, axis=0)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_, polygons_


# def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
#     '''Preprocess true boxes to training input format
#
#     Parameters
#     ----------
#     true_boxes: array, shape=(m, T, 5+69)
#         Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape
#     input_shape: array-like, hw, multiples of 32
#     anchors: array, shape=(N, 2), wh
#     num_classes: integer
#
#     Returns
#     -------
#     y_true: list of array, shape like yolo_outputs, xywh are reletive value
#
#     '''
#     assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
#     true_boxes = np.array(true_boxes, dtype='float32')
#     input_shape = np.array(input_shape, dtype='int32')
#     boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
#     boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
#
#     true_boxes[:,:, 5:NUM_ANGLES3 + 5:3] /= np.clip(np.expand_dims(np.sqrt(np.power(boxes_wh[:, :, 0], 2) + np.power(boxes_wh[:, :, 1], 2)), -1), 0.0001, 9999999)
#     true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
#     true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]
#
#     m = true_boxes.shape[0]
#     grid_shapes = [input_shape // {0: grid_size_multiplier}[l] for l in range(1)]
#     y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes + NUM_ANGLES3),
#                        dtype='float32') for l in range(1)]
#
#
#     # Expand dim to apply broadcasting.
#     anchors = np.expand_dims(anchors, 0)
#     anchor_maxes = anchors / 2.
#     anchor_mins = -anchor_maxes
#     valid_mask = boxes_wh[..., 0] > 0
#
#
#     for b in range(m):
#         # Discard zero rows.
#         wh = boxes_wh[b, valid_mask[b]]
#         if len(wh) == 0: continue
#         # Expand dim to apply broadcasting.
#         wh = np.expand_dims(wh, -2)
#         box_maxes = wh / 2.
#         box_mins = -box_maxes
#
#         intersect_mins = np.maximum(box_mins, anchor_mins)
#         intersect_maxes = np.minimum(box_maxes, anchor_maxes)
#         intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
#         intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
#         box_area = wh[..., 0] * wh[..., 1]
#         anchor_area = anchors[..., 0] * anchors[..., 1]
#         iou = intersect_area / (box_area + anchor_area - intersect_area)
#
#         # Find best anchor for each true box
#         best_anchor = np.argmax(iou, axis=-1)
#         for t, n in enumerate(best_anchor): # search for best anchor w,h size in the anchor mask setting
#             l = 0
#             if n in anchor_mask[l]:
#                 i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')# for each image b , and each box t
#                 j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
#                 k = anchor_mask[l].index(n)
#                 c = true_boxes[b, t, 4].astype('int32')  # class number e.g 10 classses = [0 ,1, 2,3,4,...9], in our case ,one class only , c=0
#
#                 y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4] # set as raw  xmin, ymin , xmax ymax
#                 y_true[l][b, j, i, k, 4] = 1 # F/B label
#                 y_true[l][b, j, i, k, 5 + c] = 1  # class label as 1 as one-hot for class score. e.g. if 3 class, c =2 , then label in [5: num_classes] into  [0, 0 ,1]
#                 y_true[l][b, j, i, k, 5 + num_classes:5 + num_classes + NUM_ANGLES] = true_boxes[b, t, 5: 5 + NUM_ANGLES]
#     return y_true

def my_preprocess_true_boxes_NPinterp(true_boxes, input_shape, anchors, num_classes):
    '''Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5+69)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    '''
    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]

    true_boxes[:,:, 5:NUM_ANGLES + 5] /= np.clip(np.expand_dims(np.sqrt(np.power(boxes_wh[:, :, 0], 2) + np.power(boxes_wh[:, :, 1], 2)), -1), 0.0001, 9999999)
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    m = true_boxes.shape[0]
    grid_shapes = [input_shape // {0: grid_size_multiplier}[l] for l in range(1)]
    y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes + NUM_ANGLES),
                       dtype='float32') for l in range(1)]


    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0


    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0: continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)
        for t, n in enumerate(best_anchor): # search for best anchor w,h size in the anchor mask setting
            l = 0
            if n in anchor_mask[l]:
                i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')# for each image b , and each box t
                j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                k = anchor_mask[l].index(n)
                c = true_boxes[b, t, 4].astype('int32')  # class number e.g 10 classses = [0 ,1, 2,3,4,...9], in our case ,one class only , c=0

                y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4] # set as raw  xmin, ymin , xmax ymax
                y_true[l][b, j, i, k, 4] = 1 # F/B label
                y_true[l][b, j, i, k, 5 + c] = 1  # class label as 1 as one-hot for class score. e.g. if 3 class, c =2 , then label in [5: num_classes] into  [0, 0 ,1]
                y_true[l][b, j, i, k, 5 + num_classes:5 + num_classes + NUM_ANGLES] = true_boxes[b, t, 5: 5 + NUM_ANGLES]
    return y_true


def box_iou(b1, b2):
    """Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    """

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def yolo_loss(args, anchors, num_classes, ignore_thresh=.5):
    """Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    """
    num_layers = 1
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]
    print("yolo_outputs.shape", yolo_outputs)
    print("y_true.shape", y_true)
    g_y_true = y_true
    # define a input shape as a 2d shape= (imgH,imgW)*grid_size_multiplier = feature map shape  * 4 = (rawiput W, raw input H)
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * grid_size_multiplier, K.dtype(y_true[0]))
    print("input_shape in loss", input_shape)
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
    loss = 0

    m = K.shape(yolo_outputs[0])[0]  # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))
    for layer in range(num_layers):
        object_mask = y_true[layer][..., 4:5] ## [..., 4:5] = [:, 4:5],  = y_true[..., 4] get the 4th element for all anchors in each pixel position of the feature map
        print("object_mask:", object_mask)
        # slice as seq[start:end:step] ==> [8, 78, step= 3], this is vertices sections
        # remember NUM_ANGLES3  = int(360 // ANGLE_STEP * 3) =72
        # recap: our output contains (4 boundingboxes, num_classes, polarangle, polardistance, problbilies for each section)
        # print("5 + num_classes + 2:", 5 + num_classes + 2)
        print("5 + num_classes + NUM_ANGLES:", 5 + num_classes + NUM_ANGLES)
        # vertices_mask = y_true[layer][..., 5 + num_classes + 2:5 + num_classes + NUM_ANGLES]  # problabilies for each section
        # print("vertices_mask:", vertices_mask)
        true_class_probs = y_true[layer][..., 5:5 + num_classes]
        print("true_class_probs:", true_class_probs)
        # return yolo_head: grid, feats, box_xy, box_wh
        grid, raw_pred, pred_xy, pred_wh= yolo_head(yolo_outputs[layer], anchors[anchor_mask[layer]], num_classes, input_shape, calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])
        print("pred_box:", pred_box) # x,y , w, h for 9 anchors , with shape (None, None, None, 9, 4)
        # found the left-up corner, which may be normalized with ground truth , since y_true[..., 2:] * Feature shapes
        raw_true_xy = y_true[layer][..., :2] * grid_shapes[layer][::-1] - grid
        # (px(distance), py(angle), pz(probability) ), should have 72 elements for all feature positions
        raw_true_polygon_distnace = y_true[layer][..., 5 + num_classes: 5 + num_classes + NUM_ANGLES]
        print("raw_true_polygon_distnace:", raw_true_polygon_distnace)
        # anchor_mask = = [[0,1,2,3,4,5,6,7,8], [0,1,2,3,4,5,6,7,8], [0,1,2,3,4,5,6,7,8]]
        # anchors[anchor_mask[layer]]
        # anchors = with shape [num, 2] ==>[9,2]
        # anchors[anchor_mask[layer]] --> just choose all anchors
        print("y_true[layer][..., 2:4]：", y_true[layer][..., 2:4]) # shape=(None, 104, 208, 9, 2)
        # each ground truth w, h, find the value corresponding to our direct prediction deviation see one note calculation GM slides for 7.31 day
        raw_true_wh = K.log(y_true[layer][..., 2:4] / anchors[anchor_mask[layer]] * input_shape[::-1])
        print("raw_true_wh",raw_true_wh) # shape=(None, 104, 208, 9, 2)
        #  object_mask is the condidence score for the bbox
        # k.switch to switch to different function accoding to the condience score.
        # if the object_mask == False, generating all zeros for the same shape as raw_true_wh
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))  # avoid log(0)=-inf

        raw_true_polygon_dist = raw_true_polygon_distnace # ground truth, px,
        # raw_true_polygon_y = raw_true_polygon_distnace[..., 1::3] # ground truth, py,

        dx = K.square(anchors[anchor_mask[layer]][..., 0:1] / 2)  # get gt anchor x
        dy = K.square(anchors[anchor_mask[layer]][..., 1:2] / 2)  # get gt anchor y
        d = K.cast(K.sqrt(dx + dy), K.dtype(raw_true_polygon_dist))  #  get gt d

        diagonal = K.sqrt(K.pow(input_shape[::-1][0], 2) + K.pow(input_shape[::-1][1], 2)) # get diagnoal of feature maps
        raw_true_polygon_dist = K.log(raw_true_polygon_dist / d * diagonal)  # the deviation of distances
        # the same as bounding boxes , according to the mask probbilities to calculate  distances from vertice to center

        # # if probilits all false, then give 0 distances
        # raw_true_polygon_x = K.switch(vertices_mask, raw_true_polygon_x, K.zeros_like(raw_true_polygon_x))

        box_loss_scale = 2 - y_true[layer][..., 2:3] * y_true[layer][..., 3:4] # weighted factor, changed with prediction of w, h
        # if w, h ground truth is small , then more loss weighted

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool') # acoording to the confidence score
        # (0 prediction , will be ignored, only consider non-zero postion, of ground truth).
        # only consider the positive anchors.

        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[layer][b, ..., 0:4], object_mask_bool[b, ..., 0])
            iou = box_iou(pred_box[b], true_box)   # calculate IoU
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou < ignore_thresh, K.dtype(true_box)))
            return b + 1, ignore_mask
        # iterate over batches calculate ignore_mask
        _, ignore_mask = tf.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack() # default axis = 0 # inorder to stack along the batch dimension
        ignore_mask = K.expand_dims(ignore_mask, -1) # in order to concate last dimension

        # K.binary_crossentropy is helpful to avoid exp overflow.
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2], from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh - raw_pred[..., 2:4])
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) + (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[..., 4:5],
                                                                                                                                                             from_logits=True) * ignore_mask
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 5:5 + num_classes], from_logits=True)
        polygon_loss_dist = object_mask  * box_loss_scale * 0.5 * K.square(raw_true_polygon_dist - raw_pred[..., 5 + num_classes:5 + num_classes + NUM_ANGLES])
        # polygon_loss_y = object_mask * vertices_mask * box_loss_scale * K.binary_crossentropy(raw_true_polygon_y, raw_pred[..., 5 + num_classes + 1:5 + num_classes + NUM_ANGLES3:3], from_logits=True)
        # vertices_confidence_loss = object_mask * K.binary_crossentropy(vertices_mask, raw_pred[..., 5 + num_classes + 2:5 + num_classes + NUM_ANGLES3:3], from_logits=True)

        print("finished losses")
        xy_loss = K.sum(xy_loss) / mf # get the mean loss for each image
        wh_loss = K.sum(wh_loss) / mf
        class_loss = K.sum(class_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        # vertices_confidence_loss = K.sum(vertices_confidence_loss) / mf
        # polygon_loss = K.sum(polygon_loss_x) / mf + K.sum(polygon_loss_y) / mf
        polygon_loss = K.sum(polygon_loss_dist) / mf
        print("finised losses for each image")
        # there is a weight for special masks losses and also weighted focal according to the confidences score in total for each image ,then * for the enirebatch
        loss += (xy_loss + wh_loss + confidence_loss + class_loss + 0.2 * polygon_loss)/ (K.sum(object_mask) + 1)*mf
    return loss


class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo.h5',
        "anchors_path": 'yolo_anchors.txt',
        "classes_path": 'yolo_classes.txt',
        "score": 0.2,
        "iou": 0.4,
        "model_image_size": (256, 256),
        "gpu_num": 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)  # set up default values
        self.__dict__.update(kwargs)  # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes, self.polygons = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = yolo_body(Input(shape=(None, None, 3)), anchors_per_level, num_classes)
            self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
        else:
            # novy output
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5 + NUM_ANGLES), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        if self.gpu_num >= 2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes, polygons = yolo_eval(self.yolo_model.output, self.anchors,
                                                     len(self.class_names), self.input_image_shape,
                                                     score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes, polygons

    def detect_image(self, image, raw_shape):
        # if self.model_image_size != (None, None):
        #     assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
        #     assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
        #     boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        # else:
        #     print('THE functionality is not implemented!')

        #
        # image_data = np.expand_dims(boxed_image, 0)  # Add batch dimension.
        out_boxes, out_scores, out_classes, polygons = self.sess.run(
            [self.boxes, self.scores, self.classes, self.polygons],
            feed_dict={
                self.yolo_model.input: image,
                self.input_image_shape: [raw_shape[0], raw_shape[1]],
                K.learning_phase(): 0
            })


        # get
        polygon_xy = np.zeros([polygons.shape[0], 2 * NUM_ANGLES])
        for b in range(0, out_boxes.shape[0]):
            cy = (out_boxes[b, 0] + out_boxes[b, 2]) // 2
            cx = (out_boxes[b, 1] + out_boxes[b, 3]) // 2
            diagonal = np.sqrt(np.power(out_boxes[b, 3] - out_boxes[b, 1], 2.0) + np.power(out_boxes[b, 2] - out_boxes[b, 0], 2.0))
            for i in range(0, NUM_ANGLES):
                print("angle :", i)
                print("dist :", polygons[b, i])
                dela_x = math.cos(math.radians(i / NUM_ANGLES * 360)) * polygons[b, i] *diagonal
                dela_y = math.sin(math.radians(i / NUM_ANGLES * 360)) * polygons[b, i] *diagonal
                x1 = cx - dela_x
                y1 = cy - dela_y
                polygon_xy[b, i] = x1
                polygon_xy[b, i + NUM_ANGLES] = y1
        return out_boxes, out_scores, out_classes, polygon_xy

    def close_session(self):
        self.sess.close()

def my_Gnearator(images_list, masks_list, batch_size, input_shape, anchors, num_classes, train_flag):
    """
    :param images_list:
    :param masks_list:
    :param batch_size:
    :param input_shape:
    :param train_flag:  STRING Train or else:
    :return:
    """
    n = len(images_list)
    i = 0
    img_data_gen_args = dict(rotation_range=rotation_range,
                             width_shift_range=width_shift_range,
                             height_shift_range=height_shift_range,
                             zoom_range=zoom_range,
                             shear_range=shear_range,
                             horizontal_flip=horizontal_flip,
                             brightness_range=brightness_range
                             )
    mask_data_gen_args = dict(rotation_range=rotation_range,
                              width_shift_range=width_shift_range,
                              height_shift_range=height_shift_range,
                              zoom_range=zoom_range,
                              shear_range=shear_range,
                              horizontal_flip=horizontal_flip
                              )
    image_datagen = ImageDataGenerator(**img_data_gen_args)
    mask_datagen = ImageDataGenerator(**mask_data_gen_args)
    while True:
        image_data_list = []
        box_data_list = []
        # mask_data = []
        # raw_img_path =[]
        # raw_mask_path = []
        # # mypolygon_data = []
        # my_annotation = []
        # print(images_list)
        # print(masks_list)
        ziped_img_mask_list = list(zip(images_list, masks_list))
        count = 1
        while count <= batch_size:
            if count == 1:
                np.random.shuffle(ziped_img_mask_list)
            images_list, masks_list = zip(*ziped_img_mask_list)

            temp_img_path = images_list[i]
            temp_mask_path = masks_list[i]
            img, box, myPolygon = my_get_random_data(temp_img_path, temp_mask_path, input_shape, image_datagen, mask_datagen,
                                          train_or_test=train_flag)
            # print("myPolygon.shape:", myPolygon.shape)
            # check there is zero: if there is boundry points

            # print("count before next:", count)
            # print("range polygon [{}, {}]".format(myPolygon.min(), myPolygon.max()))
            count += 1
            # print(count)
            if np.any(myPolygon == 0) or np.any(myPolygon == img.shape[0] - 1) or np.any(
                    myPolygon == img.shape[1] - 1):  # roll back.

                print("boundary image")
                count -= 1
                continue
            # print("count after next:", count)
            image_data_list.append(img)
            box_data_list.append(box)

        image_batch = np.array(image_data_list)
        box_batch = np.array(box_data_list)
        # preprocess the bbox into the regression targets
        y_true = my_preprocess_true_boxes_NPinterp(box_batch, input_shape, anchors, num_classes)
        yield [image_batch, *y_true], np.zeros(batch_size)


def my_get_random_data(img_path, mask_path, input_shape, image_datagen, mask_datagen, train_or_test, max_boxes=80):
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
        print("train aug")
        seed = np.random.randint(0, 2147483647)

        aug_image = image_datagen.random_transform(image, seed=seed)

        aug_mask = mask_datagen.random_transform(mask, seed=seed)

        copy_mask = aug_mask.copy().astype(np.uint8)
    else:
        print("Test no aug")
        aug_image = image
        copy_mask = mask.copy().astype(np.uint8)

    # print("mask shape after aug:", np.squeeze(aug_mask).shape)
    # aug_image = krs_image.img_to_array(aug_image)
    # aug_mask = krs_image.img_to_array(aug_mask)
    # find polygons with augmask ------------------------------------>
    # imgray = cv2.cvtColor(np.squeeze(copy_mask), cv2.COLOR_BGR2GRAY)
    # print(copy_mask)
    ret, thresh = cv2.threshold(copy_mask, 127, 255, 0)  # this require the numpy array has to be the uint8 type
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # encode contours into annotation lines ---->
    annotation_line, myPolygon = encode_polygone(img_path, contours)
    # decode contours annotation line into distance
    box_data = My_bilinear_decode_annotationlineNP_inter(annotation_line)

    # normal the image ----------------->
    aug_image = aug_image / 255.0
    # return aug_image, aug_mask, annotation_line
    return aug_image, box_data, myPolygon

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


def get_anchors(anchors_path):
    """loads the anchors from a file"""
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

if __name__ == "__main__":

    """
    Retrain the YOLO model for your own dataset.
    """


    def _main():
        phase = 1
        print("current working dir:", os.getcwd())
        cwd =  os.getcwd()
        # os.chdir("E:\\Projects\\poly-yolo\\simulator_dataset")
        current_file_dir_path = os.path.dirname(os.path.realpath(__file__))
        print("current file dir:", current_file_dir_path)
        # annotation_path = current_file_dir_path+'/myTongueTrain.txt'
        # validation_path = current_file_dir_path+'/myTongueTest.txt'
        # for the lab
        annotation_path = current_file_dir_path + '/myTongueTrainLab.txt'
        validation_path = current_file_dir_path + '/myTongueTestLab.txt'


        # log_dir = (current_file_dir_path + '/TongueModelsTang256x256_0.5lr_AngleStep{}_TonguePlus/').format(ANGLE_STEP)
        log_dir = current_file_dir_path + '/Exp_8_MishMyNpInter{}/'.format(model_index)

        plot_folder = log_dir + 'Plots/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)

        classes_path = current_file_dir_path+'/yolo_classesTongue.txt'
        anchors_path = current_file_dir_path+'/yolo_anchorsTongue.txt'
        class_names = get_classes(classes_path)
        num_classes = len(class_names)
        anchors = get_anchors(anchors_path)  # shape [# of anchors, 2]

        # input_shape = (416,832) # multiple of 32, hw
        input_shape = (256, 256) # multiple of 32, hw

        if phase == 1:
            model = create_model(input_shape, anchors, num_classes, load_pretrained=False)
        else:
            model = create_model(input_shape, anchors, num_classes, load_pretrained=True, weights_path=log_dir+'poly_yolo.h5')

        print(model.summary())

        # "plot and save model"
        # plot_model(model, to_file='model.png', show_shapes= True)


        checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                     monitor='val_loss', save_weights_only=True, save_best_only=True, period=1, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, delta=0.03)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
        # custom callback
        plotCallBack = TrainingPlotCallback(save_path= plot_folder)
        # delete old h5 cashes:
        deleteOldH5 =  DeleteEarlySavedH5models(modelSavedPath = log_dir)

        # for my data generator
        # for train dataset
        # train_input_paths = glob('E:\\dataset\\Tongue\\tongue_dataset_tang_plus\\backup\\inputs\\Tongue/*')
        # train_mask_paths = glob('E:\\dataset\\Tongue\\tongue_dataset_tang_plus\\backup\\binary_labels\\Tongue/*.jpg')
        # print("len of train imgs:", len(train_input_paths))

        # assert len(train_input_paths) == len(train_mask_paths), "train imgs and mask are not the same"
        # # for validation dataset  # we need or label and masks are the same shape
        # val_input_paths = glob('E:\\dataset\\Tongue\\mytonguePolyYolo\\test\\test_inputs/*')
        # val_mask_paths = glob('E:\\dataset\\Tongue\\mytonguePolyYolo\\test\\testLabel\\label512640/*.jpg')
        # assert len(val_input_paths) == len(val_mask_paths), "val imgs and mask are not the same"

        # # # for train dataset for the lab
        train_input_paths = glob('F:\\dataset\\tongue_dataset_tang_plus\\inputs/*')
        train_mask_paths = glob('F:\\dataset\\tongue_dataset_tang_plus\\binary_labels/*.jpg')
        print("len of train imgs:", len(train_input_paths))

        assert len(train_input_paths) == len(train_mask_paths), "train imgs and mask are not the same"
        # for validation dataset  # we need or label and masks are the same shape
        val_input_paths = glob('F:\\dataset\\mytonguePolyYolo\\test\\test_inputs/*')
        val_mask_paths = glob('F:\\dataset\\mytonguePolyYolo\\test\\testLabel\\label512640/*.jpg')
        assert len(val_input_paths) == len(val_mask_paths), "val imgs and mask are not the same"

        print("total {} training samples read".format(len(train_input_paths)))
        print("total {} val samples read".format(len(val_input_paths)))
        # create data_generator
        # for train:
        train_Gen = my_Gnearator(train_input_paths, train_mask_paths, batch_size=4, input_shape=[256, 256],
                                   anchors= anchors, num_classes=num_classes,
                                     train_flag="Train")
        val_Gen = my_Gnearator(val_input_paths, val_mask_paths, batch_size=4, input_shape=[256, 256],
                                   anchors= anchors, num_classes=num_classes,
                                   train_flag="test")


        # with open(annotation_path) as f:
        #     lines = f.readlines()
        #     print("total {} training samples read".format(len(lines)))
        # with open(validation_path) as f:
        #     lines_val = f.readlines()
        #     print("total {} val samples read".format(len(lines_val)))
        # lines =  tf.data.TextLineDataset(annotation_path)
        # print("lines:", lines)
        # for line in lines:
        #     print(line)
        # lines_val = tf.data.TextLineDataset(validation_path)
        # for i in range (0, len(lines)):
        #
        #     lines[i] = lines[i].split()
        #     # print(lines[i])
        #     for element in range(1, len(lines[i])):
        #         for symbol in range(lines[i][element].count(',') - 4, MAX_VERTICES * 2, 2):
        #             lines[i][element] = lines[i][element] + ',0,0'
        #
        # for i in range(0, len(lines_val)):
        #     lines_val[i] = lines_val[i].split()
        #     for element in range(1, len(lines_val[i])):
        #         for symbol in range(lines_val[i][element].count(',') - 4, MAX_VERTICES * 2, 2):
        #             lines_val[i][element] = lines_val[i][element] + ',0,0'

        num_val = int(len(val_input_paths))
        num_train = len(train_input_paths)


        batch_size = 4 # decrease/increase batch size according to your memory of your GPU
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))



        model.compile(optimizer=Adadelta(0.5), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        epochs = 100

        # os.chdir("/simulator_dataset/imgs") # for the simulator image path
        model.fit_generator(train_Gen,
                  # steps_per_epoch=max(1, math.ceil(num_train/batch_size)),
                  steps_per_epoch=max(1, num_train // batch_size),
                  validation_data=val_Gen,
                  validation_steps=max(1, num_val // batch_size),
                  epochs=epochs,
                  initial_epoch=0,
                  callbacks=[reduce_lr, checkpoint, plotCallBack, deleteOldH5])



    def get_classes(classes_path):
        """loads the classes"""
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names





    def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
                     weights_path='model_data/yolo_weights.h5'):
        """create the training model"""
        K.clear_session()  # get a new session
        image_input = Input(shape=(None, None, 3))
        h, w = input_shape
        num_anchors = len(anchors)
        # CODE CHANGED FOR MY NP INTERP
        y_true = Input(shape=(h // grid_size_multiplier, w // grid_size_multiplier, anchors_per_level, num_classes + 5 + NUM_ANGLES))
        print("anchors_per_level:", anchors_per_level)
        print("num_classes:", num_classes)
        model_body = yolo_body(image_input, anchors_per_level, num_classes)
        print("model_body.output.shape",model_body.outputs)
        print('Create Poly-YOLO model with {} anchors and {} classes.'.format(num_anchors, num_classes))

        if load_pretrained:
            model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
            print('Load weights {}.'.format(weights_path))

        model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                            arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
            [model_body.output, y_true])
        print("model_loss graph finished")
        model = Model([model_body.input, y_true], model_loss)

        # print(model.summary())
        return model


    # def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, is_random):
    #     """data generator for fit_generator"""
    #     n = len(annotation_lines)
    #     i = 0
    #     while True:
    #         image_data = []
    #         box_data = []
    #         for b in range(batch_size):
    #             if i == 0:
    #                 np.random.shuffle(annotation_lines)
    #             image, box = get_random_data(annotation_lines[i], input_shape, random=is_random)
    #             image_data.append(image)
    #             box_data.append(box)
    #             i = (i + 1) % n
    #         image_data = np.array(image_data)
    #         # print("image_data:", image_data.shape)
    #         box_data = np.array(box_data)
    #         y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)  # return [xy, wh, F/B, one-hot of class, dists.(NUM_ANGLES)]
    #         yield [image_data, *y_true], np.zeros(batch_size)
    #



    # def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes, random):
    #     n = len(annotation_lines)
    #     print("samples in data generator initial:", n)
    #     if n == 0 or batch_size <= 0: return None
    #     return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, random)
    #

    if __name__ == '__main__':
        _main()
