import matplotlib
# Specifying the backend to be used before importing pyplot
# to avoid "RuntimeError: Invalid DISPLAY variable"
matplotlib.use('agg')
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np

class MyGenerator(keras.utils.Sequence):
    def __init__(self, annotation_lines, batch_size, input_shape, anchors, num_classes, is_random) :
        self.annotation_lines = annotation_lines
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.anchors =  anchors
        self.num_classes =  num_classes
        self.is_random = is_random


    def __len__(self) :
        return (np.ceil(len(self.annotation_lines) / float(self.batch_size))).astype(np.int)


    def __getitem__(self, idx) :
        return self.data_gen()

    def data_gen(self):
        """data generator for fit_generator"""
        n = len(self.annotation_lines)
        i = 0
        while True:
            image_data = []
            box_data = []
            for b in range(self.batch_size):
                if i == 0:
                    np.random.shuffle(self.annotation_lines)
                image, box = self.get_random_data(annotation_lines[i], input_shape, random=is_random)
                image_data.append(image)
                box_data.append(box)
                i = (i + 1) % n
            image_data = np.array(image_data)
            # print("image_data:", image_data.shape)
            box_data = np.array(box_data)
            y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
            yield [image_data, *y_true], np.zeros(batch_size)

    def get_random_data(line, input_shape, random=True, max_boxes=80, hue_alter=20, sat_alter=30, val_alter=30,
                        proc_img=True):
        # load data
        # the color conversion is later. it is not necessary to realize bgr->rgb->hsv->rgb
        # print("get_random_data line[0]:", line[0])
        # print(os.getcwd())

        image = cv.imread(line[0])
        # print("get_random_data image:", image)

        iw = image.shape[1]
        ih = image.shape[0]
        h, w = input_shape
        box = np.array([np.array(list(map(float, box.split(','))))
                        for box in line[1:]])

        if not random:
            # resize image
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2
            image_data = 0
            if proc_img:
                # image = image.resize((nw, nh), Image.BICUBIC)
                image = cv.cvtColor(
                    cv.resize(image, (nw, nh), interpolation=cv.INTER_CUBIC), cv.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                new_image = Image.new('RGB', (w, h), (128, 128, 128))
                new_image.paste(image, (dx, dy))
                image_data = np.array(new_image) / 255.
            # correct boxes
            box_data = np.zeros((max_boxes, 5 + NUM_ANGLES3))
            if len(box) > 0:
                np.random.shuffle(box)
                if len(box) > max_boxes:
                    box = box[:max_boxes]
                box[:, [0, 2]] = box[:, [0, 2]] * scale + dx
                box[:, [1, 3]] = box[:, [1, 3]] * scale + dy
                box_data[:len(box), 0:5] = box[:, 0:5]
                for b in range(0, len(box)):
                    for i in range(5, MAX_VERTICES * 2, 2):
                        if box[b, i] == 0 and box[b, i + 1] == 0:
                            continue
                        box[b, i] = box[b, i] * scale + dx
                        box[b, i + 1] = box[b, i + 1] * scale + dy

                # box_data[:, i:NUM_ANGLES3 + 5] = 0

                for i in range(0, len(box)):
                    boxes_xy = (box[i, 0:2] + box[i, 2:4]) // 2

                    for ver in range(5, MAX_VERTICES * 2, 2):
                        if box[i, ver] == 0 and box[i, ver + 1] == 0:
                            break
                        dist_x = boxes_xy[0] - box[i, ver]
                        dist_y = boxes_xy[1] - box[i, ver + 1]
                        dist = np.sqrt(np.power(dist_x, 2) + np.power(dist_y, 2))
                        if (dist < 1): dist = 1  # to avoid inf or nan in log in loss

                        angle = np.degrees(np.arctan2(dist_y, dist_x))
                        if (angle < 0): angle += 360
                        iangle = int(angle) // ANGLE_STEP
                        relative_angle = (angle - (iangle * int(ANGLE_STEP))) / ANGLE_STEP

                        if dist > box_data[
                            i, 5 + iangle * 3]:  # check for vertex existence. only the most distant is taken
                            box_data[i, 5 + iangle * 3] = dist
                            box_data[i, 5 + iangle * 3 + 1] = relative_angle
                            box_data[
                                i, 5 + iangle * 3 + 2] = 1  # problbility  mask to be 1 for the exsitance of the vertex otherwise =0
            return image_data, box_data

        # resize image
        random_scale = rd.uniform(.6, 1.4)
        scale = min(w / iw, h / ih)
        nw = int(iw * scale * random_scale)
        nh = int(ih * scale * random_scale)

        # force nw a nh to be an even
        if (nw % 2) == 1:
            nw = nw + 1
        if (nh % 2) == 1:
            nh = nh + 1

        # jitter for slight distort of aspect ratio
        if np.random.rand() < 0.3:
            if np.random.rand() < 0.5:
                nw = int(nw * rd.uniform(.8, 1.0))
            else:
                nh = int(nh * rd.uniform(.8, 1.0))

        image = cv.resize(image, (nw, nh), interpolation=cv.INTER_CUBIC)
        nwiw = nw / iw
        nhih = nh / ih

        # clahe. applied on resized image to save time. but before placing to avoid
        # the influence of homogenous background
        clahe = cv.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
        lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
        l, a, b = cv.split(lab)
        cl = clahe.apply(l)
        limg = cv.merge((cl, a, b))
        image = cv.cvtColor(limg, cv.COLOR_LAB2BGR)

        # place image
        dx = rd.randint(0, max(w - nw, 0))
        dy = rd.randint(0, max(h - nh, 0))

        new_image = np.full((h, w, 3), 128, dtype='uint8')
        new_image, crop_coords, new_img_coords = random_crop(
            image, new_image)

        # flip image or not
        flip = rd.random() < .5
        if flip:
            new_image = cv.flip(new_image, 1)

        # distort image
        hsv = np.int32(cv.cvtColor(new_image, cv.COLOR_BGR2HSV))

        # linear hsv distortion
        hsv[..., 0] += rd.randint(-hue_alter, hue_alter)
        hsv[..., 1] += rd.randint(-sat_alter, sat_alter)
        hsv[..., 2] += rd.randint(-val_alter, val_alter)

        # additional non-linear distortion of saturation and value
        if np.random.rand() < 0.5:
            hsv[..., 1] = hsv[..., 1] * rd.uniform(.7, 1.3)
            hsv[..., 2] = hsv[..., 2] * rd.uniform(.7, 1.3)

        hsv[..., 0][hsv[..., 0] > 179] = 179
        hsv[..., 0][hsv[..., 0] < 0] = 0
        hsv[..., 1][hsv[..., 1] > 255] = 255
        hsv[..., 1][hsv[..., 1] < 0] = 0
        hsv[..., 2][hsv[..., 2] > 255] = 255
        hsv[..., 2][hsv[..., 2] < 0] = 0

        image_data = cv.cvtColor(
            np.uint8(hsv), cv.COLOR_HSV2RGB).astype('float32') / 255.0

        # add noise
        if np.random.rand() < 0.15:
            image_data = np.clip(image_data + np.random.rand() *
                                 image_data.std() * np.random.random(image_data.shape), 0, 1)

        # correct boxes
        box_data = np.zeros((max_boxes, 5 + NUM_ANGLES3))

        if len(box) > 0:
            np.random.shuffle(box)
            # rescaling separately because 5-th element is class
            box[:, [0, 2]] = box[:, [0, 2]] * nwiw  # for x
            # rescale polygon vertices
            box[:, 5::2] = box[:, 5::2] * nwiw
            # rescale polygon vertices
            box[:, [1, 3]] = box[:, [1, 3]] * nhih  # for y
            box[:, 6::2] = box[:, 6::2] * nhih

            # # mask out boxes that lies outside of croping window ## new commit deleted
            # mask = (box[:, 1] >= crop_coords[0]) & (box[:, 3] < crop_coords[1]) & (
            #     box[:, 0] >= crop_coords[2]) & (box[:, 2] < crop_coords[3])
            # box = box[mask]

            # transform boxes to new coordinate system w.r.t new_image
            box[:, :2] = box[:, :2] - [crop_coords[2], crop_coords[0]] + [new_img_coords[2], new_img_coords[0]]
            box[:, 2:4] = box[:, 2:4] - [crop_coords[2], crop_coords[0]] + [new_img_coords[2], new_img_coords[0]]
            if flip:
                box[:, [0, 2]] = (w - 1) - box[:, [2, 0]]

            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] >= w] = w - 1
            box[:, 3][box[:, 3] >= h] = h - 1
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box

            if len(box) > max_boxes:
                box = box[:max_boxes]

            box_data[:len(box), 0:5] = box[:, 0:5]

        # -------------------------------start polygon vertices processing-------------------------------#
        for b in range(0, len(box)):
            boxes_xy = (box[b, 0:2] + box[b, 2:4]) // 2
            for i in range(5, MAX_VERTICES * 2, 2):
                if box[b, i] == 0 and box[b, i + 1] == 0:
                    break
                box[b, i:i + 2] = box[b, i:i + 2] - [crop_coords[2], crop_coords[0]] + [new_img_coords[2],
                                                                                        new_img_coords[0]]  # transform
                if flip: box[b, i] = (w - 1) - box[b, i]
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
        # ---------------------------------end polygon vertices processing-------------------------------#
        return image_data, box_data