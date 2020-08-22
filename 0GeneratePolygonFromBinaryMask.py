import cv2
import os
from glob import glob
import sys

# binary_masks_paths = glob('E:\\dataset\\Tongue\\tongue_dataset_tang_plus\\binary_labels/*.jpg')
# print("total {} binary mask".format(len(binary_masks_paths)))
# output_train_txt="myTongueTrain.txt"
# # for path in binary_masks_paths:
#
#
# im = cv2.imread(binary_masks_paths[0])
# imCopy = im.copy()
# imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(imgray, 127, 255, 0)
# # cv2.imshow('Canny Edges After Contouring', edged)
# # cv2.waitKey(0)
#
# # ret, thresh = cv2.threshold(imgray, 127, 255, 0)
# # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
# image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# print(contours[0].shape)
# # cv2.drawContours(imCopy, contours, -1, (0, 255, 0))
# # cv2.imshow('draw contours', imCopy)
# # cv2.waitKey(0)
# myPolygon = contours[0].reshape([-1, 2])
# print("mypolygon:", myPolygon)
#
# # start to write into txt
# if not (os.path.exists(output_train_txt)):
#     f = open(output_train_txt, "w")   # if file not exist create the file
#     f.close()
#
# # open the file to write:
# out = open(output_train_txt, 'w')
# print("sys.maxsize:", sys.maxsize)
# min_x = sys.maxsize
# max_x = 0
# min_y = sys.maxsize
# max_y = 0
# polygons_line = ''
# skipped = 0
# polygon_line = ''
#
# c = 0 # cls id only one tongue
# # for po
# for x, y in myPolygon:
#     print("({}, {})".format(x, y))
#     if x > max_x: max_x = x
#     if y > max_y: max_y = y
#     if x < min_x: min_x = x
#     if y < min_y: min_y = y
#     polygon_line += ',{},{}'.format(x, y)
#     if max_x - min_x <= 1.0 or max_y - min_y <= 1.0:
#         skipped += 1
#         continue
#
# polygons_line += ' {},{},{},{},{}'.format(min_x, min_y, max_x, max_y, c) + polygon_line
# img_path = os.path.normpath(binary_masks_paths[0])
# # print(img_path)
# # img_path_splits = img_path.split(os.sep)
# # print("img_path_splits:", img_path_splits)
# # reconnect path components with os.path.join to make sure she seperator is ok.
# # for i in range(0,len(img_path_splits), 2):
# #     print(i)
# #     img_path = os.path.join(img_path_splits[i], img_path_splits[i+1])
# print("img path:",img_path )
# annotation_line = img_path + polygons_line
#
# print(annotation_line, file=out)


def writetxt_data_POLYYOLO(imgs_root, label_root, output_filename):
    """
    :param imgs_root: e.g E:\dataset\Tongue\舌領域抽出Paper\Sample16_new
    :param label_root:
    :output_filename: output_filename= "myTongueTrain.txt"
    :return:
    """
    imgs_paths = sorted(glob(imgs_root+'/*'))
    binaryMask_paths = sorted(glob(label_root + '/*'))
    print("total {} images".format(len(imgs_paths)))
    print("total {} binary masks".format(len(binaryMask_paths)))
    max_v =1000

    # initialize for writing strings
    # check the output file existance
    if not (os.path.exists(output_filename)):
        f = open(output_filename, "w")  # if file not exist create the file
        f.close()
    # open the file to write:
    out = open(output_filename, 'w')
    print("sys.maxsize:", sys.maxsize)

    skipped = 0
    c = 0  # cls id only one tongue

    for i in range(len(binaryMask_paths)):
        polygons_line = ''
        print(imgs_paths[i])
        print(binaryMask_paths[i])
        # find polygons
        im = cv2.imread(binaryMask_paths[i])
        # imCopy = im.copy()
        imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 127, 255, 0)
        image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for obj in contours:
            print(obj.shape)
            myPolygon = obj.reshape([-1, 2])
            print("mypolygon:", myPolygon.shape)
            if  myPolygon.shape[0] >max_v:
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
        # if polygons_line == '': continue
        img_path = os.path.normpath(imgs_paths[i])

        print("img path:", img_path)
        annotation_line = img_path + polygons_line

        print(annotation_line, file=out)
    out.close()

if __name__ == '__main__':
    # generate train txt
    # writetxt_data_POLYYOLO(imgs_root="E:\\dataset\\Tongue\\mytonguePolyYolo\\train\\Sample16_new",
    #                        label_root="E:\\dataset\\Tongue\\mytonguePolyYolo\\train\\Label16_new",
    #
    #                        output_filename="E:\\Projects\\poly-yolo\\MyPolys\\TonguePlusData\\myTongueTrain.txt")

    # # tonguePlus
    # writetxt_data_POLYYOLO(imgs_root="E:\\dataset\\Tongue\\tongue_dataset_tang_plus\\inputs",
    #                        label_root="E:\\dataset\\Tongue\\tongue_dataset_tang_plus\\binary_labels",
    #
    #                        output_filename="E:\\Projects\\poly-yolo\\MyPolys\\TonguePlusData\\myTongueTrain.txt")

    # tonguePlus
    writetxt_data_POLYYOLO(imgs_root="E:\\dataset\\Tongue\\tongue_dataset_tang_plus\\backup\\Angumented_dataset\\rotate45_widthshift0.1_heightshift0.1_zoom0.2_num8860\\input",
                           label_root="E:\\dataset\\Tongue\\tongue_dataset_tang_plus\\backup\\Angumented_dataset\\rotate45_widthshift0.1_heightshift0.1_zoom0.2_num8860\\label",

                           output_filename="E:\\dataset\\Tongue\\tongue_dataset_tang_plus\\backup\\Angumented_dataset\\rotate45_widthshift0.1_heightshift0.1_zoom0.2_num8860\\myTongueTrain_rotate45_widthshift0.1_heightshift0.1_zoom0.2_num8860.txt")
    #
    # writetxt_data_POLYYOLO(imgs_root="E:\\dataset\\Tongue\\mytonguePolyYolo\\val\\VS16_new",
    #                        label_root="E:\\dataset\\Tongue\\mytonguePolyYolo\\val\\VL16_new",
    #                        output_filename="myTongueVal.txt")

    # writetxt_data_POLYYOLO(imgs_root="E:\\dataset\\Tongue\\mytonguePolyYolo\\test\\test_inputs",
    #                        label_root="E:\\dataset\\Tongue\\mytonguePolyYolo\\test\\testLabel\\label512640",
    #                        output_filename="myTongueTest.txt")