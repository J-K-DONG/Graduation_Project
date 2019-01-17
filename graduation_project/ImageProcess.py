#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3
@author: JK_DONG
@software: PyCharm
@file: ImageProcess.py
@time: 2019-01-16 17:12
"""


import cv2
import numpy as np
from matplotlib import pyplot as plt
import random


class ImageMethod:

    def add_random_offset_test(self, image, image_num):
        clear_img = cv2.imread('clear_picture.jpeg', flags=0)
        # 为使得偏移效果明显  为图像添加白色的边框
        white = [255, 255, 255]
        constant_white = cv2.copyMakeBorder(clear_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=white)  # 边界分别为：上下左右
        black = [0, 0, 0]
        constant_black = cv2.copyMakeBorder(constant_white, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=black)  #再添加黑色的50像素的边框

        rows = constant_black.shape[0]
        cols = constant_black.shape[1]

        # rand_x = [0]  # 随机横向偏移量
        # rand_y = [0]  # 随机纵向偏移量
        # H = []  # 实现随机偏移的矩阵
        # offset_img = [constant_black]  # 随机偏移后的图片数组

        # for i in range(10):
        #     rand_x.append(random.randint(-25, 25))
        #     rand_y.append(random.randint(-25, 25))
        #     H.append(np.float32([[1, 0, rand_x[i]], [0, 1, rand_y[i]]]))
        #     offset_img.append(cv2.warpAffine(constant_black, H[i], (cols, rows)))  # 需要图像、变换矩阵、变换后的大小
        #     filename = "offset_" + str(rand_x[i]) + "_" + str(rand_y[i]) + ".jpeg"
        #     cv2.imwrite(filename, offset_img[i])
        #     plt.subplot(3, 3, i+1), plt.imshow(offset_img[i], 'gray'), plt.title("offset_" + str(rand_x[i]) + "_" + str(rand_y[i]))
        # plt.show()

        for i in range(9):
            if i == 0:
                rand_x = 0
                rand_y = 0
            else:
                rand_x = random.randint(-25, 25)
                rand_y = random.randint(-25, 25)
            H = np.float32([[1, 0, rand_x], [0, 1, rand_y]])
            offset_img = cv2.warpAffine(constant_black, H, (cols, rows))
            filename_p = "offset_" + str(rand_x) + "_" + str(rand_y)
            filename_f = filename_p + ".jpeg"
            cv2.imwrite(filename_f, offset_img)
            plt.subplot(3, 3, i+1)
            plt.imshow(offset_img, 'gray')
            plt.title(filename_p)

        plt.show()
        cv2.waitKey()

    # 为多张图像添加随机偏移 并保存偏移的图片和每张图片的偏移量
    def add_random_offset(self, image):

        img_num = 55
        rows, cols = image.shape[0:2]
        img_offset = []
        offset_data = []
        offset_xy = []
        for i in range(img_num):
            if i == 0:
                rand_x = 0
                rand_y = 0
            else:
                rand_x = random.randint(-25, 25)
                rand_y = random.randint(-25, 25)
            offset_xy.append(rand_x)
            offset_xy.append(rand_y)
            offset_data.append(offset_xy[2*i:2*i+2])
            h = np.float32([[1, 0, rand_x], [0, 1, rand_y]])
            img_offset.append(cv2.warpAffine(image, h, (cols, rows)))
            filename_plt = "img_" + str(i+1) + "_offset_" + str(rand_x) + "_" + str(rand_y)
            filename_jpeg = filename_plt + ".jpeg"
            cv2.imwrite(filename_jpeg, img_offset[i])
        cv2.waitKey()
        return img_offset, offset_data


if __name__ == "__main__":

    method = ImageMethod()
    img_offset = []
    offset_data = []
    img_clear = cv2.imread('clear_img.jpeg', flags=0)
    img_offset, offset_data = method.add_random_offset(img_clear)
    for i in range(55):
        print("第" + str(i+1) + "幅图片的偏移量列表为：", offset_data[i])




