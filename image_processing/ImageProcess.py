#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3
@author: JK_DONG
@software: PyCharm
@file: ImageProcess.py
@time: 2019-01-19 15:23

"""

import cv2
import numpy as np
import random
import os




class ImageMethod:

    # 添加高斯随机噪声
    def add_random_gaussian_noise(self, image, sigma):
        image_noise = image
        image_dir = "image_noise/"
        if not os.path.exists(image_dir):
            os.mkdir(image_dir)
        for k in range(len(image_noise)):
            means = np.mean(image_noise[k])
            rows, cols = image_noise[k].shape[0:2]
            for i in range(rows):
                for j in range(cols):  # 每一个点都增加高斯随机数
                    image_noise[k][i, j] = image_noise[k][i, j] + random.gauss(means, sigma)
                    if image_noise[k][i, j] < 0:
                        image_noise[k][i, j] = 0
                    elif image_noise[k][i, j] > 255:
                        image_noise[k][i, j] = 255
            image_path = image_dir + "img_" + str(k+1) + "_gaussian_noise_sigma_" + str(sigma) + ".jpeg"
            cv2.imwrite(image_path, image_noise[k])
        return image_noise


    # 添加椒盐噪声 (测试)(single img)
    def add_salt_pepper_noise(self, src, percetage):
        NoiseImg = src
        NoiseNum = int(percetage * src.shape[0] * src.shape[1])  # 选取随机点的数量
        for i in range(NoiseNum):  # 遍历NoiseNum个随机点并增加随机量
            rand_x = random.randint(0, src.shape[0] - 1)
            rand_y = random.randint(0, src.shape[1] - 1)  # 选取随机点
            if random.randint(0, 1) == 0:
                NoiseImg[rand_x, rand_y] = 0  # 将该点变为黑点
            else:
                NoiseImg[rand_x, rand_y] = 255  # 将该点变为白点
        return NoiseImg


    # 为多张图像添加随机偏移 并保存偏移的图片和每张图片的偏移量
    def add_random_offset(self, image):
        img_num = 55
        rows, cols = image.shape[0:2]
        img_offset = []  # 添加偏移后的图片列表
        offset_data = []  # 55张图片的偏移量列表
        offset_xy = []
        image_dir = "image_offset/"
        if not os.path.exists(image_dir):
            os.mkdir(image_dir)
        for i in range(img_num):
            if i == 0:
                rand_x = 0
                rand_y = 0
            else:
                rand_x = random.randint(-25, 25)
                rand_y = random.randint(-25, 25)
            offset_xy.append(rand_x)
            offset_xy.append(rand_y)
            offset_data.append(offset_xy[2*i:2*i+2])  # 将切片加入偏移量列表
            h = np.float32([[1, 0, rand_x], [0, 1, rand_y]])
            img_offset.append(cv2.warpAffine(image, h, (cols, rows)))
            image_path = image_dir + "img_" + str(i+1) + "_offset_" + str(rand_x) + "_" + str(rand_y) + ".jpeg"
            cv2.imwrite(image_path, img_offset[i])
        cv2.waitKey()
        return img_offset, offset_data


    def get_feature_point(self, image, feature_methord):
        if feature_methord == "sift":
            descriptor = cv2.xfeatures2d_SIFT_create()
        elif feature_methord == "surf":
            descriptor = cv2.xfeatures2d_SURF_create()
        kps, features = descriptor.detectAndCompute(image, None)
        kps = np.float32([kp.pt for kp in kps])
        return kps, features




if __name__ == "__main__":
    method = ImageMethod()
    # img_offset = []
    # offset_data = []
    # img_clear = cv2.imread('clear_img.jpeg', flags=0)
    # img_offset, offset_data = method.add_random_offset(img_clear)
    # for i in range(55):
    #     print("第" + str(i+1) + "幅图片的偏移量列表为：", offset_data[i])
    # img_noise = method.add_random_gaussian_noise(img_offset, sigma=15)

    img_detect = cv2.imread("clear_img.jpeg", 0)
    kps, descriptors = method.get_feature_point(img_detect, "sift")
    print(kps)

