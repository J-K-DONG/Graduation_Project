#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3
@author: JK_DONG
@software: PyCharm
@file: method_test.py
@time: 2019-02-17 10:59

"""

import os
import numpy as np
import random
import cv2

class MethodTest:

    def add_random_gaussian_noise(self, image, means, sigma):
        """
        添加高斯随机噪声
        :param image: 图像list
        :param sigma: 标准差
        :return: image_noise 添加高斯噪声后的图像list
        """
        image_noise = image  # 读进来是uint8类型
        image_dir = "image_noise_test_1/"
        if not os.path.exists(image_dir):
            os.mkdir(image_dir)

        # for k in range(len(image_noise)):
        # means = np.mean(image_noise)

        print("means is : " + str(means))
        # 均值大于0, 表示图像加上一个使自己变亮的噪声, 小于0, 表示图像加上一个使自己变暗的噪声。
        # means = 0
        rows, cols = image_noise.shape[0:2]
        # sigma = random.randint(0, 30)
        # print(sigma)
        temp = random.gauss(means, sigma)
        print(temp)
        for i in range(rows):
            for j in range(cols):  # 每一个点都增加高斯随机数

                image_noise[i, j] = image_noise[i, j] + temp
                if image_noise[i, j] < 0:
                    image_noise[i, j] = 0
                elif image_noise[i, j] > 255:
                    image_noise[i, j] = 255
        image_path = image_dir + "img_gaussian_noise_sigma_" + str(sigma) + ".jpeg"
        cv2.imwrite(image_path, image_noise)
        return image_noise

if __name__ == "__main__":
    m = MethodTest()
    clear_img = "clear_img.jpeg"
    img = cv2.imread(clear_img, flags=0)
    sigma = [1, 2, 5, 10, 15, 20, 25, 30]
    means = np.mean(img)
    # means = 0
    for i in range(len(sigma)):
        image_noise = m.add_random_gaussian_noise(img, means, sigma[i])