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
import shutil

import numpy as np
import random
import cv2
import time


# 方法测试类
class MethodTest:
    image_num = 1

    def add_random_gaussian_noise1(self, image):
        """
        添加高斯随机噪声
        :param image: 图像list（55张）
        :return: image_noise: 添加高斯噪声后的图像list     gaussian_sigma: 高斯噪声的方差list
        """
        image_noise = image  # 读进来是uint8类型
        gaussian_sigma = []
        if not os.path.exists(self.image_noise_dir):
            os.mkdir(self.image_noise_dir)
        for k in range(self.image_num):
            sigma = random.randint(0, 10)
            # print("第" + str(k + 1) + "张图片的方差为 : " + str(sigma))
            gaussian_sigma.append(sigma)
            rows, cols = image_noise[k].shape[0:2]
            image_noise[k] = image_noise[k] + np.random.randn(573, 759) * sigma  # 矩阵相加
            for i in range(rows):
                for j in range(cols):  # 每一个点先检测像素值是否溢出  再进行赋值
                    r1 = np.where((image_noise[k][i, j]) > 255, 255, (image_noise[k][i, j]))
                    r2 = np.where((r1 < 0), 0, r1)
                    image_noise[k][i, j] = np.round(r2)
            image_noise_temp = image_noise[k].astype('uint8')  # 将 ndarray.float64 转换为 uint8
            image_noise[k] = image_noise_temp
            image_path = self.image_noise_dir + "img_" + str(k+1) + "_gaussian_noise_sigma_" + str(sigma) + ".jpeg"
            cv2.imwrite(image_path, image_noise[k])
        return image_noise, gaussian_sigma

    def add_random_offset1(self, image):
        """
        为多张图像添加随机偏移 并保存偏移的图片和每张图片的偏移量
        :param image: 图像list
        :return:  img_offset:添加偏移量的图像list   offset_data:所有图像的偏移量list [[rand_x, rand_y],[rand_x, rand_y]...]
        """
        rows, cols = image.shape[0:2]
        img_offset = []  # 添加偏移后的图片列表
        offset_data = []  # 55张图片的偏移量列表
        offset_xy = []
        for i in range(self.image_num):
            rand_x = random.randint(-25, 25)
            rand_y = random.randint(-25, 25)
            # f.write(str(rand_x) + "," + str(rand_y) + "\n")
            offset_xy.append(rand_x)
            offset_xy.append(rand_y)
            offset_data.append(offset_xy[2*i:2*i+2])  # 将切片加入偏移量列表
            h = np.float32([[1, 0, rand_x], [0, 1, rand_y]])
            img_offset.append(cv2.warpAffine(image, h, (cols, rows)))
            image_path = self.image_offset_dir + "img_" + str(i+1) + "_offset_" + str(rand_x) + "_" + str(rand_y) + ".jpeg"
            cv2.imwrite(image_path, img_offset[i])
        cv2.waitKey()
        return img_offset, offset_data

    def load_images(self, file_path):
        """
        读取文件路径下多张图像
        :param file_path: 读入文件的相对路径
        :return: image_list: 多张图像的list
        """
        image_list = []
        if os.path.exists(file_path):
            for img_name in os.listdir(file_path):
                image_dir = file_path + str(img_name)
                image_read = cv2.imread(image_dir, flags=0)
                image_list.append(image_read)
        return image_list

    def delete_test_data(self):
        """
        删除上次运行的结果文件
        :return:
        """
        if os.path.exists(self.image_offset_dir):
            shutil.rmtree(self.image_offset_dir)
        if os.path.exists(self.image_noise_dir):
            shutil.rmtree(self.image_noise_dir)
        if os.path.exists(self.image_offset_txt):
            os.remove(self.image_offset_txt)
        print("清除上次测试数据")

    def add_random_gaussian_noise(self, image):
        """
        添加高斯随机噪声
        :param image: 图像list（55张）
        :return: image_noise: 添加高斯噪声后的图像list     gaussian_sigma: 高斯噪声的方差list
        """
        image_noise = image  # 读进来是uint8类型
        # for k in range(self.image_num):
        # sigma = random.randint(0, 10)
        sigma = 20
        rows, cols = image_noise.shape[0:2]
        image_noise = image_noise + np.random.randn(573, 759) * sigma  # 矩阵相加
        for i in range(rows):
            for j in range(cols):  # 每一个点先检测像素值是否溢出  再进行赋值
                r1 = np.where((image_noise[i, j]) > 255, 255, (image_noise[i, j]))
                r2 = np.where((r1 < 0), 0, r1)
                image_noise[i, j] = np.round(r2)
        image_noise_temp = image_noise.astype('uint8')  # 将 ndarray.float64 转换为 uint8
        image_noise = image_noise_temp
        image_path = "img_gaussian_noise_sigma_" + str(sigma) + ".jpeg"
        cv2.imwrite(image_path, image_noise)
        return image_noise

    def fuse_by_spatial_frequency(self, images, block_size, threshold):
        """

        :param images:
        :return:
        """
        (last_image, next_image) = images
        row, col = last_image.shape[0:2]
        isGPUAvailable = False
        choice_full_last = np.array([(0, 0, 0),
                                     (0, 1, 0),
                                     (0, 0, 0)])
        choice_full_next = np.array([(1, 1, 1),
                                     (1, 0, 1),
                                     (1, 1, 1)])
        if isGPUAvailable:
            fuse_region = 0
        else:
            row_num = row // block_size
            col_num = col // block_size
            fusion_choice = np.zeros((row_num + 1, col_num + 1), dtype=np.int)
            print("图像共分为 " + str(row_num + 1) + " 行 " + str(col_num + 1) + " 列")
            for i in range(row_num + 1):
                for j in range(col_num + 1):  # 图像切片比较
                    if i < row_num and j < col_num:
                        row_end_position = (i + 1) * block_size
                        col_end_position = (j + 1) * block_size
                    elif i < row_num and j == col_num:
                        row_end_position = (i + 1) * block_size
                        col_end_position = col
                    elif i == row_num and j < col_num:
                        row_end_position = row
                        col_end_position = (j + 1) * block_size
                    else:
                        row_end_position = row
                        col_end_position = col
                    last_image_block = last_image[(i * block_size):row_end_position, (j * block_size):col_end_position]
                    next_image_block = next_image[(i * block_size):row_end_position, (j * block_size):col_end_position]
                    last_image_block_sf = self.calculate_spatial_frequency(last_image_block)
                    next_image_block_sf = self.calculate_spatial_frequency(next_image_block)
                    # print("imageA的第 " + str(i + 1) + " 行 " + str(j + 1) + " 列的图像块的SF值为： ", SF_imageA_block)
                    # print("imageB的第 " + str(i + 1) + " 行 " + str(j + 1) + " 列的图像块的SF值为： ", SF_imageB_block)
                    if last_image_block_sf > next_image_block_sf + threshold:
                        last_image_block = last_image_block
                    elif last_image_block_sf < next_image_block_sf - threshold:
                        fusion_choice[i, j] = 1
                    else:
                        fusion_choice[i, j] = 2

            for n in range(row_num + 1):
                for m in range(col_num + 1):  # majority filter is 3*3
                    if n < row_num - 1 and m < col_num - 1:
                        if np.all(fusion_choice[n:n+3, m:m+3] == choice_full_last):
                            fusion_choice[n + 1, m + 1] = 0
                        elif np.all(fusion_choice[n:n+3, m:m+3] == choice_full_next):
                            fusion_choice[n + 1, m + 1] = 1
                    if fusion_choice[n, m] == 1:
                        last_image[n * block_size:(n + 1) * block_size, m * block_size:(m + 1) * block_size] = \
                            next_image[n * block_size:(n + 1) * block_size, m * block_size:(m + 1) * block_size]
                    if fusion_choice[n, m] == 2:
                        # print(1)
                        last_image[n * block_size:(n + 1) * block_size, m * block_size:(m + 1) * block_size] = \
                            ((last_image[n * block_size:(n + 1) * block_size, m * block_size:(m + 1) * block_size]
                              .astype(int) + next_image[n * block_size:(n + 1) * block_size,
                              m * block_size:(m + 1) * block_size].astype(int)) / 2).astype('uint8')
            print(fusion_choice)
            cv2.imwrite("fuseRegion.jpeg", last_image)
            fuse_region = last_image
        return fuse_region

    @staticmethod
    def calculate_spatial_frequency(image):
        """
        计算空间频率
        :param image:
        :return:
        """
        rf_temp = 0
        cf_temp = 0
        row, col = image.shape[0:2]
        image_temp = image.astype(int)
        for i in range(row):
            for j in range(col):
                if j < col - 1:
                    rf_temp = rf_temp + np.square(image_temp[i, j + 1] - image_temp[i, j])
                if i < row - 1:
                    cf_temp = cf_temp + np.square(image_temp[i + 1, j] - image_temp[i, j])
        rf = np.sqrt(float(rf_temp) / float(row * col))
        cf = np.sqrt(float(cf_temp) / float(row * col))
        sf = np.sqrt(np.square(rf) + np.square(cf))
        return sf

    @staticmethod
    def add_gaussian_blur(input_img):
        """

        :param input_img:
        :return:
        """
        kernel_size = (5, 5)
        block_size = 200
        sigma = 20
        # temp = random.randint(0, 3)
        temp = 0
        row, col = input_img.shape[0:2]
        if temp == 0:
            dx = random.randint(0, row - block_size)
            dy = random.randint(0, col - block_size)
            img_block = cv2.GaussianBlur(input_img[dx:dx + block_size, dy:dy + block_size], kernel_size, sigma)
            print(img_block)
            print(img_block.shape[0:2])
            print(dx, dy)
            input_img[dx:dx + block_size, dy:dy + block_size] = img_block
        return input_img

    @staticmethod
    def test_spatial_frequency():
        weight_matrix = np.array([(1, 1, 1, 0),
                                  (1, 1, 0, 0),
                                  (1, 1, 0, 0),
                                  (1, 1, 0, 1)])
        image_1 = np.array([(9, 9, 9, 9),
                            (9, 9, 9, 9),
                            (9, 9, 9, 9),
                            (9, 9, 9, 9)])
        image_2 = np.array([(8, 8, 8, 8),
                            (8, 8, 8, 8),
                            (8, 8, 8, 8),
                            (8, 8, 8, 8)])
        img = image_1 * weight_matrix + image_2 * (1 - weight_matrix)
        print(1 - weight_matrix)
        print(img)
        image_1 = 1
        print(image_1)


if __name__ == "__main__":

    start_time = time.time()

    m = MethodTest()

    clear_img = "clear_img.jpeg"
    img = cv2.imread(clear_img, flags=0)
    print(img)
    test = cv2.imread("1.jpeg", 0)
    images = (img, test)
    SF_img = m.calculate_spatial_frequency(img)
    print(SF_img)
    SF_noise = m.calculate_spatial_frequency(test)
    print(SF_noise)
    fuseRegion = m.fuse_by_spatial_frequency(images, 40, 1)
    end_time = time.time()
    print("the time of fusing images is{:.3f}\'s".format(end_time - start_time))

    # image_noise = m.add_random_gaussian_noise(img)
    # for i in range(20):
    #     temp = random.randint(0, 3)
    #     if temp == 0:
    #         print(temp)
    # img_blur = m.add_gaussian_blur(img)
    # cv2.imwrite("img_blur.jpeg", img_blur)

    # m.test_spatial_frequency()



