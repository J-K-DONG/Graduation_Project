#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3
@author: JK_DONG
@software: PyCharm
@file: ImageProcess.py
@time: 2019-02-12 11:47

"""

import cv2
import numpy as np
import random
import os
import shutil


class ImageMethod:
    searchRatio = 0.75  # 0.75 is common value for matches
    image_noise_dir = "image_noise/"
    image_offset_dir = "image_offset/"
    image_offset_txt = "offset_data.txt"


    def add_random_gaussian_noise(self, image):
        """
        添加高斯随机噪声
        :param image: 图像list
        :param sigma: 标准差
        :return: image_noise 添加高斯噪声后的图像list
        """
        image_noise = image  # 读进来是uint8类型
        gaussian_means = []
        gaussian_sigma = []
        if not os.path.exists(self.image_noise_dir):
            os.mkdir(self.image_noise_dir)

        for k in range(len(image_noise)):
            means = np.mean(image_noise[k])
            print("means is : " + str(means))
            gaussian_means.append(means)
            sigma = random.randint(0, 5)
            print("sigma is : " + str(sigma))
            gaussian_sigma.append(sigma)
            rows, cols = image_noise[k].shape[0:2]
            for i in range(rows):
                for j in range(cols):  # 每一个点都增加高斯随机数
                    image_noise[k][i, j] = image_noise[k][i, j] + random.gauss(means, sigma)
                    if image_noise[k][i, j] < 0:
                        image_noise[k][i, j] = 0
                    elif image_noise[k][i, j] > 255:
                        image_noise[k][i, j] = 255
            image_path = self.image_noise_dir + "img_" + str(k+1) + "_gaussian_noise_sigma_" + str(sigma) + ".jpeg"
            cv2.imwrite(image_path, image_noise[k])
        return image_noise, gaussian_means, gaussian_sigma


    def add_salt_pepper_noise(self, src, percetage):
        """
        添加椒盐噪声 (测试)(single img)
        :param src: 单张图像
        :param percetage:
        :return:
        """
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

    def add_random_offset(self, image):
        """
        为多张图像添加随机偏移 并保存偏移的图片和每张图片的偏移量
        :param image: 图像list
        :return:  img_offset:添加偏移量的图像list   offset_data:所有图像的偏移量list [[rand_x, rand_y],[rand_x, rand_y]...]
        """
        img_num = 55
        rows, cols = image.shape[0:2]
        img_offset = []  # 添加偏移后的图片列表
        offset_data = []  # 55张图片的偏移量列表
        offset_xy = []
        # image_dir = "image_offset/"
        if not os.path.exists(self.image_offset_dir):
            os.mkdir(self.image_offset_dir)
        f = open('offset_data.txt', 'a')
        for i in range(img_num):
            rand_x = random.randint(-25, 25)
            rand_y = random.randint(-25, 25)
            f.write("[" + str(rand_x) + ", " + str(rand_y) + "]\n")
            offset_xy.append(rand_x)
            offset_xy.append(rand_y)
            offset_data.append(offset_xy[2*i:2*i+2])  # 将切片加入偏移量列表
            h = np.float32([[1, 0, rand_x], [0, 1, rand_y]])
            img_offset.append(cv2.warpAffine(image, h, (cols, rows)))
            image_path = self.image_offset_dir + "img_" + str(i+1) + "_offset_" + str(rand_x) + "_" + str(rand_y) + ".jpeg"
            cv2.imwrite(image_path, img_offset[i])
        cv2.waitKey()
        return img_offset, offset_data

    def get_feature_point(self, image, feature_methord):
        """
        特征点检测&描述
        :param image: 检测图像
        :param feature_methord: 检测方法 sift  or  surf
        :return:  kps:该图像的特征   features:图像特征点的描述符
        """
        if feature_methord == "sift":
            descriptor = cv2.xfeatures2d.SIFT_create()
        elif feature_methord == "surf":
            descriptor = cv2.xfeatures2d.SURF_create()
            # print("use surf")
        kps, features = descriptor.detectAndCompute(image, None)
        kps = np.float32([kp.pt for kp in kps])
        return kps, features

    def match_descriptors(self, featuresA, featuresB, match_method):
        '''
        功能：匹配特征点
        :param featuresA: 第一张图像的特征点描述符
        :param featuresB: 第二张图像的特征点描述符
        :return:返回匹配的对数matches
        '''
        # 建立暴力匹配器
        if match_method == "surf" or match_method == "sift":
            matcher = cv2.DescriptorMatcher_create("BruteForce")
            # 使用KNN检测来自A、B图的SIFT特征匹配对，K=2，返回一个列表
            rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
            matches = []
            for m in rawMatches:
                # 当最近距离跟次近距离的比值小于ratio值时，保留此匹配对
                if len(m) == 2 and m[0].distance < m[1].distance * self.searchRatio:
                    # 存储两个点在featuresA, featuresB中的索引值
                    matches.append((m[0].trainIdx, m[0].queryIdx))
        return matches

    def get_offset_by_mode(self, kpsA, kpsB, matches, offsetEvaluate=3):
        """
        功能：通过求众数的方法获得偏移量
        :param kpsA: 第一张图像的特征（原图）
        :param kpsB: 第二张图像的特征（测试图片）
        :param matches: 配准列表
        :param offsetEvaluate: 如果众数的个数大于本阈值，则配准正确，默认为10
        :return: 返回(totalStatus, [dx, dy]), totalStatus 是否正确，[dx, dy]默认[0, 0]
        """
        totalStatus = True
        if len(matches) == 0:
            totalStatus = False
            return (totalStatus, [0, 0])
        dxList = []
        dyList = []
        for trainIdx, queryIdx in matches:
            ptA = (kpsA[queryIdx][1], kpsA[queryIdx][0])
            ptB = (kpsB[trainIdx][1], kpsB[trainIdx][0])
            # dxList.append(int(round(ptA[0] - ptB[0])))
            # dyList.append(int(round(ptA[1] - ptB[1])))
            if int(ptA[0] - ptB[0]) == 0 and int(ptA[1] - ptB[1]) == 0:
                continue
            dxList.append(int(ptB[0] - ptA[0]))
            dyList.append(int(ptB[1] - ptA[1]))
        if len(dxList) == 0:
            dxList.append(0)
            dyList.append(0)
        # Get Mode offset in [dxList, dyList], thanks for clovermini
        zipped = zip(dxList, dyList)
        zip_list = list(zipped)
        zip_dict = dict((a, zip_list.count(a)) for a in zip_list)
        zip_dict_sorted = dict(sorted(zip_dict.items(), key=lambda x: x[1], reverse=True))
        print("第1组偏移量：[" + str(list(zip_dict_sorted)[0][1]) + ", " + str(list(zip_dict_sorted)[0][0]) + "] num : " + str(zip_dict_sorted[list(zip_dict_sorted)[0]]))
        print("第2组偏移量：[" + str(list(zip_dict_sorted)[1][1]) + ", " + str(list(zip_dict_sorted)[1][0]) + "] num : " + str(zip_dict_sorted[list(zip_dict_sorted)[1]]))
        print("第3组偏移量：[" + str(list(zip_dict_sorted)[2][1]) + ", " + str(list(zip_dict_sorted)[2][0]) + "] num : " + str(zip_dict_sorted[list(zip_dict_sorted)[2]]))
        print("第4组偏移量：[" + str(list(zip_dict_sorted)[3][1]) + ", " + str(list(zip_dict_sorted)[3][0]) + "] num : " + str(zip_dict_sorted[list(zip_dict_sorted)[3]]))
        print("第5组偏移量：[" + str(list(zip_dict_sorted)[4][1]) + ", " + str(list(zip_dict_sorted)[4][0]) + "] num : " + str(zip_dict_sorted[list(zip_dict_sorted)[4]]))
        print("第6组偏移量：[" + str(list(zip_dict_sorted)[5][1]) + ", " + str(list(zip_dict_sorted)[5][0]) + "] num : " + str(zip_dict_sorted[list(zip_dict_sorted)[5]]))

        dx = list(zip_dict_sorted)[0][0]
        dy = list(zip_dict_sorted)[0][1]
        num = zip_dict_sorted[list(zip_dict_sorted)[0]]
        # print("dx = " + str(dx) + ", dy = " + str(dy) + ", num = " + str(num))

        if num < offsetEvaluate:
            totalStatus = False
            print(str(num))
        # self.printAndWrite("  In Mode, The number of num is " + str(num) + " and the number of offsetEvaluate is "+str(offsetEvaluate))
        return (totalStatus, [dy, dx])  # opencv中处理图像的dx dy 与习惯是相反的  所以将两者调换位置

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



if __name__ == "__main__":
    method = ImageMethod()
    img_offset = []  # 添加偏移量的图片list
    offset_data = []  # 偏移量list
    gaussion_means = []
    gaussion_sigma = []

    match_mode_num = 0
    match_offset_num = 0
    offset_num = 55

    method.delete_test_data()

    img_clear = cv2.imread('clear_img.jpeg', flags=0)  # 单通道
    img_offset, offset_data = method.add_random_offset(img_clear)
    img_noise, gaussion_means, gaussion_sigma = method.add_random_gaussian_noise(img_offset)
    clear_kps, clear_features = method.get_feature_point(img_clear, "surf")
    print(2)
    for i in range(offset_num):
        print("-------------------------------")
        print("第" + str(i+1) + "幅图片的实际偏移量为 ：" + str(offset_data[i]) + "   means : " + str(gaussion_means[i]) + " sigma : " + str(gaussion_sigma[i]))
        offset_kps_temp, offset_features_temp = method.get_feature_point(img_noise[i], "surf")
        offset_matches_temp = method.match_descriptors(clear_features, offset_features_temp, match_method="surf")
        total_status, [dx, dy] = method.get_offset_by_mode(clear_kps, offset_kps_temp, offset_matches_temp)
        if total_status:
            match_mode_num = match_mode_num + 1
        if [dx, dy] == offset_data[i]:
            match_offset_num = match_offset_num + 1
            match_result = True
        else:
            match_result = False
        print("第" + str(i+1) + "张偏移图片匹配结果：", match_result, [dx, dy])

    print("--------------------------------")
    match_mode_percentage = match_mode_num / offset_num
    print('通过众数计算结果的正确率为：{:.2%}'.format(match_mode_percentage))

    match_offset_percentage = match_offset_num / offset_num
    print('通过对比偏移量和计算结果的实际正确率为：{:.2%}'.format(match_offset_percentage))



