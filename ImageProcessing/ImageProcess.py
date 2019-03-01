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
import datetime
import ImageProcessing.ImageUtility as Utility
import ImageProcessing.myGpuFeatures as myGpuFeatures
import ImageProcessing.ImageFusion as ImageFusion
import glob
import time


class ImageMethod():

    # image_noise_dir = "image_noise/"
    # image_offset_dir = "image_offset/"
    match_method = "surf"

    image_offset_txt = "offset_data.txt"

    # 关于特征搜索的设置
    feature_method = "surf"

    # 关于特征配准的设置
    offsetEvaluate = 3
    searchRatio = 0.75  # 0.75 is common value for matches

    # 关于融合方法的设置
    fuse_method = "notFuse"

    # 关于 GPU 加速的设置
    isGPUAvailable = False

    # 关于 GPU-SURF 的设置
    surfHessianThreshold = 100.0
    surfNOctaves = 4
    surfNOctaveLayers = 3
    surfIsExtended = True
    surfKeypointsRatio = 0.01
    surfIsUpright = False

    # 关于 GPU-ORB 的设置
    orbNfeatures = 5000
    orbScaleFactor = 1.2
    orbNlevels = 8
    orbEdgeThreshold = 31
    orbFirstLevel = 0
    orbWTA_K = 2
    orbPatchSize = 31
    orbFastThreshold = 20
    orbBlurForDescriptor = False
    orbMaxDistance = 30

    def add_random_gaussian_noise(self, image):
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


    def add_salt_pepper_noise(self, src, percetage):
        """
        添加椒盐噪声 (测试  不完善)(single img)
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
        rows, cols = image.shape[0:2]
        img_offset = []  # 添加偏移后的图片列表
        offset_data = []  # 55张图片的偏移量列表
        offset_xy = []
        if not os.path.exists(self.image_offset_dir):
            os.mkdir(self.image_offset_dir)
        f = open('offset_data.txt', 'a')
        for i in range(self.image_num):
            rand_x = random.randint(-25, 25)
            rand_y = random.randint(-25, 25)
            f.write(str(rand_x) + "," + str(rand_y) + "\n")
            offset_xy.append(rand_x)
            offset_xy.append(rand_y)
            offset_data.append(offset_xy[2*i:2*i+2])  # 将切片加入偏移量列表
            h = np.float32([[1, 0, rand_x], [0, 1, rand_y]])
            img_offset.append(cv2.warpAffine(image, h, (cols, rows)))
            image_path = self.image_offset_dir + "img_" + str(i+1) + "_offset_" + str(rand_x) + "_" + str(rand_y) + ".jpeg"
            cv2.imwrite(image_path, img_offset[i])
        cv2.waitKey()
        return img_offset, offset_data


    def get_feature_point(self, image):
        """
        特征点检测&描述
        :param image: 检测图像
        :return:  kps:该图像的特征   features:图像特征点的描述符
        """
        if self.isGPUAvailable == False: # CPU mode
            if self.feature_method == "sift":
                descriptor = cv2.xfeatures2d.SIFT_create()
            elif self.feature_method == "surf":
                descriptor = cv2.xfeatures2d.SURF_create()
            elif self.feature_method == "orb":
                descriptor = cv2.ORB_create(self.orbNfeatures, self.orbScaleFactor, self.orbNlevels, self.orbEdgeThreshold, self.orbFirstLevel, self.orbWTA_K, 0, self.orbPatchSize, self.orbFastThreshold)
            # 检测SIFT特征点，并计算描述子
            kps, features = descriptor.detectAndCompute(image, None)
            # 将结果转换成NumPy数组
            kps = np.float32([kp.pt for kp in kps])
        else:                           # GPU mode
            if self.feature_method == "sift":
                # 目前GPU-SIFT尚未开发，先采用CPU版本的替代
                descriptor = cv2.xfeatures2d.SIFT_create()
                kps, features = descriptor.detectAndCompute(image, None)
                kps = np.float32([kp.pt for kp in kps])
            elif self.feature_method == "surf":
                kps, features = self.np_to_kp_and_descriptors(myGpuFeatures.detectAndDescribeBySurf(image, self.surfHessianThreshold, self.surfNOctaves,self.surfNOctaveLayers, self.surfIsExtended, self.surfKeypointsRatio, self.surfIsUpright))
            elif self.feature_method == "orb":
                kps, features = self.np_to_kp_and_descriptors(myGpuFeatures.detectAndDescribeByOrb(image, self.orbNfeatures, self.orbScaleFactor, self.orbNlevels, self.orbEdgeThreshold, self.orbFirstLevel, self.orbWTA_K, 0, self.orbPatchSize, self.orbFastThreshold, self.orbBlurForDescriptor))
        # 返回特征点集，及对应的描述特征
        return (kps, features)


    def np_to_kp_and_descriptors(self, array):
        """
        功能：将GPU返回的numpy数据转成相应格式的kps和descriptors
        :param array:
        :return:
        """
        kps = []
        descriptors = array[:, :, 1]
        for i in range(array.shape[0]):
            kps.append([array[i, 0, 0], array[i, 1, 0]])
        return (kps, descriptors)


    def match_descriptors(self, featuresA, featuresB):
        '''
        功能：匹配特征点
        :param featuresA: 第一张图像的特征点描述符
        :param featuresB: 第二张图像的特征点描述符
        :return:返回匹配的对数matches
        '''
        # 建立暴力匹配器
        if self.match_method == "surf" or self.match_method == "sift":
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


    def get_offset_by_mode(self, kpsA, kpsB, matches):
        """
        功能：通过求众数的方法获得偏移量
        :param kpsA: 第一张图像的特征（原图）
        :param kpsB: 第二张图像的特征（测试图片）
        :param matches: 配准列表
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
            dxList.append(int(round(ptB[0] - ptA[0])))  # 像素值四舍五入
            dyList.append(int(round(ptB[1] - ptA[1])))
        if len(dxList) == 0:
            dxList.append(0)
            dyList.append(0)
        # Get Mode offset in [dxList, dyList], thanks for clovermini
        zipped = zip(dxList, dyList)
        zip_list = list(zipped)
        zip_dict = dict((a, zip_list.count(a)) for a in zip_list)
        zip_dict_sorted = dict(sorted(zip_dict.items(), key=lambda x: x[1], reverse=True))
        # 输出几组偏移量观察一下
        # print("第1组偏移量：[" + str(list(zip_dict_sorted)[0][1]) + ", " + str(list(zip_dict_sorted)[0][0]) + "] num : " + str(zip_dict_sorted[list(zip_dict_sorted)[0]]))
        # print("第2组偏移量：[" + str(list(zip_dict_sorted)[1][1]) + ", " + str(list(zip_dict_sorted)[1][0]) + "] num : " + str(zip_dict_sorted[list(zip_dict_sorted)[1]]))
        # print("第3组偏移量：[" + str(list(zip_dict_sorted)[2][1]) + ", " + str(list(zip_dict_sorted)[2][0]) + "] num : " + str(zip_dict_sorted[list(zip_dict_sorted)[2]]))
        # print("第4组偏移量：[" + str(list(zip_dict_sorted)[3][1]) + ", " + str(list(zip_dict_sorted)[3][0]) + "] num : " + str(zip_dict_sorted[list(zip_dict_sorted)[3]]))
        # print("第5组偏移量：[" + str(list(zip_dict_sorted)[4][1]) + ", " + str(list(zip_dict_sorted)[4][0]) + "] num : " + str(zip_dict_sorted[list(zip_dict_sorted)[4]]))
        # print("第6组偏移量：[" + str(list(zip_dict_sorted)[5][1]) + ", " + str(list(zip_dict_sorted)[5][0]) + "] num : " + str(zip_dict_sorted[list(zip_dict_sorted)[5]]))

        dx = list(zip_dict_sorted)[0][0]
        dy = list(zip_dict_sorted)[0][1]
        num = zip_dict_sorted[list(zip_dict_sorted)[0]]
        # print("dx = " + str(dx) + ", dy = " + str(dy) + ", num = " + str(num))

        if num < self.offsetEvaluate:
            totalStatus = False
            print(str(num))
        # self.printAndWrite("  In Mode, The number of num is " + str(num) + " and the number of offsetEvaluate is "+str(offsetEvaluate))
        return (totalStatus, [dy, dx])  # opencv中处理图像的dx dy 与习惯是相反的  所以将两者调换位置


    def get_stitch_by_offset(self, fileList, is_image_available, offsetListOrigin):
        '''
        通过偏移量列表和文件列表得到最终的拼接结果
        :param fileList: 图像列表
        :param offsetListOrigin: 偏移量列表
        :return: ndaarry，图像
        '''
        # 如果你不细心，不要碰这段代码
        # 已优化到根据指针来控制拼接，CPU下最快了
        dxSum = dySum = 0
        imageList = []
        # imageList.append(cv2.imread(fileList[0], 0))
        imageList.append(cv2.imdecode(np.fromfile(fileList[0], dtype=np.uint8), cv2.IMREAD_GRAYSCALE))
        resultRow = imageList[0].shape[0]  # 拼接最终结果的横轴长度,先赋值第一个图像的横轴
        resultCol = imageList[0].shape[1]  # 拼接最终结果的纵轴长度,先赋值第一个图像的纵轴

        rangeX = [[0, 0] for x in range(len(offsetListOrigin))]  # 主要用于记录X方向最大最小边界
        rangeY = [[0, 0] for x in range(len(offsetListOrigin))]  # 主要用于记录Y方向最大最小边界
        offsetList = offsetListOrigin.copy()
        rangeX[0][1] = imageList[0].shape[0]
        rangeY[0][1] = imageList[0].shape[1]

        for i in range(1, len(offsetList)):
            if is_image_available[i] is False:
                continue
            # self.printAndWrite("  stitching " + str(fileList[i]))
            # 适用于流形拼接的校正,并更新最终图像大小
            # tempImage = cv2.imread(fileList[i], 0)
            tempImage = cv2.imdecode(np.fromfile(fileList[i], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            # dxSum = dxSum + offsetList[i][0]
            # dySum = dySum + offsetList[i][1]
            dxSum = offsetList[i][0]
            dySum = offsetList[i][1]
            # self.printAndWrite("  The dxSum is " + str(dxSum) + " and the dySum is " + str(dySum))
            if dxSum <= 0:
                for j in range(0, i):
                    offsetList[j][0] = offsetList[j][0] + abs(dxSum)
                    rangeX[j][0] = rangeX[j][0] + abs(dxSum)
                    rangeX[j][1] = rangeX[j][1] + abs(dxSum)
                resultRow = resultRow + abs(dxSum)
                rangeX[i][1] = resultRow
                dxSum = rangeX[i][0] = offsetList[i][0] = 0
            else:
                offsetList[i][0] = dxSum
                resultRow = max(resultRow, dxSum + tempImage.shape[0])
                rangeX[i][1] = resultRow
            if dySum <= 0:
                for j in range(0, i):
                    offsetList[j][1] = offsetList[j][1] + abs(dySum)
                    rangeY[j][0] = rangeY[j][0] + abs(dySum)
                    rangeY[j][1] = rangeY[j][1] + abs(dySum)
                resultCol = resultCol + abs(dySum)
                rangeY[i][1] = resultCol
                dySum = rangeY[i][0] = offsetList[i][1] = 0
            else:
                offsetList[i][1] = dySum
                resultCol = max(resultCol, dySum + tempImage.shape[1])
                rangeY[i][1] = resultCol
            imageList.append(tempImage)
        stitchResult = np.zeros((resultRow, resultCol), np.int) - 1
        print("  The rectified offsetList is " + str(offsetList))

        # 如上算出各个图像相对于原点偏移量，并最终计算出输出图像大小，并构造矩阵，如下开始赋值
        for i in range(0, len(offsetList)):
            print("  stitching " + str(fileList[i]))
            if i == 0:
                stitchResult[offsetList[0][0]: offsetList[0][0] + imageList[0].shape[0],
                offsetList[0][1]: offsetList[0][1] + imageList[0].shape[1]] = imageList[0]
            else:
                if self.fuse_method == "notFuse":
                    # 适用于无图像融合，直接覆盖
                    # self.printAndWrite("Stitch " + str(i+1) + "th, the roi_ltx is " + str(offsetList[i][0]) + " and the roi_lty is " + str(offsetList[i][1]))
                    stitchResult[offsetList[i][0]: offsetList[i][0] + imageList[i].shape[0],
                    offsetList[i][1]: offsetList[i][1] + imageList[i].shape[1]] = imageList[i]
                else:
                    # 适用于图像融合算法，切出 roiA 和 roiB 供图像融合
                    minOccupyX = rangeX[i - 1][0]
                    maxOccupyX = rangeX[i - 1][1]
                    minOccupyY = rangeY[i - 1][0]
                    maxOccupyY = rangeY[i - 1][1]
                    # self.printAndWrite("Stitch " + str(i + 1) + "th, the offsetList[i][0] is " + str(
                    #     offsetList[i][0]) + " and the offsetList[i][1] is " + str(offsetList[i][1]))
                    # self.printAndWrite("Stitch " + str(i + 1) + "th, the minOccupyX is " + str(
                    #     minOccupyX) + " and the maxOccupyX is " + str(maxOccupyX) + " and the minOccupyY is " + str(
                    #     minOccupyY) + " and the maxOccupyY is " + str(maxOccupyY))
                    roi_ltx = max(offsetList[i][0], minOccupyX)
                    roi_lty = max(offsetList[i][1], minOccupyY)
                    roi_rbx = min(offsetList[i][0] + imageList[i].shape[0], maxOccupyX)
                    roi_rby = min(offsetList[i][1] + imageList[i].shape[1], maxOccupyY)
                    # self.printAndWrite("Stitch " + str(i + 1) + "th, the roi_ltx is " + str(
                    #     roi_ltx) + " and the roi_lty is " + str(roi_lty) + " and the roi_rbx is " + str(
                    #     roi_rbx) + " and the roi_rby is " + str(roi_rby))
                    roiImageRegionA = stitchResult[roi_ltx:roi_rbx, roi_lty:roi_rby].copy()
                    stitchResult[offsetList[i][0]: offsetList[i][0] + imageList[i].shape[0],
                    offsetList[i][1]: offsetList[i][1] + imageList[i].shape[1]] = imageList[i]
                    roiImageRegionB = stitchResult[roi_ltx:roi_rbx, roi_lty:roi_rby].copy()
                    stitchResult[roi_ltx:roi_rbx, roi_lty:roi_rby] = self.fuseImage([roiImageRegionA, roiImageRegionB],
                                                                                    offsetListOrigin[i][0],
                                                                                    offsetListOrigin[i][1])
        stitchResult[stitchResult == -1] = 0
        return stitchResult.astype(np.uint8)


    def fuseImage(self, images, dx, dy):
        (imageA, imageB) = images
        # cv2.namedWindow("A", 0)
        # cv2.namedWindow("B", 0)
        # cv2.imshow("A", imageA.astype(np.uint8))
        # cv2.imshow("B", imageB.astype(np.uint8))
        fuseRegion = np.zeros(imageA.shape, np.uint8)
        # imageA[imageA == 0] = imageB[imageA == 0]
        # imageB[imageB == 0] = imageA[imageB == 0]
        imageFusion = ImageFusion.ImageFusion()
        if self.fuse_method == "notFuse":
            imageB[imageA == -1] = imageB[imageA == -1]
            imageA[imageB == -1] = imageA[imageB == -1]
            fuseRegion = imageB
        elif self.fuse_method == "average":
            imageA[imageA == 0] = imageB[imageA == 0]
            imageB[imageB == 0] = imageA[imageB == 0]
            imageA[imageA == -1] = 0
            imageB[imageB == -1] = 0
            fuseRegion = imageFusion.fuseByAverage([imageA, imageB])
        elif self.fuse_method == "maximum":
            imageA[imageA == 0] = imageB[imageA == 0]
            imageB[imageB == 0] = imageA[imageB == 0]
            imageA[imageA == -1] = 0
            imageB[imageB == -1] = 0
            fuseRegion = imageFusion.fuseByMaximum([imageA, imageB])
        elif self.fuse_method == "minimum":
            imageA[imageA == 0] = imageB[imageA == 0]
            imageB[imageB == 0] = imageA[imageB == 0]
            imageA[imageA == -1] = 0
            imageB[imageB == -1] = 0
            fuseRegion = imageFusion.fuseByMinimum([imageA, imageB])
        elif self.fuse_method == "fadeInAndFadeOut":
            fuseRegion = imageFusion.fuseByFadeInAndFadeOut(images, dx, dy)
        elif self.fuse_method == "trigonometric":
            fuseRegion = imageFusion.fuseByTrigonometric(images, dx, dy)
        elif self.fuse_method == "multiBandBlending":
            imageA[imageA == 0] = imageB[imageA == 0]
            imageB[imageB == 0] = imageA[imageB == 0]
            imageA[imageA == -1] = 0
            imageB[imageB == -1] = 0
            # imageA = imageA.astye(np.uint8);  imageB = imageB.astye(np.uint8);
            fuseRegion = imageFusion.fuseByMultiBandBlending([imageA, imageB])
        elif self.fuse_method == "spatialFrequency":
            fuseRegion = imageFusion.fuseBySpatialFrequency([imageA, imageB])
        elif self.fuse_method == "optimalSeamLine":
            fuseRegion = imageFusion.fuseByOptimalSeamLine(images, self.direction)
        return fuseRegion

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


    def calculate_random_offset(self):
        """

        :return:
        """
        match_mode_num = 0  # 通过众数计算得到的正确偏移量个数
        match_offset_num = 0  # 计算结果与实际相同的结果个数

        # img_clear = cv2.imread('clear_img.jpeg', flags=0)  # 单通道
        # img_offset, offset_data = self.add_random_offset(img_clear)
        # img_noise, gaussian_sigma = self.add_random_gaussian_noise(img_offset)

        images_address = glob.glob(os.path.join(os.path.join(os.getcwd(), "image_noise"), "*jpeg"))
        img_noise = []
        for i in range(0, len(images_address)):
            img = cv2.imread(images_address[i], 0)
            img_noise.append(img)

        offset_data = []
        offset_data.append([0, 0])
        with open(self.image_offset_txt, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline()  # 整行读取数据
                if not lines:
                    break
                lines = lines.split("\n")[0]
                offset_data.append([int(lines.split(",")[0]), int(lines.split(",")[1])])

        imageA = img_noise[0]
        kpsA, featuresA = self.get_feature_point(imageA)
        offset_list = []
        offset_list.append([0, 0])
        is_image_available = []
        is_image_available.append(True)

        for i in range(1, len(img_noise)):
            print("-------------------------------")
            print("第" + str(i) + "幅图片的实际偏移量为 ：" + str(offset_data[i]))
            kpsB, featuresB = self.get_feature_point(img_noise[i])
            offset_matches_temp = self.match_descriptors(featuresA, featuresB)
            total_status, [dx, dy] = self.get_offset_by_mode(kpsA, kpsB, offset_matches_temp)
            if total_status:
                match_mode_num = match_mode_num + 1
                is_image_available.append(True)
                offset_list.append([dx, dy])
            else:
                is_image_available.append(False)
                offset_list.append([0, 0])

            if [dx, dy] == offset_data[i]:
                match_offset_num = match_offset_num + 1
                match_result = True
            else:
                match_result = False
            print("第" + str(i) + "张偏移图片匹配结果：", match_result, [dx, dy])
        print(len(images_address))
        print(images_address)
        print(len(offset_list))
        print(offset_list)
        print(len(is_image_available))
        print(is_image_available)

        print("--------------------------------")
        match_mode_percentage = match_mode_num / (len(img_noise) - 1)
        print('通过众数计算结果的正确率为：{:.2%}'.format(match_mode_percentage))

        match_offset_percentage = match_offset_num / (len(img_noise) - 1)
        print('通过对比偏移量和计算结果的实际正确率为：{:.2%}'.format(match_offset_percentage))

        # 拼接图像
        print("start stitching")
        start_time = time.time()
        stitch_image = self.get_stitch_by_offset(images_address, is_image_available, offset_list)
        end_time = time.time()
        print("The time of fusing is {:.3f} \'s".format(end_time - start_time))
        return stitch_image

if __name__ == "__main__":
    starttime = datetime.datetime.now()
    method = ImageMethod()
    stitch_image = method.calculate_random_offset()
    cv2.imwrite("result.jpeg", stitch_image)
    endtime = datetime.datetime.now()
    print(endtime - starttime)


