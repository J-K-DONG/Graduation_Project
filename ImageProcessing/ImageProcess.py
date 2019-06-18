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
import ImageUtility as Utility
# import myGpuFeatures as myGpuFeatures
import ImageFusion as ImageFusion
import glob
import time
import datetime
import pycuda


class ImageFeature:
    """
    用来保存第一张图像的特征点和特征，为后续加速配准使用
    """
    kps = None
    features = None


class ImageTrack(Utility.Method):
    """
    图像追踪类
    """
    def __init__(self, images_dir=os.getcwd(), fuse_method="notFuse", sf_gpu_available=False, feature_method="sift", search_ratio=0.75):
        # 关于录入文件的设置
        self.images_dir = images_dir
        self.image_shape = None
        self.offset_list = []  # 图像偏移的真实结果
        self.is_available_list = []
        self.images_address_list = None

        # 关于图像增强的操作
        self.is_enhance = False
        self.is_clahe = False
        self.clip_limit = 20
        self.tile_size = 5

        # 关于特征搜索的设置
        self.feature_method = feature_method    # "sift","surf" or "orb"
        self.search_ratio = search_ratio        # 0.75 is common value for matches
        self.last_image_feature = ImageFeature()  # 保存上一张图像特征，方便使用

        # 关于特征配准的设置
        self.offset_calculate = "mode"  # "mode" or "ransac"
        self.offset_evaluate = 3  # 40 means nums of matches for mode, 3.0 means  of matches for ransac

        # 关于 GPU-SURF 的设置
        self.surf_hessian_threshold = 100.0
        self.surf_n_octaves = 4
        self.surf_n_octave_layers = 3
        self.surf_is_extended = True
        self.surf_key_points_ratio = 0.01
        self.surf_is_upright = False

        # 关于 GPU-ORB 的设置
        self.orb_n_features = 5000
        self.orb_scale_factor = 1.2
        self.orb_n_levels = 8
        self.orb_edge_threshold = 31
        self.orb_first_level = 0
        self.orb_wta_k = 2
        self.orb_patch_size = 31
        self.orb_fast_threshold = 20
        self.orb_blur_for_descriptor = False
        self.orb_max_distance = 30

        # 关于融合方法的设置
        self.fuse_method = fuse_method

        # 关于sf计算
        self.sf_gpu_available = sf_gpu_available

    def generate_random_noise_and_offset(self, input_image, image_num=5):
        """
        生成随机图像，增加噪声，增加随机偏移，并将图像保存至self.images_dir，随机偏移保存至offset_data.txt
        :param input_image: 输入图像
        :param image_num: 需生成的数目
        :return: 无
        """
        # 删除文件夹和偏移量文件
        self.make_out_dir(self.images_dir)
        self.delete_files_in_folder(self.images_dir)
        if os.path.exists(os.path.join(os.getcwd(), "offset_data.txt")):
            os.remove(os.path.join(os.getcwd(), "offset_data.txt"))

        f = open('offset_data.txt', 'a')
        self.print_and_log("Generating images")
        names_list = []
        roi_length = 50

        for k in range(0, image_num + 1):
            random_image = input_image.copy()
            sigma = 0
            block_size = 0
            if k == 0:
                offset = [0, 0]
                sigma = 0
                dx_blur = 0
                dy_blur = 0
                block_size = 0
            else:
                # 增加随机偏移
                random_image, offset = self.add_random_offset(random_image)

                # 增加随机高斯噪声
                # random_image, sigma = self.add_random_gaussian_noise(random_image)

                # # 增加随机椒盐噪声
                # random_image = self.add_salt_pepper_noise(random_image)

                # 增加随机高斯模糊
                random_image, dx_blur, dy_blur, block_size = self.add_gaussian_blur(random_image)

            f.write(str(offset[0]) + "," + str(offset[1]) + "\n")

            # 裁剪roi区域，避免四周有黑色边缘
            height, width = random_image.shape
            random_image = random_image[roi_length: height - roi_length, roi_length: width - roi_length]

            # 命名并保存
            image_name = "img_" + str(k).zfill(3) + "_gaussian_noise_sigma_" + str(sigma) + "_offset_" + str(offset[0])\
                         + "_" + str(offset[1])
            image_name_blur = "_Gaussian_blur_" + str(dx_blur) + "_" + str(dy_blur) + "_" + str(block_size) + ".jpeg"
            image_name = image_name + image_name_blur
            names_list.append(image_name)
            cv2.imwrite(os.path.join(self.images_dir, image_name), random_image)
            self.print_and_log(" generate {}".format(image_name))
        f.close()
        self.print_and_log("generate done")

    @staticmethod
    def add_random_offset(input_image):
        """
        给图像增加随机位移
        :param input_image: 输入图像
        :return: 输出位移的图像
        """
        height, width = input_image.shape[0:2]
        rand_x = random.randint(-25, 25)
        rand_y = random.randint(-25, 25)
        h_matrix = np.float32([[1, 0, -rand_y], [0, 1, -rand_x]])
        random_offset_image = cv2.warpAffine(input_image, h_matrix, (width, height))
        return random_offset_image, [rand_x, rand_y]

    @staticmethod
    def add_random_gaussian_noise(input_image, sigma_range=10):
        """
        给图像增加随机高斯噪声
        :param input_image:输入图像
        :param sigma_range: 最大方差
        :return: 输出增加高斯噪声的图像
        """
        h, w = input_image.shape[0:2]
        sigma = random.randint(0, sigma_range)
        random_noise_image = input_image + np.random.randn(h, w) * sigma  # 矩阵相加
        for i in range(h):
            for j in range(w):  # 每一个点先检测像素值是否溢出  再进行赋值
                r1 = np.where((random_noise_image[i, j]) > 255, 255, (random_noise_image[i, j]))
                r2 = np.where((r1 < 0), 0, r1)
                random_noise_image[i, j] = np.round(r2)
        return random_noise_image.astype('uint8'), sigma

    @staticmethod
    def add_salt_pepper_noise(input_image, percentage=0.5):
        """
        添加椒盐噪声 (测试  不完善)(single img)
        :param input_image: 输入图像
        :param percentage: 随机点的比例
        :return:输出增加椒盐噪声的图像
        """
        random_noise_image = input_image
        noise_point_num = int(percentage * input_image.shape[0] * input_image.shape[1])  # 选取随机点的数量
        for i in range(noise_point_num):  # 遍历NoiseNum个随机点并增加随机量
            rand_x = random.randint(0, input_image.shape[0] - 1)
            rand_y = random.randint(0, input_image.shape[1] - 1)  # 选取随机点
            if random.randint(0, 1) == 0:
                random_noise_image[rand_x, rand_y] = 0  # 将该点变为黑点
            else:
                random_noise_image[rand_x, rand_y] = 255  # 将该点变为白点
        return random_noise_image

    @staticmethod
    def add_gaussian_blur(input_img):
        """
        在随机区域添加高斯模糊，首先随机生成区块大小，其次随机确定区块起始位置，再次将该区块高斯模糊（kernel=(41，41),sigma=0）
        :param input_img:输入图像
        :return:输出高斯模糊结果,模糊起始位置大小(dx,dy）,区块大小，block_size
        """
        kernel_size = (41, 41)
        block_size = random.randint(0, 200)
        sigma = 0
        row, col = input_img.shape[0:2]
        dx = random.randint(0, row - block_size)
        dy = random.randint(0, col - block_size)
        blur_block = cv2.GaussianBlur(input_img[dx:dx + block_size, dy:dy + block_size], kernel_size, sigma)
        input_img[dx:dx + block_size, dy:dy + block_size] = blur_block
        return input_img, dx, dy, block_size

    @staticmethod
    def denoise_gaussian_noise(input_img):
        """
        平滑高斯噪声
        :param input_img:输入图像
        :return:输出平滑高斯噪声的结果
        """
        kernel_size = (3, 3)
        sigma = 0
        input_img = cv2.GaussianBlur(input_img, kernel_size, sigma)
        return input_img

    def start_track_and_fuse(self):
        """
        开始追踪
        :return:返回追踪和融合结果图
        """
        self.print_and_log("Start tracking")

        self.images_address_list = glob.glob(os.path.join(self.images_dir, "*.jpeg"))
        # self.images_address_list = self.images_dir

        # match_mode_num = 0  # 通过众数计算得到的正确偏移量个数
        # match_offset_num = 0  # 计算结果与实际相同的结果个数

        # 读取gt_offset
        # gt_offset = []
        # with open("offset_data.txt", 'r') as file_to_read:
        #     while True:
        #         lines = file_to_read.readline()  # 整行读取数据
        #         if not lines:
        #             break
        #         lines = lines.split("\n")[0]
        #         gt_offset.append([int(lines.split(",")[0]), int(lines.split(",")[1])])  # 以'，'为分割符 分为前后两个字符

        # print(self.images_address_list)
        last_image = cv2.imdecode(np.fromfile(self.images_address_list[0], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        self.image_shape = last_image.shape  # 485 * 659
        # print(last_image.shape)
        last_kps, last_features = self.calculate_feature(last_image)  # 第000张图像
        self.last_image_feature.kps = last_kps
        self.last_image_feature.features = last_features  # 存为全局变量
        self.is_available_list.append(True)
        start_time = time.time()

        # 定义去噪的参数
        kernel_size = 3
        sigma = 0
        for i in range(1, len(self.images_address_list)):  # 001 开始到 055

            next_image = cv2.imdecode(np.fromfile(self.images_address_list[i], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            print(self.images_address_list[i])

            # 平滑噪声
            # next_image = cv2.GaussianBlur(next_image, (kernel_size, kernel_size), sigma)

            total_status, offset = self.calculate_offset_by_feature(next_image)
            if total_status:
                # match_mode_num = match_mode_num + 1
                self.is_available_list.append(True)
                self.offset_list.append(offset)
            else:
                self.is_available_list.append(False)
                self.offset_list.append([0, 0])
            # if offset == gt_offset[i]:
            #     match_offset_num = match_offset_num + 1
            #     match_result = True
            # else:
            #     match_result = False
            # self.print_and_log("第{}张偏移图片匹配结果：{}, 真实值：{}， 计算值：{}".format(i, match_result, gt_offset[i], offset))
        end_time = time.time()
        self.print_and_log("The time of matching is {:.3f} \'s".format(end_time - start_time))
        self.print_and_log("--------------------------------")
        # match_mode_percentage = match_mode_num / (len(self.images_address_list) - 1)
        # self.print_and_log('通过众数计算结果的正确率为：{:.2%}'.format(match_mode_percentage))
        #
        # match_offset_percentage = match_offset_num / (len(self.images_address_list) - 1)
        # self.print_and_log('通过对比偏移量和计算结果的实际正确率为：{:.2%}'.format(match_offset_percentage))

        # print(self.offset_list)
        # 拼接图像
        print("Start stitching")
        start_time = time.time()
        tracked_image = self.get_stitch_by_offset()

        # print(type(tracked_image))

        # tracked_image = tracked_image * 0.5 + cv2.imread("clear_img.jpeg", flags=0) * 0.5
        end_time = time.time()
        print("Fuse finished")
        print("The time of fusing is {:.3f} \'s".format(end_time - start_time))

        return tracked_image

    def calculate_feature(self, input_image):
        """
        计算图像特征点
        :param input_image: 输入图像
        :return: 返回特征点(kps)，及其相应特征描述符
        """
        # 判断是否有增强
        if self.is_enhance:
            if self.is_clahe:
                clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(self.tile_size, self.tile_size))
                input_image = clahe.apply(input_image)
            elif self.is_clahe is False:
                input_image = cv2.equalizeHist(input_image)
        kps, features = self.detect_and_describe(input_image)
        return kps, features

    def calculate_offset_by_feature(self, next_image):
        """
        通过全局特征匹配计算偏移量
        :param next_image: 下一张图像
        :return: 返回配准结果status和偏移量(offset = [dx,dy])
        """
        offset = [0, 0]
        status = False

        # get the feature points
        last_kps = self.last_image_feature.kps
        last_features = self.last_image_feature.features

        next_kps, next_features = self.calculate_feature(next_image)
        # if last_features is not None and next_features is not None:
        if len(last_features) > 500 and len(next_features) > 500:
            matches = self.match_descriptors(last_features, next_features)
            # match all the feature points
            if self.offset_calculate == "mode":
                (status, offset) = self.get_offset_by_mode(last_kps, next_kps, matches)
            elif self.offset_calculate == "ransac":
                (status, offset, adjustH) = self.get_offset_by_ransac(last_kps, next_kps, matches)
        else:
            return status, "there are one image have no features"
        if status is False:
            return status, "the two image have less common features"
        return status, offset

    def get_stitch_by_offset(self):
        """
        根据偏移量计算返回拼接结果
        :return: 拼接结果图像
        """
        # 如果你不细心，不要碰这段代码
        # 已优化到根据指针来控制拼接，CPU下最快了


        stitch_start = time.time()
        min_dx, min_dy = 0, 0
        result_row = self.image_shape[0]  # 拼接最终结果的横轴长度,先赋值第一个图像的横轴 695
        result_col = self.image_shape[1]  # 拼接最终结果的纵轴长度,先赋值第一个图像的纵轴 531
        self.offset_list.insert(0, [0, 0])  # 增加第一张图像相对于最终结果的原点的偏移量
        temp_offset_list = self.offset_list.copy()
        offset_list_origin = self.offset_list.copy()
        for i in range(1, len(temp_offset_list)):
            if self.is_available_list[i] is False:
                continue
            dx = self.offset_list[i][0]
            dy = self.offset_list[i][1]
            if dx <= 0:
                if dx < min_dx:
                    for j in range(0, i):
                        temp_offset_list[j][0] = temp_offset_list[j][0] + abs(dx - min_dx)  # 将之前的偏移量逐个增加
                    result_row = result_row + abs(dx - min_dx)  # 最终图像的row增加
                    min_dx = dx
                    temp_offset_list[i][0] = 0  # 将该图像的row置为最小偏移
                else:
                    temp_offset_list[i][0] = temp_offset_list[0][0] + dx
            else:
                temp_offset_list[i][0] = dx + temp_offset_list[0][0]
                result_row = max(result_row, temp_offset_list[i][0] + self.image_shape[0])
            if dy <= 0:
                if dy < min_dy:
                    for j in range(0, i):
                        temp_offset_list[j][1] = temp_offset_list[j][1] + abs(dy - min_dy)
                    result_col = result_col + abs(dy - min_dy)
                    min_dy = dy
                    temp_offset_list[i][1] = 0
                else:
                    temp_offset_list[i][1] = temp_offset_list[0][1] + dy
            else:
                temp_offset_list[i][1] = dy + temp_offset_list[0][1]
                result_col = max(result_col, temp_offset_list[i][1] + self.image_shape[1])
        # stitch_result = np.ones((result_row, result_col), np.int) * 255
        stitch_result = np.zeros((result_row, result_col), np.int) - 1
        self.offset_list = temp_offset_list
        self.print_and_log("  The rectified offsetList is " + str(self.offset_list))
        # 如上算出各个图像相对于原点偏移量，并最终计算出输出图像大小，并构造矩阵，如下开始赋值

        fuse_start = time.time()
        for i in range(0, len(self.offset_list)):
            if self.is_available_list[i] is False:
                continue
            image = cv2.imdecode(np.fromfile(self.images_address_list[i], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            self.print_and_log("  stitching " + str(self.images_address_list[i]))
            if i == 0:
                stitch_result[self.offset_list[0][0]: self.offset_list[0][0] + image.shape[0],
                              self.offset_list[0][1]: self.offset_list[0][1] + image.shape[1]] = image
            else:
                if self.fuse_method == "notFuse":
                    # 适用于无图像融合，直接覆盖
                    stitch_result[
                        self.offset_list[i][0]: self.offset_list[i][0] + image.shape[0],
                        self.offset_list[i][1]: self.offset_list[i][1] + image.shape[1]] = image
                else:
                    # 适用于图像融合算法，切出 roiA 和 roiB 供图像融合
                    roi_ltx = self.offset_list[i][0]
                    roi_lty = self.offset_list[i][1]
                    roi_rbx = self.offset_list[i][0] + image.shape[0]
                    roi_rby = self.offset_list[i][1] + image.shape[1]
                    # 从原本图像切出来感兴趣区域 last_roi_fuse_region
                    last_roi_fuse_region = stitch_result[roi_ltx:roi_rbx, roi_lty:roi_rby].copy()
                    # 将该拼接的图像赋值给 stitch_result  先将图像覆盖上去
                    stitch_result[
                        self.offset_list[i][0]: self.offset_list[i][0] + image.shape[0],
                        self.offset_list[i][1]: self.offset_list[i][1] + image.shape[1]] = image
                    # 再切出来感兴趣区域 next_roi_fuse_region
                    next_roi_fuse_region = stitch_result[roi_ltx:roi_rbx, roi_lty:roi_rby].copy()
                    # 融合后再放到该位置
                    stitch_result[roi_ltx:roi_rbx, roi_lty:roi_rby] = self.fuse_image(
                        [last_roi_fuse_region, next_roi_fuse_region],
                        [offset_list_origin[i][0], offset_list_origin[i][1]])
        stitch_result[stitch_result == -1] = 0
        fuse_end = time.time()
        print("The time of stitching is {:.3f} \'s".format(fuse_start - stitch_start))
        print("The time of fusing is {:.3f} \'s".format(fuse_end - fuse_start))
        return stitch_result.astype(np.uint8)

    def fuse_image(self, overlap_rfrs, offset):
        """
        融合两个重合区域,其中rfr代表（roi_fuse_region）
        :param overlap_rfrs:重合区域
        :param offset: 原本两图像的位移
        :return:返回融合结果
        """
        (last_rfr, next_rfr) = overlap_rfrs
        (dx, dy) = offset
        if self.fuse_method != "fadeInAndFadeOut" and self.fuse_method != "trigonometric":
            # 将各自区域中为背景的部分用另一区域填充，目的是消除背景
            # 权值为-1是为了方便渐入检出融合和三角融合计算
            last_rfr[last_rfr == -1] = 0
            next_rfr[next_rfr == -1] = 0
            last_rfr[last_rfr == 0] = next_rfr[last_rfr == 0]
            next_rfr[next_rfr == 0] = last_rfr[next_rfr == 0]
        fuse_region = np.zeros(last_rfr.shape, np.uint8)
        image_fusion = ImageFusion.ImageFusion(sf_gpu_available=self.sf_gpu_available)
        print("fusing  method is : ", self.fuse_method)
        if self.fuse_method == "notFuse":
            fuse_region = next_rfr
        elif self.fuse_method == "average":
            fuse_region = image_fusion.fuse_by_average([last_rfr, next_rfr])
        elif self.fuse_method == "maximum":
            fuse_region = image_fusion.fuse_by_maximum([last_rfr, next_rfr])
        elif self.fuse_method == "minimum":
            fuse_region = image_fusion.fuse_by_minimum([last_rfr, next_rfr])
        elif self.fuse_method == "fadeInAndFadeOut":
            fuse_region = image_fusion.fuse_by_fade_in_and_fade_out(overlap_rfrs, dx, dy)
        elif self.fuse_method == "trigonometric":
            fuse_region = image_fusion.fuse_by_trigonometric(overlap_rfrs, dx, dy)
        elif self.fuse_method == "multiBandBlending":
            fuse_region = image_fusion.fuse_by_multi_band_blending([last_rfr, next_rfr])
        elif self.fuse_method == "spatialFrequency":
            fuse_region = image_fusion.fuse_by_spatial_frequency([last_rfr, next_rfr])
        elif self.fuse_method == "spatialFrequencyAndMultiBandBlending":
            fuse_region = image_fusion.fuse_by_sf_and_mbb([last_rfr, next_rfr])
        return fuse_region

    def get_offset_by_mode(self, last_kps, next_kps, matches):
        """
        通过众数的方法求取位移
        :param last_kps: 上一张图像的特征点
        :param next_kps: 下一张图像的特征点
        :param matches: 匹配矩阵
        :return: 返回拼接结果图像
        """
        total_status = True
        if len(matches) == 0:
            total_status = False
            return total_status, "the two images have no matches"
        dx_list = []
        dy_list = []
        for trainIdx, queryIdx in matches:
            last_pt = (last_kps[queryIdx][1], last_kps[queryIdx][0])
            next_pt = (next_kps[trainIdx][1], next_kps[trainIdx][0])
            if int(last_pt[0] - next_pt[0]) == 0 and int(last_pt[1] - next_pt[1]) == 0:
                continue
            dx_list.append(int(round(last_pt[0] - next_pt[0])))
            dy_list.append(int(round(last_pt[1] - next_pt[1])))
            # dx_list.append(int(last_pt[0] - next_pt[0]))
            # dy_list.append(int(last_pt[1] - next_pt[1]))
        if len(dx_list) == 0:
            dx_list.append(0)
            dy_list.append(0)
        # Get Mode offset in [dxList, dyList], thanks for clovermini
        zipped = zip(dx_list, dy_list)
        zip_list = list(zipped)
        zip_dict = dict((a, zip_list.count(a)) for a in zip_list)
        zip_dict_sorted = dict(sorted(zip_dict.items(), key=lambda x: x[1], reverse=True))
        dx = list(zip_dict_sorted)[0][0]
        dy = list(zip_dict_sorted)[0][1]
        num = zip_dict_sorted[list(zip_dict_sorted)[0]]
        if num < self.offset_evaluate:
            total_status = False
            return total_status, "the two images have less common offset"
        else:
            return total_status, [dx, dy]

    def get_offset_by_ransac(self, last_kps, next_kps, matches):
        """
        通过ransac方法求取位移
        :param last_kps: 上一张图像的特征点
        :param next_kps: 下一张图像的特征点
        :param matches: 匹配矩阵
        :return: 返回拼接结果图像
        """
        total_status = False
        last_pts = np.float32([last_kps[i] for (_, i) in matches])
        next_pts = np.float32([next_kps[i] for (i, _) in matches])
        if len(matches) == 0:
            return total_status, [0, 0], 0
        (H, status) = cv2.findHomography(last_pts, next_pts, cv2.RANSAC, 3, 0.9)
        true_count = 0
        for i in range(0, len(status)):
            if status[i]:
                true_count = true_count + 1
        if true_count >= self.offset_evaluate:
            total_status = True
            adjust_h = H.copy()
            adjust_h[0, 2] = 0
            adjust_h[1, 2] = 0
            adjust_h[2, 0] = 0
            adjust_h[2, 1] = 0
            return total_status, [np.round(np.array(H).astype(np.int)[1, 2]) * (-1),
                                  np.round(np.array(H).astype(np.int)[0, 2]) * (-1)], adjust_h
        else:
            return total_status, [0, 0], 0

    @staticmethod
    def np_to_list_for_keypoints(array):
        """
        GPU返回numpy形式的特征点，转成list形式
        :param array:
        :return:
        """
        kps = []
        row, col = array.shape
        for i in range(row):
            kps.append([array[i, 0], array[i, 1]])
        return kps

    @staticmethod
    def np_to_list_for_matches(array):
        """
        GPU返回numpy形式的匹配对，转成list形式
        :param array:
        :return:
        """
        descriptors = []
        row, col = array.shape
        for i in range(row):
            descriptors.append((array[i, 0], array[i, 1]))
        return descriptors

    @staticmethod
    def np_to_kps_and_descriptors(array):
        """
        GPU返回numpy形式的kps，descripotrs，转成list形式
        :param array:
        :return:
        """
        kps = []
        descriptors = array[:, :, 1]
        for i in range(array.shape[0]):
            kps.append([array[i, 0, 0], array[i, 1, 0]])
        return kps, descriptors

    def detect_and_describe(self, image):
        """
        给定一张图像，求取特征点和特征描述符
        :param image: 输入图像
        :return: kps，features
        """
        descriptor = None
        kps = None
        features = None
        if self.is_gpu_available is False:  # CPU mode
            if self.feature_method == "sift":
                descriptor = cv2.xfeatures2d.SIFT_create()
            elif self.feature_method == "surf":
                descriptor = cv2.xfeatures2d.SURF_create()
            elif self.feature_method == "orb":
                descriptor = cv2.ORB_create(self.orb_n_features, self.orb_scale_factor, self.orb_n_levels,
                                            self.orb_edge_threshold, self.orb_first_level, self.orb_wta_k, 0,
                                            self.orb_patch_size, self.orb_fast_threshold)
            # 检测SIFT特征点，并计算描述子
            kps, features = descriptor.detectAndCompute(image, None)
            # 将结果转换成NumPy数组
            kps = np.float32([kp.pt for kp in kps])
        else:  # GPU mode
            if self.feature_method == "sift":
                # 目前GPU-SIFT尚未开发，先采用CPU版本的替代
                descriptor = cv2.xfeatures2d.SIFT_create()
                kps, features = descriptor.detectAndCompute(image, None)
                kps = np.float32([kp.pt for kp in kps])
            elif self.feature_method == "surf":
                kps, features = self.np_to_kps_and_descriptors(
                    myGpuFeatures.detectAndDescribeBySurf(image, self.surf_hessian_threshold,
                                                          self.surf_n_octaves, self.surf_n_octave_layers,
                                                          self.surf_is_extended, self.surf_key_points_ratio,
                                                          self.surf_is_upright))
            elif self.feature_method == "orb":
                kps, features = self.np_to_kps_and_descriptors(
                    myGpuFeatures.detectAndDescribeByOrb(image, self.orb_n_features, self.orb_scale_factor,
                                                         self.orb_n_levels, self.orb_edge_threshold,
                                                         self.orb_first_level, self.orb_wta_k, 0,
                                                         self.orb_patch_size, self.orb_fast_threshold,
                                                         self.orb_blur_for_descriptor))
        # 返回特征点集，及对应的描述特征
        return kps, features

    def match_descriptors(self, last_features, next_features):
        """
        根据两张图像的特征描述符，找到相应匹配对
        :param last_features: 上一张图像特征描述符
        :param next_features: 下一张图像特征描述符
        :return: matches
        """
        matches = None
        if self.feature_method == "surf" or self.feature_method == "sift":
            matcher = cv2.DescriptorMatcher_create("BruteForce")
            # 使用KNN检测来自A、B图的SIFT特征匹配对，K=2，返回一个列表
            raw_matches = matcher.knnMatch(last_features, next_features, 2)
            matches = []
            for m in raw_matches:
                # 当最近距离跟次近距离的比值小于ratio值时，保留此匹配对
                if len(m) == 2 and m[0].distance < m[1].distance * self.search_ratio:
                    # 存储两个点在featuresA, featuresB中的索引值
                    matches.append((m[0].trainIdx, m[0].queryIdx))
        elif self.feature_method == "orb":
            matcher = cv2.DescriptorMatcher_create("BruteForce-Hamming")
            raw_matches = matcher.match(last_features, next_features)
            matches = []
            for m in raw_matches:
                matches.append((m.trainIdx, m.queryIdx))
        # if self.isGPUAvailable == False:        # CPU Mode
        #     # 建立暴力匹配器
        #     if self.featureMethod == "surf" or self.featureMethod == "sift":
        #         matcher = cv2.DescriptorMatcher_create("BruteForce")
        #         # 使用KNN检测来自A、B图的SIFT特征匹配对，K=2，返回一个列表
        #         rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        #         matches = []
        #         for m in rawMatches:
        #         # 当最近距离跟次近距离的比值小于ratio值时，保留此匹配对
        #             if len(m) == 2 and m[0].distance < m[1].distance * self.searchRatio:
        #                 # 存储两个点在featuresA, featuresB中的索引值
        #                 matches.append((m[0].trainIdx, m[0].queryIdx))
        #     elif self.featureMethod == "orb":
        #         matcher = cv2.DescriptorMatcher_create("BruteForce-Hamming")
        #         rawMatches = matcher.match(featuresA, featuresB)
        #         matches = []
        #         for m in rawMatches:
        #             matches.append((m.trainIdx, m.queryIdx))
        #     # self.printAndWrite("  The number of matches is " + str(len(matches)))
        # else:                                   # GPU Mode
        #     if self.featureMethod == "surf":
        #         matches = self.npToListForMatches(myGpuFeatures.matchDescriptors(np.array(featuresA),
        # np.array(featuresB), 2, self.searchRatio))
        #     elif self.featureMethod == "orb":
        #         matches = self.npToListForMatches(myGpuFeatures.matchDescriptors(np.array(featuresA),
        # np.array(featuresB), 3, self.orbMaxDistance))
        return matches


if __name__ == "__main__":
    # 生成随机噪声和随机位移图像
    project_address = os.getcwd()
    tracker = ImageTrack(os.path.join(project_address, "random_images"))
    clear_image = cv2.imread("clear_img.jpeg", 0)
    tracker.generate_random_noise_and_offset(clear_image, image_num=55)
    # image_noise, sigma = tracker.add_random_gaussian_noise(clear_image, 40)
    # track = ImageTrack()
    #
    # kernel = 41
    # kernel_size = (kernel, kernel)
    # sigma = 0
    # image_blur = cv2.GaussianBlur(image, kernel_size, sigma)
    #
    # print(type(image_blur))
    # cv2.imwrite("image_noise.jpeg", image_noise)
    #
    # weight = cv2.imread("weight_4.jpeg", flags=0)
    # # weight = weight.astype(np.float32)
    # weight = cv2.bilateralFilter(src=weight, d=30, sigmaColor=10, sigmaSpace=7)
    # # weight = weight.astype(np.uint8)
    # cv2.imwrite("weight_f.jpeg", weight)




    # # 生成合成图像
    # result = np.zeros((500, 1000), dtype=np.uint8)
    # images_list = glob.glob(".\\random_images\\*jpeg")
    # for i in range(0, 50):
    #     print(i)
    #     image = cv2.imread(images_list[i], 0)
    #     row_start = (i // 10) * 100
    #     col_start = (i % 10) * 100
    #     resized_image = cv2.resize(image, (100, 100), cv2.INTER_CUBIC)
    #     result[row_start: row_start + 100, col_start: col_start + 100] = resized_image
    # cv2.imshow("result", result)
    # cv2.waitKey(0)

    # result = cv2.imread("result.jpeg", flags=0)
    # kernel_size = (15, 15)
    # result_1 = cv2.GaussianBlur(result, kernel_size, 0)
    # cv2.imwrite("result_1.jpeg", result_1)



