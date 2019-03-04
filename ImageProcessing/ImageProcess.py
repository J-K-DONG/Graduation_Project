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
import myGpuFeatures as myGpuFeatures
import ImageFusion as ImageFusion
import glob
import time


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
    def __init__(self, images_dir, fuse_method="notFuse", feature_method="sift", search_ratio=0.75):
        # 关于录入文件的设置
        self.images_dir = images_dir
        self.image_shape = None
        self.offset_list = []
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
        self.offset_evaluate = 3  # 40 menas nums of matches for mode, 3.0 menas  of matches for ransac

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
        offset_list = []
        sigma_list = []
        roi_length = 50

        for k in range(0, image_num + 1):
            random_image = input_image.copy()
            if k == 0:
                offset = [0, 0]
                sigma = 0
            else:
                # 增加随机位移
                random_image, offset = self.add_random_offset(random_image)

                # 增加随机高斯噪声
                random_image, sigma = self.add_random_gaussian_noise(random_image)

                # # 增加随机椒盐噪声
                # random_image = self.add_salt_pepper_noise(random_image)

            offset_list.append(offset)
            sigma_list.append(sigma)
            f.write(str(offset[0]) + "," + str(offset[1]) + "\n")

            # 裁剪roi区域，避免四周有黑色边缘
            height, width = random_image.shape
            random_image = random_image[roi_length: height - roi_length, roi_length: width - roi_length]

            # 命名并保存
            image_name = "img_" + str(k).zfill(3) + "_gaussian_noise_sigma_" + str(sigma) + \
                         "_offset_" + str(offset[0]) + "_" + str(offset[1]) + ".jpeg"
            names_list.append(image_name)
            cv2.imwrite(os.path.join(self.images_dir, image_name), random_image)
            self.print_and_log(" generate {}".format(image_name))
        f.close()
        self.print_and_log("generate done")

    def start_track_and_fuse(self):
        """
        开始追踪
        :return:返回追踪和融合结果图
        """
        self.print_and_log("Start tracking")
        self.images_address_list = glob.glob(os.path.join(self.images_dir, "*.jpeg"))
        match_mode_num = 0  # 通过众数计算得到的正确偏移量个数
        match_offset_num = 0  # 计算结果与实际相同的结果个数

        # 读取gt_offset
        gt_offset = []
        with open("offset_data.txt", 'r') as file_to_read:
            while True:
                lines = file_to_read.readline()  # 整行读取数据
                if not lines:
                    break
                lines = lines.split("\n")[0]
                gt_offset.append([int(lines.split(",")[0]), int(lines.split(",")[1])])
        last_image = cv2.imdecode(np.fromfile(self.images_address_list[0], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        self.image_shape = last_image.shape
        last_kps, last_features = self.calculate_feature(last_image)
        self.last_image_feature.kps = last_kps
        self.last_image_feature.features = last_features
        self.is_available_list.append(True)
        start_time = time.time()
        for i in range(1, len(self.images_address_list)):
            next_image = cv2.imdecode(np.fromfile(self.images_address_list[i], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            total_status, offset = self.calculate_offset_by_feature(next_image)
            if total_status:
                match_mode_num = match_mode_num + 1
                self.is_available_list.append(True)
                self.offset_list.append(offset)
            else:
                self.is_available_list.append(False)
                self.offset_list.append([0, 0])
            if offset == gt_offset[i]:
                match_offset_num = match_offset_num + 1
                match_result = True
            else:
                match_result = False
            self.print_and_log("第{}张偏移图片匹配结果：{}, 真实值：{}， 计算值：{}".format(i, match_result, gt_offset[i], offset))
        end_time = time.time()
        self.print_and_log("The time of matching is {:.3f} \'s".format(end_time - start_time))
        self.print_and_log("--------------------------------")
        match_mode_percentage = match_mode_num / (len(self.images_address_list) - 1)
        self.print_and_log('通过众数计算结果的正确率为：{:.2%}'.format(match_mode_percentage))

        match_offset_percentage = match_offset_num / (len(self.images_address_list) - 1)
        self.print_and_log('通过对比偏移量和计算结果的实际正确率为：{:.2%}'.format(match_offset_percentage))

        # 拼接图像
        print("Start stitching")
        start_time = time.time()
        tracked_image = self.get_stitch_by_offset()
        end_time = time.time()
        print("The time of fusing is {:.3f} \'s".format(end_time - start_time))
        return tracked_image

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
    def add_random_gaussian_noise(input_image, sigma_range=60):
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
    # def add_random_gaussian_noise1(self, image):
    #     """
    #     添加高斯随机噪声
    #     :param image: 图像list（55张）
    #     :return: image_noise: 添加高斯噪声后的图像list     gaussian_sigma: 高斯噪声的方差list
    #     """
    #     image_noise = image  # 读进来是uint8类型
    #     gaussian_sigma = []
    #     if not os.path.exists(self.image_noise_dir):
    #         os.mkdir(self.image_noise_dir)
    #     for k in range(self.image_num):
    #         sigma = random.randint(0, 10)
    #         # print("第" + str(k + 1) + "张图片的方差为 : " + str(sigma))
    #         gaussian_sigma.append(sigma)
    #         rows, cols = image_noise[k].shape[0:2]
    #         image_noise[k] = image_noise[k] + np.random.randn(573, 759) * sigma  # 矩阵相加
    #         for i in range(rows):
    #             for j in range(cols):  # 每一个点先检测像素值是否溢出  再进行赋值
    #                 r1 = np.where((image_noise[k][i, j]) > 255, 255, (image_noise[k][i, j]))
    #                 r2 = np.where((r1 < 0), 0, r1)
    #                 image_noise[k][i, j] = np.round(r2)
    #         image_noise_temp = image_noise[k].astype('uint8')  # 将 ndarray.float64 转换为 uint8
    #         image_noise[k] = image_noise_temp
    #         image_path = self.image_noise_dir + "img_" + str(k+1) + "_gaussian_noise_sigma_" + str(sigma) + ".jpeg"
    #         cv2.imwrite(image_path, image_noise[k])
    #     return image_noise, gaussian_sigma



    # def add_random_offset1(self, image):
    #     """
    #     为多张图像添加随机偏移 并保存偏移的图片和每张图片的偏移量
    #     :param image: 图像list
    #     :return:  img_offset:添加偏移量的图像list   offset_data:所有图像的偏移量list [[rand_x, rand_y],[rand_x, rand_y]...]
    #     """
    #     rows, cols = image.shape[0:2]
    #     img_offset = []  # 添加偏移后的图片列表
    #     offset_data = []  # 55张图片的偏移量列表
    #     offset_xy = []
    #     for i in range(self.image_num):
    #         rand_x = random.randint(-25, 25)
    #         rand_y = random.randint(-25, 25)
    #         f.write(str(rand_x) + "," + str(rand_y) + "\n")
    #         offset_xy.append(rand_x)
    #         offset_xy.append(rand_y)
    #         offset_data.append(offset_xy[2*i:2*i+2])  # 将切片加入偏移量列表
    #         h = np.float32([[1, 0, rand_x], [0, 1, rand_y]])
    #         img_offset.append(cv2.warpAffine(image, h, (cols, rows)))
    #         image_path = self.image_offset_dir + "img_" + str(i+1) + "_offset_" + str(rand_x) + "_" + str(rand_y) + ".jpeg"
    #         cv2.imwrite(image_path, img_offset[i])
    #     cv2.waitKey()
    #     return img_offset, offset_data

    # def load_images(self, file_path):
    #     """
    #     读取文件路径下多张图像
    #     :param file_path: 读入文件的相对路径
    #     :return: image_list: 多张图像的list
    #     """
    #     image_list = []
    #     if os.path.exists(file_path):
    #         for img_name in os.listdir(file_path):
    #             image_dir = file_path + str(img_name)
    #             image_read = cv2.imread(image_dir, flags=0)
    #             image_list.append(image_read)
    #     return image_list

    # def delete_test_data(self):
    #     """
    #     删除上次运行的结果文件
    #     :return:
    #     """
    #     if os.path.exists(self.image_offset_dir):
    #         shutil.rmtree(self.image_offset_dir)
    #     if os.path.exists(self.image_noise_dir):
    #         shutil.rmtree(self.image_noise_dir)
    #     if os.path.exists(self.image_offset_txt):
    #         os.remove(self.image_offset_txt)
    #     print("清除上次测试数据")

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
        if last_features is not None and next_features is not None:
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
        min_dx, min_dy = 0, 0
        result_row = self.image_shape[0]  # 拼接最终结果的横轴长度,先赋值第一个图像的横轴
        result_col = self.image_shape[1]  # 拼接最终结果的纵轴长度,先赋值第一个图像的纵轴
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
                        temp_offset_list[j][0] = temp_offset_list[j][0] + abs(dx - min_dx)
                    result_row = result_row + abs(dx - min_dx)
                    min_dx = dx
                    temp_offset_list[i][0] = 0
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
                    # 将该拼接的图像赋值给 stitch_result
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
        image_fusion = ImageFusion.ImageFusion()
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
