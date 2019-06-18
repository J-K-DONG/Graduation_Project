import numpy as np
import datetime
import cv2
from scipy import signal
import torch.nn.functional as f
import torch


class ImageWeightMatrix:
    _gpu_device = 0

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

    def get_spatial_frequency_matrix_by_for(self, images, block_size=41):
        """
        空间频率滤波的权值矩阵计算
        :param images: 输入两个相同区域的图像
        :return: 权值矩阵，第一张比第二张清晰的像素点为1，第二张比第一张清晰的像素点为0
        """
        (last_image, next_image) = images
        row, col = last_image.shape[0:2]
        weight_matrix = np.ones(last_image.shape)  # 全1矩阵
        # if self.is_gpu_available:  # gpu模式
        #
        #     pass
        #
        # else:  # cpu模式
        choice_full_zeros = np.array([(0, 0, 0),
                                      (0, 1, 0),
                                      (0, 0, 0)])
        choice_full_ones = np.array([(1, 1, 1),
                                     (1, 0, 1),
                                     (1, 1, 1)])
        row_num = row // block_size
        col_num = col // block_size
        fusion_choice = np.ones((row_num + 1, col_num + 1), dtype=np.int)

        # print("图像共分为 " + str(row_num + 1) + " 行 " + str(col_num + 1) + " 列")

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
                # print("imageA的第 " + str(i + 1) + " 行 " + str(j + 1) + " 列的图像块的SF值为： ", last_image_block_sf)
                # print("imageB的第 " + str(i + 1) + " 行 " + str(j + 1) + " 列的图像块的SF值为： ", next_image_block_sf)
                if last_image_block_sf >= next_image_block_sf:  # 该区域第一张图像较清楚 权值矩阵赋值为1
                    # print("choose 1")
                    weight_matrix[(i * block_size):row_end_position, (j * block_size):col_end_position] = \
\
                        np.ones((row_end_position - (i * block_size), col_end_position - (j * block_size)))
            else:  # 该区域第二张图像较清楚 赋值为0
                # print("choose 0")
                fusion_choice[i, j] = 0
                weight_matrix[(i * block_size):row_end_position, (j * block_size):col_end_position] = \
\
                    np.zeros((row_end_position - (i * block_size), col_end_position - (j * block_size)))
        # 用 3 * 3 的 majority filter 过滤一遍
        if i > 1 and j > 1:
            if np.all(fusion_choice[(i - 2):(i + 1), (j - 2):(j + 1)] == choice_full_zeros):  # 取全0
                # print("满足010")
                fusion_choice[i - 1, j - 1] = 0
                weight_matrix[(i - 1) * block_size:i * block_size, (j - 1) * block_size:j * block_size] = \
\
                    np.zeros((block_size, block_size))
        elif np.all(fusion_choice[(i - 2):(i + 1), (j - 2):(j + 1)] == choice_full_ones):  # 取全1
            # print("满足101")
            fusion_choice[i - 1, j - 1] = 1
            weight_matrix[(i - 1) * block_size:i * block_size, (j - 1) * block_size:j * block_size] = \
\
                np.ones((block_size, block_size))

        weight_matrix = weight_matrix.astype(np.float32)

        # 双边滤波
        weight_matrix = cv2.bilateralFilter(src=weight_matrix, d=30, sigmaColor=10, sigmaSpace=7)

        return weight_matrix

    def get_spatial_frequency_matrix_by_numpy(self, images, block_size=41):
        """

        :param images:
        :return:
        """
        time_1 = datetime.datetime.now()
        (last_image, next_image) = images
        row, col = last_image.shape[0:2]
        weight_matrix = np.ones(last_image.shape)  # 全1

        choice_full_zeros = np.array([(0, 0, 0),
                                      (0, 1, 0),
                                      (0, 0, 0)])
        choice_full_ones = np.array([(1, 1, 1),
                                     (1, 0, 1),
                                     (1, 1, 1)])
        # 矩阵并行处理
        rf_last_total_pow = (last_image[1:row, 1:col].__sub__(last_image[1:row, 0:col - 1])).__pow__(2)  # 向左减并平方
        cf_last_total_pow = (last_image[1:row, 1:col].__sub__(last_image[0:row - 1, 1:col])).__pow__(2)  # 向上减并平方
        rf_next_total_pow = (next_image[1:row, 1:col].__sub__(next_image[1:row, 0:col - 1])).__pow__(2)
        cf_next_total_pow = (next_image[1:row, 1:col].__sub__(next_image[0:row - 1, 1:col])).__pow__(2)
        #         print(rf_last_total_pow.shape)

        if row % block_size == 0:
            row_num = row // block_size - 1
        else:
            row_num = row // block_size
        if col % block_size == 0:
            col_num = col // block_size - 1
        else:
            col_num = col // block_size

        fusion_choice = np.ones((row_num + 1, col_num + 1))

        time_2 = datetime.datetime.now()

        # 下一步需要并行化的部分：
        for i in range(row_num + 1):
            for j in range(col_num + 1):
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

                sf_last = np.sum(
                    rf_last_total_pow[(i * block_size):row_end_position, (j * block_size):col_end_position]) + np.sum(cf_last_total_pow[(i * block_size):row_end_position, (j * block_size):col_end_position])
                sf_next = np.sum(
                    rf_next_total_pow[(i * block_size):row_end_position, (j * block_size):col_end_position]) + np.sum(cf_next_total_pow[(i * block_size):row_end_position, (j * block_size):col_end_position])

                if sf_last < sf_next:  # 该区域第二张图像较清楚 赋值为0
                    fusion_choice[i, j] = 0
                    weight_matrix[(i * block_size):row_end_position, (j * block_size):col_end_position] -= 1

                # 用 3 * 3 的 majority filter 过滤一遍
                if i > 1 and j > 1:
                    if np.all(fusion_choice[(i - 2):(i + 1), (j - 2):(j + 1)] == choice_full_zeros):  # 取全0
                        # print("满足010")
                        fusion_choice[i - 1, j - 1] = 0
                        weight_matrix[(i - 1) * block_size:i * block_size, (j - 1) * block_size:j * block_size] -= 1
                    elif np.all(fusion_choice[(i - 2):(i + 1), (j - 2):(j + 1)] == choice_full_ones):  # 取全1
                        # print("满足101")
                        fusion_choice[i - 1, j - 1] = 1
                        weight_matrix[(i - 1) * block_size:i * block_size, (j - 1) * block_size:j * block_size] += 1

        time_3 = datetime.datetime.now()
        weight_matrix = weight_matrix.astype(np.float32)

        # 双边滤波
        weight_matrix = cv2.bilateralFilter(src=weight_matrix, d=30, sigmaColor=10, sigmaSpace=7)

        print("fusing time is {}".format(time_3 - time_1))
        return weight_matrix

    def get_spatial_frequency_matrix_by_signal(self, images, block_size=41):
        """

        :param images:
        :param block_size:
        :return:
        """
        (last_image, next_image) = images
        # weight_matrix = np.ones(last_image.shape)

        right_shift_kernel = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
        bottom_shift_kernel = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
        last_right_shift = signal.correlate2d(last_image, right_shift_kernel, boundary='symm', mode='same')
        last_bottom_shift = signal.correlate2d(last_image, bottom_shift_kernel, boundary='symm', mode='same')
        next_right_shift = signal.correlate2d(next_image, right_shift_kernel, boundary='symm', mode='same')
        next_bottom_shift = signal.correlate2d(next_image, bottom_shift_kernel, boundary='symm', mode='same')
        #     print(last_right_shift - last_image)
        last_sf = np.power(last_right_shift - last_image, 2) + np.power(last_bottom_shift - last_image, 2)
        next_sf = np.power(next_right_shift - next_image, 2) + np.power(next_bottom_shift - next_image, 2)
        # print(last_sf)
        add_kernel = np.ones((block_size, block_size))
        last_sf_convolve = signal.correlate2d(last_sf, add_kernel, boundary='symm', mode='same')
        next_sf_convolve = signal.correlate2d(next_sf, add_kernel, boundary='symm', mode='same')
        sf_compare = np.where(last_sf_convolve > next_sf_convolve, 1, 0)
        weight_matrix = sf_compare

        # 过滤
        kernel_full_zeros = np.array([(0, 0, 0),
                                      (0, 1, 0),
                                      (0, 0, 0)])
        kernel_full_ones = np.array([(1, 1, 1),
                                     (1, 0, 1),
                                     (1, 1, 1)])
        weight_matrix_full_zeros = signal.correlate2d(weight_matrix, kernel_full_zeros, boundary='symm', mode='same')
        weight_matrix_full_ones = signal.correlate2d(weight_matrix, kernel_full_ones, boundary='symm', mode='same')
        weight_matrix[weight_matrix_full_zeros == 1] = 0
        weight_matrix[weight_matrix_full_ones == 8] = 1
        weight_matrix = weight_matrix.astype(np.float32)

        # 双边滤波
        weight_matrix = cv2.bilateralFilter(src=weight_matrix, d=30, sigmaColor=10, sigmaSpace=7)
        return weight_matrix

    def get_spatial_frequency_matrix_by_pytorch(self, images, block_size=39):
        block_num = block_size // 2
        (last_image, next_image) = images
        weight_matrix = np.ones(last_image.shape)

        if torch.cuda.is_available():
            # 将图像打入GPU并增加维度
            last_cuda = torch.from_numpy(last_image).float().cuda(self._gpu_device).reshape(
                (1, 1, last_image.shape[0], last_image.shape[1]))
            next_cuda = torch.from_numpy(next_image).float().cuda(self._gpu_device).reshape(
                (1, 1, next_image.shape[0], next_image.shape[1]))
            # 创建向右/向下平移的卷积核 + 打入GPU + 增加维度
            right_shift_kernel = torch.FloatTensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]]).cuda(
                self._gpu_device).reshape((1, 1, 3, 3))
            bottom_shift_kernel = torch.FloatTensor([[0, 1, 0], [0, 0, 0], [0, 0, 0]]).cuda(
                self._gpu_device).reshape((1, 1, 3, 3))

            last_right_shift = f.conv2d(last_cuda, right_shift_kernel, padding=1)
            # print("当前正在使用GPU编号为 : {}".format(torch.cuda.current_device()))
            last_bottom_shift = f.conv2d(last_cuda, bottom_shift_kernel, padding=1)
            next_right_shift = f.conv2d(next_cuda, right_shift_kernel, padding=1)
            next_bottom_shift = f.conv2d(next_cuda, bottom_shift_kernel, padding=1)

            last_sf = torch.pow((last_right_shift - last_cuda), 2) + torch.pow((last_bottom_shift - last_cuda), 2)
            next_sf = torch.pow((next_right_shift - next_cuda), 2) + torch.pow((next_bottom_shift - next_cuda), 2)

            add_kernel = torch.ones((block_size, block_size)).float().cuda(self._gpu_device).reshape(
                (1, 1, block_size, block_size))
            last_sf_convolve = f.conv2d(last_sf, add_kernel, padding=block_num)
            next_sf_convolve = f.conv2d(next_sf, add_kernel, padding=block_num)

            weight_zeros = torch.zeros((last_sf_convolve.shape[2], last_sf_convolve.shape[3])).cuda(
                self._gpu_device)
            weight_ones = torch.ones((last_sf_convolve.shape[2], last_sf_convolve.shape[3])).cuda(self._gpu_device)
            sf_compare = torch.where(
                last_sf_convolve.squeeze(0).squeeze(0) > next_sf_convolve.squeeze(0).squeeze(0), weight_ones,
                weight_zeros)

            weight_matrix = sf_compare.cpu().numpy().astype(np.float32)
            # print(weight_matrix)

            # # 双边滤波
            weight_matrix = cv2.bilateralFilter(src=weight_matrix, d=30, sigmaColor=10, sigmaSpace=7)
            # print(weight_matrix)

        return weight_matrix
