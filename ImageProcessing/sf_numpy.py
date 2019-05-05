import numpy as np
import cv2
import datetime


class ImageFusion:
    #CPU并行加速v1.0


    @staticmethod
    def calculate_spatial_frequency(images, block_size=41):

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
        #矩阵并行处理
        rf_last_total_pow = (last_image[1:row, 1:col].__sub__(last_image[1:row, 0:col-1])).__pow__(2)  # 向左减并平方
        cf_last_total_pow = (last_image[1:row, 1:col].__sub__(last_image[0:row-1, 1:col])).__pow__(2)  # 向上减并平方
        rf_next_total_pow = (next_image[1:row, 1:col].__sub__(next_image[1:row, 0:col-1])).__pow__(2)
        cf_next_total_pow = (next_image[1:row, 1:col].__sub__(next_image[0:row-1, 1:col])).__pow__(2)
#         print(rf_last_total_pow.shape)

        if row % block_size == 0:
            row_num = row // block_size - 1
        else:
            row_num = row // block_size
        if col % block_size == 0:
            col_num = col // block_size -1
        else:
            col_num = col // block_size

        fusion_choice = np.ones((row_num + 1, col_num + 1))

        time_2 = datetime.datetime.now()

        #下一步需要并行化的部分：
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

                sf_last = np.sum(rf_last_total_pow[(i * block_size):row_end_position, (j * block_size):col_end_position]) + np.sum(cf_last_total_pow[(i * block_size):row_end_position, (j * block_size):col_end_position])
                sf_next = np.sum(rf_next_total_pow[(i * block_size):row_end_position, (j * block_size):col_end_position]) + np.sum(cf_next_total_pow[(i * block_size):row_end_position, (j * block_size):col_end_position])

#                 if sf_last < sf_next:  # 该区域第二张图像较清楚 赋值为0
#                     fusion_choice[i, j] = 0
#                     weight_matrix[(i * block_size):row_end_position, (j * block_size):col_end_position] -= 1

#                 # 用 3 * 3 的 majority filter 过滤一遍
#                 if i > 1 and j > 1:
#                     if np.all(fusion_choice[(i - 2):(i + 1), (j - 2):(j + 1)] == choice_full_zeros):  # 取全0
#                         # print("满足010")
#                         fusion_choice[i - 1, j - 1] = 0
#                         weight_matrix[(i - 1) * block_size:i * block_size, (j - 1) * block_size:j * block_size] -= 1
#                     elif np.all(fusion_choice[(i - 2):(i + 1), (j - 2):(j + 1)] == choice_full_ones):  # 取全1
#                         # print("满足101")
#                         fusion_choice[i - 1, j - 1] = 1
#                         weight_matrix[(i - 1) * block_size:i * block_size, (j - 1) * block_size:j * block_size] += 1


                if sf_last < sf_next:  # 该区域第二张图像较清楚 赋值为0
                    fusion_choice[i, j] = 0
                    weight_matrix[(i * block_size) + 1:row_end_position + 1, (j * block_size) + 1:col_end_position + 1] -= 1

                # 用 3 * 3 的 majority filter 过滤一遍
                if i > 1 and j > 1:
                    if np.all(fusion_choice[(i - 2):(i + 1), (j - 2):(j + 1)] == choice_full_zeros):  # 取全0
                        # print("满足010")
                        fusion_choice[i - 1, j - 1] = 0
                        weight_matrix[(i - 1) * block_size + 1:i * block_size + 1, (j - 1) * block_size + 1:j * block_size + 1] -= 1
                    elif np.all(fusion_choice[(i - 2):(i + 1), (j - 2):(j + 1)] == choice_full_ones):  # 取全1
                        # print("满足101")
                        fusion_choice[i - 1, j - 1] = 1
                        weight_matrix[(i - 1) * block_size + 1:i * block_size + 1, (j - 1) * block_size + 1:j * block_size + 1] += 1
        time_3 = datetime.datetime.now()

        print("time_2 - time_1 = {}".format(time_2 - time_1))
        print("time_3 - time_2 = {}".format(time_3 - time_2))
        print("time_3 - time_1 = {}",format(time_3 - time_1))
        weight_matrix = weight_matrix.astype(np.float32)
#         weight_matrix = cv2.bilateralFilter(src=weight_matrix, d=30, sigmaColor=10, sigmaSpace=7)

        return weight_matrix

    def fuse_by_sf_and_mbb(self, images, block_size):
        """
        多分辨率样条和空间频率融合叠加,空间频率生成的权值矩阵，生成高斯金字塔然后与拉普拉斯金字塔结合，
        最后将上述金字塔生成图像
        :param images: 输入两个相同区域的图像
        :return: 融合后的图像
        """
        (last_image, next_image) = images
        last_lp, last_gp = self.get_laplacian_pyramid(last_image)
        next_lp, next_gp = self.get_laplacian_pyramid(next_image)

        start_time = datetime.datetime.now()
        weight_matrix = self.calculate_spatial_frequency(images, block_size)
        end_time = datetime.datetime.now()
        print("The time of calculating sf is {}".format(end_time - start_time))

        # wm_gp 为weight_matrix的高斯金字塔
        wm_gp = self.get_gaussian_pyramid(weight_matrix)
        fuse_lp = []
        for i in range(self.pyramid_level):
            fuse_lp.append(last_lp[i] * wm_gp[self.pyramid_level - i - 1] +
                           next_lp[i] * (1 - wm_gp[self.pyramid_level - i - 1]))
        fuse_region = np.uint8(self.reconstruct(fuse_lp))
        return fuse_region, weight_matrix

    pyramid_level = 4

    def get_gaussian_pyramid(self, input_image):
        """
        获得图像的高斯金字塔
        :param input_image:输入图像
        :return: 高斯金字塔，以list形式返回，第一个是原图，以此类推
        """

        g = input_image.copy().astype(np.float64)
        gp = [g]  # 金字塔结构存到list中
        for i in range(self.pyramid_level):
            g = cv2.pyrDown(g)
            gp.append(g)
        return gp

    def get_laplacian_pyramid(self, input_image):
        """
        求一张图像的拉普拉斯金字塔
        :param input_image: 输入图像
        :return: 拉普拉斯金字塔(laplacian_pyramid, lp, 从小到大)，高斯金字塔(gaussian_pyramid, gp,从大到小),
                  均以list形式
        """
        gp = self.get_gaussian_pyramid(input_image)
        lp = [gp[self.pyramid_level - 1]]
        for i in range(self.pyramid_level - 1, -1, -1):
            ge = cv2.pyrUp(gp[i])
            ge = cv2.resize(ge, (gp[i - 1].shape[1], gp[i - 1].shape[0]), interpolation=cv2.INTER_CUBIC)
            lp.append(cv2.subtract(gp[i - 1], ge))
        return lp, gp

    @staticmethod
    def reconstruct(input_pyramid):
        """
        根据拉普拉斯金字塔重构图像，该list第一个是最小的原图，后面是更大的拉普拉斯表示
        :param input_pyramid: 输入的金字塔
        :return: 返回重构的结果图
        """
        construct_result = input_pyramid[0]
        for i in range(1, len(input_pyramid)):
            construct_result = cv2.pyrUp(construct_result)
            construct_result = cv2.resize(construct_result, (input_pyramid[i].shape[1], input_pyramid[i].shape[0]),
                                          interpolation=cv2.INTER_CUBIC)
            construct_result = cv2.add(construct_result, input_pyramid[i])
        return construct_result

if __name__ == "__main__":

    image_fusion = ImageFusion()
    last_image = cv2.imread("test_3.jpeg", flags=0)
    next_image = cv2.imread("test_4.jpeg", flags=0)
    images = (last_image, next_image)
    block_size = 33

    start_time = datetime.datetime.now()
    image, weight = image_fusion.fuse_by_sf_and_mbb(images, block_size)
    end_time = datetime.datetime.now()

    cv2.imwrite("fuse_image_test_3_4_numpy.jpeg", image)
    print("sf计算采用部分并行的模式")
    print("The time of fusing is {}".format(end_time - start_time))
    print(image.shape)
    print(last_image.shape)
    plt.figure(figsize=(30,30))
    plt.subplot(221)
    plt.imshow(last_image, cmap="gray")
    plt.subplot(222)
    plt.imshow(next_image, cmap="gray")
    plt.subplot(223)
    plt.imshow(weight * 255, cmap="gray")
    plt.subplot(224)
    plt.imshow(image, cmap="gray")
