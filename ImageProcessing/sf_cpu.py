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

def get_spatial_frequency_matrix(self, images):
    """
    空间频率滤波的权值矩阵计算
    :param images: 输入两个相同区域的图像
    :return: 权值矩阵，第一张比第二张清晰的像素点为1，第二张比第一张清晰的像素点为0
    """
    block_size = 40
    (last_image, next_image) = images
    row, col = last_image.shape[0:2]
    weight_matrix = np.ones(last_image.shape)  # 全1矩阵

    if self.is_gpu_available:   # gpu模式
        pass
    else:   # cpu模式
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
                        np.ones((row_end_position - (i * block_size), col_end_position - (j * block_size)))
                else:  # 该区域第二张图像较清楚 赋值为0
                    # print("choose 0")
                    fusion_choice[i, j] = 0
                    weight_matrix[(i * block_size):row_end_position, (j * block_size):col_end_position] = \
                        np.zeros((row_end_position - (i * block_size), col_end_position - (j * block_size)))

                # 用 3 * 3 的 majority filter 过滤一遍
                if i > 1 and j > 1:
                    if np.all(fusion_choice[(i - 2):(i + 1), (j - 2):(j + 1)] == choice_full_zeros):  # 取全0
                        # print("满足010")
                        fusion_choice[i - 1, j - 1] = 0
                        weight_matrix[(i - 1) * block_size:i * block_size, (j - 1) * block_size:j * block_size] = \
                            np.zeros((block_size, block_size))
                    elif np.all(fusion_choice[(i - 2):(i + 1), (j - 2):(j + 1)] == choice_full_ones):  # 取全1
                        # print("满足101")
                        fusion_choice[i - 1, j - 1] = 1
                        weight_matrix[(i - 1) * block_size:i * block_size, (j - 1) * block_size:j * block_size] = \
                            np.ones((block_size, block_size))
    return weight_matrix
