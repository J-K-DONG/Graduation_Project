from scipy import signal
import matplotlib.pyplot as plt
import datetime
def calculate_spatial_frequency(images, block_size = 15):
    (last_image, next_image) = images
    weight_matrix = np.ones(last_image.shape)

    right_shift_kernel  = np.array([[0,0,0],[1,0,0],[0,0,0]])
    bottom_shift_kernel = np.array([[0,1,0],[0,0,0],[0,0,0]])
    last_right_shift = signal.correlate2d(last_image,right_shift_kernel,boundary='symm',mode='same')
    last_bottom_shift = signal.correlate2d(last_image,bottom_shift_kernel,boundary='symm',mode='same')
    next_right_shift = signal.correlate2d(next_image,right_shift_kernel,boundary='symm',mode='same')
    next_bottom_shift = signal.correlate2d(next_image,bottom_shift_kernel,boundary='symm',mode='same')
#     print(last_right_shift - last_image)
    last_sf = np.power(last_right_shift - last_image, 2) + np.power(last_bottom_shift - last_image, 2)
    next_sf = np.power(next_right_shift - next_image, 2) + np.power(next_bottom_shift - next_image, 2)
    print(last_sf)

    add_kernel = np.ones((block_size, block_size))

    last_sf_convolve = signal.correlate2d(last_sf,add_kernel,boundary='symm',mode='same')
    next_sf_convolve = signal.correlate2d(next_sf,add_kernel,boundary='symm',mode='same')

    sf_compare = np.where(last_sf_convolve > next_sf_convolve, 1, 0)
    weight_matrix = sf_compare


#     row, col = last_image.shape[0:2]
#     if row % block_size == 0:
#         row_num = row // block_size - 1
#     else:
#         row_num = row // block_size
#     if col % block_size == 0:
#         col_num = col // block_size -1
#     else:
#         col_num = col // block_size

#     kernel_length = int((block_size - 1) / 2)
#     for i in range(row_num + 1):
#         for j in range(col_num + 1):  # 图像切片比较
#             if i < row_num and j < col_num:
#                 row_end_position = (i + 1) * block_size
#                 col_end_position = (j + 1) * block_size
#             elif i < row_num and j == col_num:
#                 row_end_position = (i + 1) * block_size
#                 col_end_position = col
#             elif i == row_num and j < col_num:
#                 row_end_position = row
#                 col_end_position = (j + 1) * block_size
#             if sf_compare[(i+1) * kernel_length - 1, (j + 1) * kernel_length - 1] == 0: # 注意！
#                 weight_matrix[(i * block_size):row_end_position, (j * block_size):col_end_position] -= 0
    # 过滤
    kernel_full_zeros = np.array([(0, 0, 0),
                                  (0, 1, 0),
                                  (0, 0, 0)])
    kernel_full_ones  =  np.array([(1, 1, 1),
                                   (1, 0, 1),
                                   (1, 1, 1)])
    weight_matrix_full_zeros = signal.correlate2d(weight_matrix,kernel_full_zeros,boundary='symm',mode='same')
    weight_matrix_full_ones = signal.correlate2d(weight_matrix,kernel_full_ones,boundary='symm',mode='same')
    weight_matrix[weight_matrix_full_zeros == 1] = 0
    weight_matrix[weight_matrix_full_ones == 8] = 1

    # 双边滤波
    weight_matrix = weight_matrix.astype(np.float32)
    weight_matrix_bilateralFilter = cv2.bilateralFilter(src=weight_matrix, d=30, sigmaColor=10, sigmaSpace=7)
    sf_compare = sf_compare.astype(np.float32)
#     sf_compare = cv2.bilateralFilter(src=sf_compare, d=30, sigmaColor=10, sigmaSpace=7)


#     plt.figure(figsize=(30,30))
#     plt.subplot(221)
#     plt.imshow(sf_compare * 255, cmap="gray")
#     plt.subplot(222)
#     plt.imshow((weight_matrix) * 255, cmap="gray")
#     plt.subplot(223)
#     plt.imshow(sf_compare_bliateralFilter * 255, cmap="gray")

#     plt.subplot(224)
#     plt.imshow(weight_matrix_bilateralFilter * 255, cmap="gray")

    return sf_compare


if __name__ == "__main__":

    last_image = cv2.imread("test_3.jpeg", flags=0)
    next_image = cv2.imread("test_4.jpeg", flags=0)
    images = (last_image, next_image)
    start_time = datetime.datetime.now()
    weight = calculate_spatial_frequency(images)
    end_time = datetime.datetime.now()
    image_convolve = weight * last_image + (1 - weight) * next_image
    cv2.imwrite("test_3_4_convolve.jpeg", image_convolve)

    print("The time of fusing is {}".format(end_time - start_time))
    plt.figure(figsize=(30,30))
    plt.subplot(221)
    plt.imshow(last_image, cmap="gray")
    plt.subplot(222)
    plt.imshow(next_image, cmap="gray")
    plt.subplot(223)
    plt.imshow(weight * 255, cmap="gray")
    plt.subplot(224)
    plt.imshow(weight * last_image + (1 - weight) * next_image, cmap="gray")
    plt.show()
    cv2.imwrite("fuse_result.jpg", weight * last_image + (1 - weight) * next_image)
