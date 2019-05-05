import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import datetime
import torch.nn.functional as f
import pycuda.driver as cuda
from six.moves import range

class SFCalculate():

    def __init__(self, gpu_device=3):
        self._gpu_device = gpu_device
        print("------------------------------------")
        print("当前可用GPU共 {} 个".format(cuda.Device.count()))
        dev = cuda.Device(self._gpu_device)
        print('当前正在使用GPU编号为 : Device #%d: %s' % (self._gpu_device, dev.name()))
        print('计算能力为: %d.%d' % dev.compute_capability())
        print('总显存大小为: %s MB' % (dev.total_memory()//(1024)//(1024)))


    def calculate_spatial_frequency(self, images, block_size=5):
        block_num = block_size // 2
        (last_image, next_image) = images
        weight_matrix = np.ones(last_image.shape)

        if torch.cuda.is_available():
            # 将图像打入GPU并增加维度
            last_cuda = torch.from_numpy(last_image).float().cuda(self._gpu_device).reshape((1, 1, last_image.shape[0], last_image.shape[1]))
            next_cuda = torch.from_numpy(next_image).float().cuda(self._gpu_device).reshape((1, 1, next_image.shape[0], next_image.shape[1]))
            # 创建向右/向下平移的卷积核 + 打入GPU + 增加维度
            right_shift_kernel = torch.FloatTensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]]).cuda(self._gpu_device).reshape((1, 1, 3, 3))
            bottom_shift_kernel = torch.FloatTensor([[0, 1, 0], [0, 0, 0], [0, 0, 0]]).cuda(self._gpu_device).reshape((1, 1, 3, 3))

            last_right_shift = f.conv2d(last_cuda, right_shift_kernel, padding=1)
#             print("当前正在使用GPU编号为 : {}".format(torch.cuda.current_device()))
            last_bottom_shift = f.conv2d(last_cuda, bottom_shift_kernel, padding=1)
            next_right_shift = f.conv2d(next_cuda, right_shift_kernel, padding=1)
            next_bottom_shift = f.conv2d(next_cuda, bottom_shift_kernel, padding=1)

            last_sf = torch.pow((last_right_shift - last_cuda), 2) + torch.pow((last_bottom_shift - last_cuda), 2)
            next_sf = torch.pow((next_right_shift - next_cuda), 2) + torch.pow((next_bottom_shift - next_cuda), 2)

            add_kernel = torch.ones((block_size, block_size)).float().cuda(self._gpu_device).reshape((1, 1, block_size, block_size))
            last_sf_convolve = f.conv2d(last_sf, add_kernel, padding=block_num)
            next_sf_convolve = f.conv2d(next_sf, add_kernel, padding=block_num)

            weight_zeros = torch.zeros((last_sf_convolve.shape[2], last_sf_convolve.shape[3])).cuda(self._gpu_device)
            weight_ones = torch.ones((last_sf_convolve.shape[2], last_sf_convolve.shape[3])).cuda(self._gpu_device)
            sf_compare = torch.where(last_sf_convolve.squeeze(0).squeeze(0) > next_sf_convolve.squeeze(0).squeeze(0), weight_ones, weight_zeros)


            weight_matrix = sf_compare.cpu().numpy().astype(np.uint8)
            weight_matrix = cv2.bilateralFilter(src=weight_matrix, d=30, sigmaColor=10, sigmaSpace=7)

        return weight_matrix


if __name__ == "__main__":

    sf = SFCalculate()
    image_1 = cv2.imread("test_3.jpeg", flags=0)
    image_2 = cv2.imread("test_4.jpeg", flags=0)
    images = (image_1, image_2)
    start = datetime.datetime.now()
    weight_matrix = sf.calculate_spatial_frequency(images)
    end = datetime.datetime.now()
    print("The duration of calculating weight_matrix is : {}".format(end - start))


    image_convolve = weight_matrix * image_1 + (1 - weight_matrix) * image_2
    cv2.imwrite("test_3_4_12onvolve.jpeg", image_convolve)

    # 看效果
    weight_matrix[150, :] = 1
    weight_matrix[350, :] = 1
    weight_matrix[:, 500] = 1
    weight_matrix[:, 700] = 1

    plt.figure(figsize=(30,30))
    plt.subplot(221)
    plt.imshow(image_1, cmap="gray")
    plt.subplot(222)
    plt.imshow(image_2, cmap="gray")
    plt.subplot(223)
    plt.imshow(weight_matrix * 255, cmap="gray")
    plt.subplot(224)
    plt.imshow(weight_matrix * image_1 + (1 - weight_matrix) * image_2, cmap="gray")
    plt.show()


    
