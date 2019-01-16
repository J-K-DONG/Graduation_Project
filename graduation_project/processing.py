from numpy import *
import cv2
import random
import numpy as np


# 添加椒盐噪声
def SaltAndPepper(src, percetage):
    NoiseImg = src
    NoiseNum = int(percetage * src.shape[0] * src.shape[1])     # 选取随机点的数量
    for i in range(NoiseNum):  # 遍历NoiseNum个随机点并增加随机量
        rand_x = random.randint(0, src.shape[0] - 1)
        rand_y = random.randint(0, src.shape[1] - 1)    # 选取随机点
        if random.randint(0, 1) == 0:
            NoiseImg[rand_x, rand_y] = 0    # 将该点变为黑点
        else:
            NoiseImg[rand_x, rand_y] = 255  # 将该点变为白点
    return NoiseImg


# 添加高斯噪声
def GaussianNoise(src, means, sigma):
    NoiseImg = src
    rows = NoiseImg.shape[0]
    cols = NoiseImg.shape[1]
    for i in range(rows):
        for j in range(cols):   # 每一个点都增加高斯随机数
            NoiseImg[i, j] = NoiseImg[i, j] + random.gauss(means, sigma)
            if NoiseImg[i, j] < 0:
                NoiseImg[i, j] = 0
            elif NoiseImg[i, j] > 255:
                NoiseImg[i, j] = 255
    return NoiseImg


if __name__ == '__main__':
    clear_img_1 = cv2.imread('clear_picture.jpeg', flags=0)  # 读取原图并赋给一个对象
    cv2.imshow('clear_img', clear_img_1)    # 显示原图

# 正态分布:
    # mu = 0
    # sigma = 1

    mu = np.mean(clear_img_1)  # 取了整幅图的平均值
    sigma = 15
    GaussianImg = GaussianNoise(clear_img_1, mu, sigma)
    cv2.imshow('gaussian_sigma_15', GaussianImg)
    cv2.imwrite('gaussian_sigma_15.jpg', GaussianImg)

    clear_img_2 = cv2.imread('clear_picture.jpeg', flags=0)  # 赋给第二个对象
    SaltAndPepperImg = SaltAndPepper(clear_img_2, 0.05)
    cv2.imshow('salt&pepper + 0.05', SaltAndPepperImg)
    cv2.imwrite('salt&pepper_0.05.jpg', SaltAndPepperImg)
    cv2.waitKey()