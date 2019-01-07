
from numpy import *
from scipy import *
import numpy as np
import cv2




def SaltAndPepper(src, percetage):
    NoiseImg = src
    NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(NoiseNum):
        randX = random.random_integers(0, src.shape[0] - 1)
        randY = random.random_integers(0, src.shape[1] - 1)
        if random.random_integers(0, 1) == 0:
            NoiseImg[randX, randY] = 0
        else:
            NoiseImg[randX, randY] = 255
    return NoiseImg


#定义添加高斯噪声的函数
def addGaussianNoise(image,percetage):
    G_Noiseimg = image
    G_NoiseNum=int(percetage*image.shape[0]*image.shape[1])
    for i in range(G_NoiseNum):
        temp_x = np.random.randint(20,40)
        temp_y = np.random.randint(20,40)
        G_Noiseimg[temp_x][temp_y] = 255
    return G_Noiseimg


if __name__ == "__main__":
    srcImage = cv2.imread("clearpicture.jpeg")
    cv2.namedWindow("Original image")
    cv2.imshow("Original image", srcImage)

    grayImage = srcImage
   # grayImage = cv2.cvtColor(srcImage, cv2.COLOR_BGR2GRAY)  # 灰度变换
   # cv2.imshow("grayimage", grayImage)

    gauss_noiseImage = addGaussianNoise(grayImage, 0.3)  # 添加10%的高斯噪声
    cv2.imshow("Add_GaussianNoise Image", gauss_noiseImage)
    cv2.imwrite("Glena.jpg ", gauss_noiseImage)

    SaltAndPepper_noiseImage = SaltAndPepper(grayImage, 0.1)  # 再添加10%的椒盐噪声
    cv2.imshow("Add_SaltAndPepperNoise Image", SaltAndPepper_noiseImage)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


