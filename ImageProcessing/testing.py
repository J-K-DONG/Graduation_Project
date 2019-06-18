import cv2
import numpy as np


class Method:

    def start_track_and_fuse(self):
        """
        开始追踪
        :return:返回追踪和融合结果图
        """
        print("Start tracking")
        blur_block = 5
        # last_image = cv2.imread("image_blur_" + str(blur_block) + ".jpeg", flags=0)
        last_image = cv2.imread("clear_img.jpeg", flags=0)
        print(type(last_image[0][0]))

        # last_image = cv2.imread("clear_img.jpeg", flags=0)
        print("last_image row col 为", last_image.shape)
        print("-----------------------------")

        last_kps, last_features = self.calculate_feature(last_image)  # 第000张图像

        print("last_kps len : ", len(last_kps))
        print("kps type : ", type(last_kps))
        print(last_kps)
        print("kps[0] type is : ", type(last_kps[0]))
        print("kps[0] is : ", last_kps[0])
        print("-----------------------------")

        print("features len is : ", len(last_features))
        print("features type is : ", type(last_features))
        print(last_features)
        print("features[0] type is : ", type(last_features[0]))
        print("features[0] is : ")
        print(last_features[0])
        print("-----------------------------")

        next_image = cv2.imread("result.jpeg", flags=0)

        total_status, offset = self.calculate_offset_by_feature(last_kps, last_features, next_image)
        if total_status:
            print("do match")
            print(offset)
            # match_mode_num = match_mode_num + 1
        #         self.is_available_list.append(True)
        #         self.offset_list.append(offset)
        else:
            print("do not match")
        #         self.is_available_list.append(False)
        #         self.offset_list.append([0, 0])

        return offset

    def calculate_feature(self, input_image):
        """
        计算图像特征点
        :param input_image: 输入图像
        :return: 返回特征点(kps)，及其相应特征描述符
        """
        kps, features = self.detect_and_describe(input_image)
        return kps, features

    def detect_and_describe(self, image):
        """
        给定一张图像，求取特征点和特征描述符
        :param image: 输入图像
        :return: kps，features
        """
        descriptor = None
        kps = None
        features = None

        descriptor = cv2.xfeatures2d.SIFT_create()
        # 检测SIFT特征点，并计算描述子
        kps, features = descriptor.detectAndCompute(image, None)
        # 将结果转换成NumPy数组
        kps = np.float32([kp.pt for kp in kps])
        # 返回特征点集，及对应的描述特征
        return kps, features

    def calculate_offset_by_feature(self, last_kps, last_features, next_image):
        """
        通过全局特征匹配计算偏移量
        :param next_image: 下一张图像
        :return: 返回配准结果status和偏移量(offset = [dx,dy])
        """
        offset = [0, 0]
        status = False

        next_kps, next_features = self.calculate_feature(next_image)
        if len(last_features) > 500 and len(next_features) > 500:
            matches = self.match_descriptors(last_features, next_features)
            (status, offset) = self.get_offset_by_mode(last_kps, next_kps, matches)
        else:
            return status, "there are one image have no features"
        if status is False:
            return status, "the two image have less common features"
        return status, offset

    def match_descriptors(self, last_features, next_features):
        """
        根据两张图像的特征描述符，找到相应匹配对
        :param last_features: 上一张图像特征描述符
        :param next_features: 下一张图像特征描述符
        :return: matches
        """
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        # 使用KNN检测来自A、B图的SIFT特征匹配对，K=2，返回一个列表
        raw_matches = matcher.knnMatch(last_features, next_features, 2)

        print("raw_matches len is : ", len(raw_matches))
        print("raw_matches type is : ", type(raw_matches))
        print(raw_matches)
        print("raw_matches[0] type is : ", type(raw_matches[0]))
        print("raw_matches[0] is : ", raw_matches[0])
        print("raw_matches[0][0] type is : ", type(raw_matches[0][0]))
        print("min distance is : ", raw_matches[10][0].distance)
        print("second distance is : ", raw_matches[10][1].distance)
        print("train - query = ", raw_matches[10][0].distance - raw_matches[10][1].distance)
        print(((50*50+100*100)**0.5)*2)


        matches = []
        for m in raw_matches:
            # 当最近距离跟次近距离的比值小于ratio值时，保留此匹配对
            if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
                # 存储两个点在featuresA, featuresB中的索引值
                matches.append((m[0].trainIdx, m[0].queryIdx))

        return matches

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
        if num < 40:
            total_status = False
            return total_status, "the two images have less common offset"
        else:
            return total_status, [dx, dy]


if __name__ == "__main__":
    # m = Method()
    # offset = m.start_track_and_fuse()

    image_1 = cv2.imread("clear_img.jpeg", flags=0)
    image_2 = image_1[0:485, 0:609]
    cv2.imwrite("image_2.jpeg", image_2)
    image_3 = image_1[100:585, 150:759]
    cv2.imwrite("image_3.jpeg", image_3)






