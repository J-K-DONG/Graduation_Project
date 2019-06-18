import cv2
import os
import skimage.measure


class Method:
    # 关于 GPU 加速的设置
    is_gpu_available = False

    # 关于打印信息的设置
    input_dir = ""
    is_out_log_file = False
    log_file = "evaluate.txt"
    is_print_screen = True

    def print_and_log(self, content):
        """
        向屏幕或者txt打印信息
        :param content:
        :return:
        """
        if self.is_print_screen:
            print(content)
        if self.is_out_log_file:
            f = open(os.path.join(self.input_dir, self.log_file), "a")
            f.write(content)
            f.write("\n")
            f.close()

    @staticmethod
    def make_out_dir(dir_path):
        """
        创造一个文件夹
        :param dir_path:文件夹目录
        :return:
        """
        try:
            os.makedirs(dir_path)
        except OSError:
            pass

    @staticmethod
    def delete_files_in_folder(dir_address):
        """
        删除一个文件夹下所有文件
        :param dir_address: 文件夹目录
        :return:
        """
        file_list = os.listdir(dir_address)
        file_num = len(file_list)
        if file_num != 0:
            for i in range(file_num):
                path = os.path.join(dir_address, file_list[i])
                if os.path.isdir(path) is False:
                    os.remove(path)

    @staticmethod
    def resize_image(origin_image, resize_times, inter_method=cv2.INTER_AREA):
        """
        缩放图像
        :param origin_image:原始图像
        :param resize_times: 缩放比率
        :param inter_method: 插值方法
        :return: 缩放结果
        """
        (h, w) = origin_image.shape
        resize_h = int(h * resize_times)
        resize_w = int(w * resize_times)
        # cv2.INTER_AREA是测试后最好的方法
        resized_image = cv2.resize(origin_image, (resize_w, resize_h), interpolation=inter_method)
        return resized_image
