import cv2
import os
import skimage.measure


class Method:
    # 关于 GPU 加速的设置
    is_gpu_available = True

    # 关于打印信息的设置
    input_dir = ""
    is_out_log_file = False
    log_file = "evaluate.txt"
    is_print_screen = True

    def print_and_log(self, content):
        if self.is_print_screen:
            print(content)
        if self.is_out_log_file:
            f = open(os.path.join(self.input_dir, self.log_file), "a")
            f.write(content)
            f.write("\n")
            f.close()

    @staticmethod
    def make_out_dir(dir_path):
        try:
            os.makedirs(dir_path)
        except OSError:
            pass

    @staticmethod
    def delete_files_in_folder(dir_address):
        file_list = os.listdir(dir_address)
        file_num = len(file_list)
        if file_num != 0:
            for i in range(file_num):
                path = os.path.join(dir_address, file_list[i])
                if os.path.isdir(path) is False:
                    os.remove(path)

    @staticmethod
    def resize_image(origin_image, resize_times, inter_method=cv2.INTER_AREA):
        (h, w) = origin_image.shape
        resize_h = int(h * resize_times)
        resize_w = int(w * resize_times)
        # cv2.INTER_AREA是测试后最好的方法
        resized_image = cv2.resize(origin_image, (resize_w, resize_h), interpolation=inter_method)
        return resized_image

    @staticmethod
    def compare_result_gt(stitch_image, gt_image):
        assert stitch_image.shape == gt_image.shape, "The shape of two image is not same"
        mse_score = skimage.measure.compare_mse(stitch_image, gt_image)
        psnr_score = skimage.measure.compare_psnr(stitch_image, gt_image)
        ssim_score = skimage.measure.compare_ssim(stitch_image, gt_image)
        print(" The mse is {}, psnr is {}, ssim is {}".format(mse_score, psnr_score, ssim_score))


if __name__ == "__main__":
    # 根据图像生成视频
    image = cv2.imread("stitching_by_human.png")
    project_address = os.getcwd()
    method = Method()
    method.generate_video_from_image(image, os.path.join(project_address, "result"))
