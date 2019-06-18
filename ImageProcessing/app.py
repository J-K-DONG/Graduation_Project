from PyQt5 import QtCore, QtGui, QtWidgets
import os
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QDialog
from Canvas import Canvas
from UiMain import UiMainWindow
from UiDialogProgressBar import UiDialog
import cv2
import ImageUtility as Utility
import ImageProcess as Process
import sys
import torch
import pycuda.driver as cuda
# import glob
import time
import numpy as np

# 全局变量
sf_gpu_available = False
images_path = []
fuse_method = "notFuse"


class ImageProcessing(QtWidgets.QMainWindow, UiMainWindow, Utility.Method, QDialog):

    def __init__(self, parent=None):
        super(QDialog, self).__init__(parent=parent)
        self.setupUi(self)

        self.listWidget.clear()

        # 图像绝对地址 名称 当前图像编号
        self.pic_path_ls = []
        self.pic_name_ls = []
        self.pic_current_index = 0
        self.setWindowIcon(QIcon('ico.png'))

        # 菜单栏
        self.picture_import.triggered.connect(self.load_image_ls)
        self.save_result.triggered.connect(self.saveFileDialog)

        self.mean_denoising.triggered.connect(self.mean_denoising_method)  # 均值去噪
        self.gauss_denoising.triggered.connect(self.gauss_denoising_method)  # 高斯去噪
        self.multi_band_blending_fusing.triggered.connect(self.fuse_by_multi_band_blending)  # 空间金字塔融合
        self.sf_and_mbb_fusing.triggered.connect(self.fuse_by_sf_and_mbb)  # 深度融合
        self.sf_fusing.triggered.connect(self.fuse_by_spatial_frequency)

        # 左边
        self.left_keep.hide()
        self.left_viewer = Canvas(self.layoutWidget)
        self.left_layout.addWidget(self.left_viewer)
        self.left_viewer.setMinimumSize(QtCore.QSize(150, 0))
        self.left_viewer.side = 'left'
        self.left_img = None  # pixMap object

        # 右边
        self.right_keep.hide()
        self.layoutWidget.update()
        self.right_viewer = Canvas(self.layoutWidget)
        self.right_layout.addWidget(self.right_viewer)
        self.right_viewer.setMinimumSize(QtCore.QSize(150, 0))
        self.right_viewer.side = 'right'
        self.right_img = None

        # 双向绑定
        self.right_viewer.make_connection(self.left_viewer)
        self.left_viewer.make_connection(self.right_viewer)

        # 列表右键
        self.contextMenu = QtWidgets.QMenu(self)
        self.contextDeleteAction = QtWidgets.QAction(u'删除', self)
        self.contextMenu.addAction(self.contextDeleteAction)
        self.listWidgetItemsToDelete = []

        self.listWidget.clicked.connect(self.check_item_clicked)
        self.listWidget.customContextMenuRequested.connect(self.show_context_menu)
        self.contextDeleteAction.triggered.connect(self.remove_selected)

        # self.fuse_method = "notFuse"
        # GPU判断
        if torch.cuda.is_available():
            global sf_gpu_available
            sf_gpu_available = True
            self._gpu_device = 0
            print("GPU情况：可用")
        else:
            print("GPU情况：不可用")
        print(torch.cuda.is_available())
        print("------------------------------------")
        print("当前可用GPU共 {} 个".format(cuda.Device.count()))
        dev = cuda.Device(self._gpu_device)
        print('当前正在使用GPU编号为 : Device #%d: %s' % (self._gpu_device, dev.name()))
        print('计算能力为: %d.%d' % dev.compute_capability())
        print('总显存大小为: %s MB' % (dev.total_memory()//(1024)//(1024)))

        # 处理完成的信号  绑定显示操作
        thread.sig_show.connect(self.show_result)

    def show_context_menu(self, pos):
        """
        列表操作
        :param pos:
        :return:
        """
        print(pos)
        items = self.listWidget.selectedIndexes()
        print(items)
        if not items:
            return
        self.listWidgetItemsToDelete = []
        for i in items:
            self.listWidgetItemsToDelete.append(i.row())
        self.contextMenu.show()
        self.contextMenu.exec_(QtGui.QCursor.pos())

    def remove_selected(self):
        """
        删除选定的图像
        :return:
        """
        for i in self.listWidgetItemsToDelete:
            self.listWidget.removeItemWidget(self.listWidget.takeItem(i))
        self.pic_path_ls = [j for i, j in enumerate(self.pic_path_ls) if i not in self.listWidgetItemsToDelete]
        self.pic_name_ls = [j for i, j in enumerate(self.pic_name_ls) if i not in self.listWidgetItemsToDelete]
        if self.pic_current_index in self.listWidgetItemsToDelete and self.pic_path_ls:
            self.pic_current_index = 0
            img_path = self.pic_path_ls[self.pic_current_index]

            self.left_img = QtGui.QPixmap(img_path)
            self.left_viewer.setPhoto(self.left_img)
            self.listWidgetItemsToDelete = []

        if not self.pic_path_ls:
            self.left_viewer.setPhoto()
            self.right_viewer.setPhoto()

    def check_item_clicked(self, pos):
        """
        点击选择图像作为当前处理对象
        :param pos:
        :return:
        """
        self.pic_current_index = pos.row()
        self.display_image()

    def refresh_list_box(self):
        """
        刷新图像列表
        :return:
        """
        self.listWidget.clear()
        self.listWidget.addItems(self.pic_name_ls)

    def load_image_ls(self):
        """
        导入图像集
        :return:
        """
        files = self.get_filenames()  # list
        if not files:
            return
        self.pic_path_ls = files
        self.pic_name_ls = [os.path.split(i)[-1] for i in files]
        self.refresh_list_box()
        self.pic_current_index = 0  # 当前图像编号

        img_path = self.pic_path_ls[self.pic_current_index]

        self.left_img = QtGui.QPixmap(img_path)
        self.left_viewer.setPhoto(self.left_img)
        # self.right_img = None
        #
        # self.right_viewer.setPhoto(self.right_img)

    def display_image(self):
        """
        显示图像
        :return:
        """
        img_path = self.pic_path_ls[self.pic_current_index]
        self.left_img = QtGui.QPixmap(img_path)

        self.right_img = None
        self.left_viewer.setPhoto(self.left_img)
        #self.right_viewer.setPhoto(self.right_img)

    def get_filenames(self):
        """
        读入图像集的名称
        :return:
        """
        options = QtWidgets.QFileDialog.Options()
        #   options |= QtWidgets.QFileDialog.DontUseNativeDialog
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "导入图集", "",
                                                          "JEPG Files (*.jpeg);;All Files (*.*)", options=options)
        return files

    def saveFileDialog(self):
        """
        保存图像处理结果
        :return
        """
        print('save picture')
        if not self.right_viewer.hasPhoto():
            return
        options = QtWidgets.QFileDialog.Options()
        #options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self,"图像保存", "",
                                                            "All Files (*);;JPG Files (*.jpg)", options=options)
        if fileName:
            print(fileName)

    def mean_denoising_method(self):
        """
        均值去噪 处理三通道图像
        :return:
        """
        self.left_viewer.fitInView()
        img_path = self.pic_path_ls[self.pic_current_index]
        img1 = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img = cv2.fastNlMeansDenoisingColored(img1, None, 6, 10, 7, 21)
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        image = QtGui.QImage(img, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
        self.right_img = QtGui.QPixmap(image)
        self.right_viewer.setPhoto(self.right_img)
        print('denoising finished')

    def gauss_denoising_method(self):
        """
        高斯模糊去平滑高斯噪声
        :return:
        """
        self.left_viewer.fitInView()
        img_path = self.pic_path_ls[self.pic_current_index]
        img1 = cv2.imread(img_path, flags=0)

        img = cv2.GaussianBlur(img1, (5, 5), 0)
        (height, width) = img.shape
        bytes_per_line = width
        image = QtGui.QImage(img, width, height, bytes_per_line, QtGui.QImage.Format_Indexed8)
        self.right_img = QtGui.QPixmap(image)
        self.right_viewer.setPhoto(self.right_img)
        print('denoising finished')

    def fuse_by_spatial_frequency(self):
        global fuse_method
        fuse_method = "spatialFrequency"
        self.track_and_fuse()

    def fuse_by_multi_band_blending(self):
        global fuse_method
        fuse_method = "multiBandBlending"
        self.track_and_fuse()

    def fuse_by_sf_and_mbb(self):
        global fuse_method
        fuse_method = "spatialFrequencyAndMultiBandBlending"
        self.track_and_fuse()

    def track_and_fuse(self):
        """
        纠偏及融合的封装方法
        :param fuse_method:
        :return:
        """
        global images_path
        images_path = self.pic_path_ls
        print("start processing")
        thread.start()
        stitchProcessForm.show()

        # while(thread.isRunning()):
        #     continue

    def show_result(self):
        """
        显示图像序列融合处理的结果
        :return:
        """
        # print(tracked_image)
        tracked_image = cv2.imread("result.jpeg", flags=0)
        height, width = tracked_image.shape[0:2]
        bytes_per_line = width
        # print(tracked_image.shape[0:2])

        tracked_image = QtGui.QImage(tracked_image, width, height, bytes_per_line, QtGui.QImage.Format_Indexed8)

        self.right_img = QtGui.QPixmap(tracked_image)
        self.right_viewer.setPhoto(self.right_img)
        print('finish processing')


class StitchAndFuseThread(QThread, Process.ImageTrack):
    """
    图像处理线程类
    """
    sig_track = pyqtSignal(int)
    sig_correct = pyqtSignal(str)
    sig_fuse = pyqtSignal(int, str)
    sig_value = pyqtSignal(str)
    # sig_length = pyqtSignal(int)
    sig_end = pyqtSignal(str, str)
    sig_show = pyqtSignal(str)

    def __init__(self):
        super(QThread, self).__init__()

    def run(self):
        """
        整个处理过程的线程函数 分为三部分：1.特征提取与配准 2.图像校准 3.图像融合
        :return:
        """
        # 特征配准开始计时
        track_start = time.time()

        # 获取全局变量
        self.fuse_method = fuse_method
        # print(self.fuse_method)
        self.sf_gpu_available = sf_gpu_available
        # print(self.sf_gpu_available)
        self.images_address_list = images_path
        # print(self.images_address_list)

        # 置空 清除上一次处理的数据
        self.offset_list = []
        self.print_and_log("Start tracking")

        # 返回特征点提取和配准过程开始 以及 第一阶段的任务数量 信号
        self.sig_track.emit(len(self.images_address_list))

        # 处理第一张图像的特征点
        self.sig_value.emit("特征提取与匹配中...")
        last_image = cv2.imdecode(np.fromfile(self.images_address_list[0], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        self.image_shape = last_image.shape  # 485 * 659
        # print(last_image.shape)
        last_kps, last_features = self.calculate_feature(last_image)  # 第000张图像
        self.last_image_feature.kps = last_kps
        self.last_image_feature.features = last_features  # 存为全局变量D
        self.is_available_list.append(True)

        # 发出处理过程中任务总数的信号
        # self.sig_length.emit((len(self.images_address_list) - 1) * 2)
        for i in range(1, len(self.images_address_list)):  # 001 开始到 055
            # 发出 ”开始进行新的特征提取和匹配处理任务“ 信号
            self.sig_value.emit("特征提取与匹配中...")
            next_image = cv2.imdecode(np.fromfile(self.images_address_list[i], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            print(self.images_address_list[i])
            total_status, offset = self.calculate_offset_by_feature(next_image)
            if total_status:
                # match_mode_num = match_mode_num + 1
                self.is_available_list.append(True)
                self.offset_list.append(offset)
            else:
                self.is_available_list.append(False)
                self.offset_list.append([0, 0])

        correct_start = time.time()
        track_time = correct_start - track_start

        # 拼接图像
        self.sig_correct.emit('%.3f' % track_time)
        print("Start stitching")
        min_dx, min_dy = 0, 0
        result_row = self.image_shape[0]  # 拼接最终结果的横轴长度,先赋值第一个图像的横轴 695
        result_col = self.image_shape[1]  # 拼接最终结果的纵轴长度,先赋值第一个图像的纵轴 531
        self.offset_list.insert(0, [0, 0])  # 增加第一张图像相对于最终结果的原点的偏移量
        temp_offset_list = self.offset_list.copy()
        offset_list_origin = self.offset_list.copy()
        for i in range(1, len(temp_offset_list)):
            # thread.sig_value.emit("图像校准中...")
            if self.is_available_list[i] is False:
                continue
            dx = self.offset_list[i][0]
            dy = self.offset_list[i][1]
            if dx <= 0:
                if dx < min_dx:
                    for j in range(0, i):
                        temp_offset_list[j][0] = temp_offset_list[j][0] + abs(dx - min_dx)  # 将之前的偏移量逐个增加
                    result_row = result_row + abs(dx - min_dx)  # 最终图像的row增加
                    min_dx = dx
                    temp_offset_list[i][0] = 0  # 将该图像的row置为最小偏移
                else:
                    temp_offset_list[i][0] = temp_offset_list[0][0] + dx
            else:
                temp_offset_list[i][0] = dx + temp_offset_list[0][0]
                result_row = max(result_row, temp_offset_list[i][0] + self.image_shape[0])
            if dy <= 0:
                if dy < min_dy:
                    for j in range(0, i):
                        temp_offset_list[j][1] = temp_offset_list[j][1] + abs(dy - min_dy)
                    result_col = result_col + abs(dy - min_dy)
                    min_dy = dy
                    temp_offset_list[i][1] = 0
                else:
                    temp_offset_list[i][1] = temp_offset_list[0][1] + dy
            else:
                temp_offset_list[i][1] = dy + temp_offset_list[0][1]
                result_col = max(result_col, temp_offset_list[i][1] + self.image_shape[1])
        # stitch_result = np.ones((result_row, result_col), np.int) * 255
        stitch_result = np.zeros((result_row, result_col), np.int) - 1
        self.offset_list = temp_offset_list
        self.print_and_log("  The rectified offsetList is " + str(self.offset_list))
        print("The shape of result is : ", result_col, result_row)

        fuse_start = time.time()
        correct_time = fuse_start - correct_start

        # 如上算出各个图像相对于原点偏移量，并最终计算出输出图像大小，并构造矩阵，如下开始赋值
        # 开始融合
        self.sig_fuse.emit(len(self.offset_list), '%.3f' % correct_time)
        # print(len(self.offset_list))
        print("start fusing")
        for i in range(0, len(self.offset_list)):
            self.sig_value.emit("图像融合中...")
            if self.is_available_list[i] is False:
                continue
            image = cv2.imdecode(np.fromfile(self.images_address_list[i], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            self.print_and_log("  stitching " + str(self.images_address_list[i]))
            if i == 0:
                stitch_result[self.offset_list[0][0]: self.offset_list[0][0] + image.shape[0],
                self.offset_list[0][1]: self.offset_list[0][1] + image.shape[1]] = image
            else:
                if self.fuse_method == "notFuse":
                    # 适用于无图像融合，直接覆盖
                    stitch_result[
                    self.offset_list[i][0]: self.offset_list[i][0] + image.shape[0],
                    self.offset_list[i][1]: self.offset_list[i][1] + image.shape[1]] = image
                else:
                    # 适用于图像融合算法，切出 roiA 和 roiB 供图像融合
                    roi_ltx = self.offset_list[i][0]
                    roi_lty = self.offset_list[i][1]
                    roi_rbx = self.offset_list[i][0] + image.shape[0]
                    roi_rby = self.offset_list[i][1] + image.shape[1]
                    # 从原本图像切出来感兴趣区域 last_roi_fuse_region
                    last_roi_fuse_region = stitch_result[roi_ltx:roi_rbx, roi_lty:roi_rby].copy()
                    # 将该拼接的图像赋值给 stitch_result  先将图像覆盖上去
                    stitch_result[
                    self.offset_list[i][0]: self.offset_list[i][0] + image.shape[0],
                    self.offset_list[i][1]: self.offset_list[i][1] + image.shape[1]] = image
                    # 再切出来感兴趣区域 next_roi_fuse_region
                    next_roi_fuse_region = stitch_result[roi_ltx:roi_rbx, roi_lty:roi_rby].copy()
                    # 融合后再放到该位置
                    stitch_result[roi_ltx:roi_rbx, roi_lty:roi_rby] = self.fuse_image(
                        [last_roi_fuse_region, next_roi_fuse_region],
                        [offset_list_origin[i][0], offset_list_origin[i][1]])
        stitch_result[stitch_result == -1] = 0
        tracked_image = stitch_result.astype(np.uint8)
        cv2.imwrite("result.jpeg", tracked_image)

        process_end = time.time()
        fuse_time = process_end - fuse_start
        # 返回处理完成的信号
        self.sig_end.emit('%.3f' % fuse_time, '%.3f' % (process_end - track_start))
        self.sig_show.emit(str(tracked_image.shape))
        print("-----------------debug signal--------------------")
        self.print_and_log("Fuse finished")
        self.print_and_log("--------------------------------")
        self.print_and_log("The time of tracking is {:.3f} \'s".format(track_time))
        self.print_and_log("The time of correcting is {:.3f} \'s".format(correct_time))
        self.print_and_log("The time of fusing is {:.3f} \'s".format(fuse_time))
        self.print_and_log("The time of processing is {:.3f} \'s".format(process_end - track_start))

        print("process finished")


class StitchProcessWindow(QDialog, UiDialog):
    # 拼接过程展示界面
    def __init__(self, parent=None):
        super(QDialog, self).__init__(parent)
        self.setupUi(self)

        self.setFixedSize(self.width(), self.height())
        self.setWindowTitle("眼底OCT图像融合过程")
        self.setWindowIcon(QIcon('icon.png'))
        self.val = 0
        self.track_length = 0
        self.fuse_length = 0
        self.progressBar.setValue(0)
        self.max_val = 0

        # 信号绑定对应操作
        thread.sig_track.connect(self.set_track_start)
        thread.sig_correct.connect(self.set_correct_start)
        thread.sig_fuse.connect(self.set_fuse_start)
        thread.sig_value.connect(self.set_val)
        thread.sig_end.connect(self.set_end)
        # thread.sig_length.connect(self.set_length)

        # 完成按钮
        self.bt_backMainWindow.clicked.connect(self.back)
        # self.terminate_processing.clicked.connect(self.back)

        # 处理完成前不能响应
        self.bt_backMainWindow.setEnabled(False)
        # self.terminate_processing.setEnabled(True)

    def set_track_start(self, track_length):
        """
        设置开始特征点配准信号对应的显示操作
        :return:
        """
        self.track_length = track_length
        self.progressBar.setMaximum((track_length - 1) * 2)
        self.lb_stitchInf.setText("处理开始！")
        self.label_track_time.setText("正在统计图像特征配准过程耗时...")

    def set_correct_start(self, track_time):
        """
        设置开始图像校准信号对应的显示操作
        :return:
        """
        self.label_track_process.setText("图像特征点提取和匹配已完成!")
        self.label_track_time.setText("图像特征点配准过程耗时: " + track_time + " 秒")
        self.label_correct_process.setText("正在进行图像校准...")
        self.label_correct_time.setText("正在统计图像校准过程耗时...")

    def set_fuse_start(self, fuse_length, correct_time):
        """
        设置开始图像融合信号对应的显示操作
        :return:
        """
        self.fuse_length = fuse_length
        self.label_correct_process.setText("图像校准已完成！")
        self.label_correct_time.setText("图像校准过程耗时: " + correct_time + " 秒")
        self.label_fuse_time.setText("正在统计图像融合过程耗时...")

    def set_val(self, process):
        """
        设置当前处理的序号
        :param process:
        :return:
        """
        # 显示拼接进度和进度条
        self.val = self.val + 1
        self.progressBar.setValue(self.val)
        self.lb_stitchInf.setText(process)
        if self.val < self.track_length:
            self.label_track_process.setText("正在处理图像特征配准过程中第（ " + str(self.val) + " / " + str(self.track_length) + " ）张图像...")
        else:
            self.label_fuse_process.setText("正在处理图像融合过程中第（ " + str(self.val - self.track_length) + " / " + str(self.fuse_length) + " ）张图像...")

    # def set_length(self, length):
    #     """
    #     设置整个处理过程中总体的量
    #     :param length:
    #     :return:
    #     """
    #     self.progressBar.setMaximum(length)

    def set_end(self, fuse_time, process_time):
        """
        处理过程已完成
        :return:
        """
        self.lb_stitchInf.setText("处理完成！")
        self.label_fuse_process.setText("图像融合已完成！")
        self.label_fuse_time.setText("图像融合过程耗时: " + fuse_time + " 秒")
        self.label_process_time.setText("总耗时: " + process_time + " 秒")
        self.bt_backMainWindow.setEnabled(True)
        # self.terminate_processing.setEnabled(False)

    def back(self):
        """
        释放此线程
        :return:
        """
        self.val = 0
        self.progressBar.setValue(0)
        self.track_length = 0
        self.fuse_length = 0
        self.label_track_process.setText("")
        self.label_track_time.setText("")
        self.label_correct_process.setText("")
        self.label_correct_time.setText("")
        self.label_fuse_process.setText("")
        self.label_fuse_time.setText("")
        self.label_process_time.setText("")
        self.bt_backMainWindow.setEnabled(False)
        # self.terminate_processing.setEnabled(True)
        thread.quit()
        self.close()


if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    thread = StitchAndFuseThread()
    ImageProcessing = ImageProcessing()
    stitchProcessForm = StitchProcessWindow()
    ImageProcessing.show()
    sys.exit(app.exec_())
