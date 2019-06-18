# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'dialog_progressBar.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class UiDialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(660, 340)

        # 进度条
        self.progressBar = QtWidgets.QProgressBar(Dialog)
        self.progressBar.setGeometry(QtCore.QRect(30, 70, 600, 51))
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")

        # 左上角文本框
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(50, 20, 131, 41))
        self.label.setObjectName("label")

        # 右上角文本框
        self.lb_stitchInf = QtWidgets.QLabel(Dialog)
        self.lb_stitchInf.setGeometry(QtCore.QRect(480, 20, 131, 41))
        self.lb_stitchInf.setText("处理开始...")
        self.lb_stitchInf.setObjectName("lb_stitchInf")

        # 各阶段耗时的文本框
        # 特征提取和匹配进度
        self.label_track_process = QtWidgets.QLabel(Dialog)
        self.label_track_process.setGeometry(QtCore.QRect(50, 150, 431, 41))
        self.label_track_process.setText("")
        self.label_track_process.setObjectName("label_track_process")

        # 特征提取和匹配耗时
        self.label_track_time = QtWidgets.QLabel(Dialog)
        self.label_track_time.setGeometry(QtCore.QRect(50, 170, 431, 41))
        self.label_track_time.setText("")
        self.label_track_time.setObjectName("label_track_time")

        # 图像校准进度
        self.label_correct_process = QtWidgets.QLabel(Dialog)
        self.label_correct_process.setGeometry(QtCore.QRect(50, 190, 431, 41))
        self.label_correct_process.setText("")
        self.label_correct_process.setObjectName("label_correct_process")

        # 图像校准耗时
        self.label_correct_time = QtWidgets.QLabel(Dialog)
        self.label_correct_time.setGeometry(QtCore.QRect(50, 210, 431, 41))
        self.label_correct_time.setText("")
        self.label_correct_time.setObjectName("label_correct_time")

        # 图像融合进度
        self.label_fuse_process = QtWidgets.QLabel(Dialog)
        self.label_fuse_process.setGeometry(QtCore.QRect(50, 230, 431, 41))
        self.label_fuse_process.setText("")
        self.label_fuse_process.setObjectName("label_fuse_process")

        # 图像融合耗时
        self.label_fuse_time = QtWidgets.QLabel(Dialog)
        self.label_fuse_time.setGeometry(QtCore.QRect(50, 250, 431, 41))
        self.label_fuse_time.setText("")
        self.label_fuse_time.setObjectName("label_fuse_time")

        # 总耗时
        self.label_process_time = QtWidgets.QLabel(Dialog)
        self.label_process_time.setGeometry(QtCore.QRect(50, 270, 431, 41))
        self.label_process_time.setText("")
        self.label_process_time.setObjectName("label_process_time")

        # 右下角文本框
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(730, 130, 72, 41))
        self.label_2.setObjectName("label_2")

        # 查看图片框
        self.bt_backMainWindow = QtWidgets.QPushButton(Dialog)
        self.bt_backMainWindow.setGeometry(QtCore.QRect(500, 300, 93, 28))
        self.bt_backMainWindow.setObjectName("bt_backMainWindow")

        # 终止处理框
        # self.terminate_processing = QtWidgets.QPushButton(Dialog)
        # self.terminate_processing.setGeometry(QtCore.QRect(380, 300, 93, 28))
        # self.terminate_processing.setObjectName("terminate_processing")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label_2.setText(_translate("Dialog", ""))
        self.label.setText(_translate("Dialog", "处理进度："))
        self.bt_backMainWindow.setText(_translate("Dialog", "查看结果"))
        # self.terminate_processing.setText(_translate("Dialog", "停止处理"))

