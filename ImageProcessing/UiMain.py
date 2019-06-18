# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class UiMainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1254, 465)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.splitter_3 = QtWidgets.QSplitter(self.centralwidget)
        self.splitter_3.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_3.setObjectName("splitter_3")
        self.listWidget = QtWidgets.QListWidget(self.splitter_3)
        self.listWidget.setMinimumSize(QtCore.QSize(50, 0))
        self.listWidget.setMaximumSize(QtCore.QSize(300, 16777215))
        self.listWidget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.listWidget.setObjectName("listWidget")
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        self.splitter_2 = QtWidgets.QSplitter(self.splitter_3)
        self.splitter_2.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_2.setObjectName("splitter_2")
        self.splitter = QtWidgets.QSplitter(self.splitter_2)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.layoutWidget = QtWidgets.QWidget(self.splitter)
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")

        self.verticalLayout_left = QtWidgets.QVBoxLayout()
        self.verticalLayout_left.setObjectName("verticalLayout_left")
        self.left_layout = QtWidgets.QVBoxLayout()
        self.left_layout.setObjectName("left_layout")
        self.left_keep = QtWidgets.QGraphicsView(self.layoutWidget)
        self.left_keep.setObjectName("left_keep")
        self.left_layout.addWidget(self.left_keep)
        self.verticalLayout_left.addLayout(self.left_layout)
        self.label = QtWidgets.QLabel(self.layoutWidget)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.verticalLayout_left.addWidget(self.label)
        self.horizontalLayout.addLayout(self.verticalLayout_left)

        self.verticalLayout_right = QtWidgets.QVBoxLayout()
        self.verticalLayout_right.setObjectName("verticalLayout_right")
        self.right_layout = QtWidgets.QVBoxLayout()
        self.right_layout.setObjectName("right_layout")
        self.right_keep = QtWidgets.QGraphicsView(self.layoutWidget)
        self.right_keep.setObjectName("right_keep")
        self.right_layout.addWidget(self.right_keep)
        self.verticalLayout_right.addLayout(self.right_layout)
        self.label_2 = QtWidgets.QLabel(self.layoutWidget)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_right.addWidget(self.label_2)
        self.horizontalLayout.addLayout(self.verticalLayout_right)

        self.gridLayout.addWidget(self.splitter_3, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1154, 26))
        self.menubar.setObjectName("menubar")

        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")

        self.menuEdit = QtWidgets.QMenu(self.menubar)
        self.menuEdit.setObjectName("menuEdit")

        self.image_denoising = QtWidgets.QMenu(self.menuEdit)
        self.image_denoising.setObjectName("image_denoising")

        self.image_fusing = QtWidgets.QMenu(self.menuEdit)
        self.image_fusing.setObjectName("image_fusing")

        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")

        MainWindow.setStatusBar(self.statusbar)

        self.picture_import = QtWidgets.QAction(MainWindow)
        self.picture_import.setObjectName("picture_import")

        self.save_result = QtWidgets.QAction(MainWindow)
        self.save_result.setObjectName("save_result")

        self.mean_denoising = QtWidgets.QAction(MainWindow)
        self.mean_denoising.setObjectName("mean_denoising")

        self.gauss_denoising = QtWidgets.QAction(MainWindow)
        self.gauss_denoising.setObjectName("gauss_denoising")

        self.sf_fusing = QtWidgets.QAction(MainWindow)
        self.sf_fusing.setObjectName("sf_fusing")

        self.multi_band_blending_fusing = QtWidgets.QAction(MainWindow)
        self.multi_band_blending_fusing.setObjectName("multi_band_blending_fusing")

        self.sf_and_mbb_fusing = QtWidgets.QAction(MainWindow)
        self.sf_and_mbb_fusing.setObjectName("sf_and_mbb_fusing")

        self.menuFile.addAction(self.picture_import)
        self.menuFile.addAction(self.save_result)

        self.image_denoising.addAction(self.mean_denoising)
        self.image_denoising.addAction(self.gauss_denoising)

        self.image_fusing.addAction(self.sf_fusing)
        self.image_fusing.addAction(self.multi_band_blending_fusing)
        self.image_fusing.addAction(self.sf_and_mbb_fusing)

        self.menuEdit.addAction(self.image_denoising.menuAction())
        self.menuEdit.addAction(self.image_fusing.menuAction())

        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "眼底OCT图像纠偏融合软件"))

        __sortingEnabled = self.listWidget.isSortingEnabled()
        self.listWidget.setSortingEnabled(False)
        item = self.listWidget.item(0)
        item.setText(_translate("MainWindow", "New Item"))
        item = self.listWidget.item(1)
        item.setText(_translate("MainWindow", "New Item"))
        item = self.listWidget.item(2)
        item.setText(_translate("MainWindow", "New Item"))
        item = self.listWidget.item(3)
        item.setText(_translate("MainWindow", "New Item"))
        item = self.listWidget.item(4)
        item.setText(_translate("MainWindow", "New Item"))
        item = self.listWidget.item(5)
        item.setText(_translate("MainWindow", "New Item"))
        item = self.listWidget.item(6)
        item.setText(_translate("MainWindow", "New Item"))
        item = self.listWidget.item(7)
        item.setText(_translate("MainWindow", "New Item"))
        self.listWidget.setSortingEnabled(__sortingEnabled)

        self.label.setText(_translate("MainWindow", "原图"))
        self.label_2.setText(_translate("MainWindow", "结果图"))

        self.menuFile.setTitle(_translate("MainWindow", "文件"))
        self.menuEdit.setTitle(_translate("MainWindow", "编辑"))
        self.image_denoising.setTitle(_translate("MainWindow", "图像去噪"))
        self.image_fusing.setTitle(_translate("MainWindow", "纠偏融合"))

        self.picture_import.setText(_translate("MainWindow", "导入图像集"))
        self.picture_import.setShortcut(_translate("MainWindow", "Shift+O"))

        self.save_result.setText(_translate("MainWindow", "导出结果图"))
        self.save_result.setShortcut(_translate("MainWindow", "Shift+S"))

        self.mean_denoising.setText(_translate("MainWindow", "均值去噪"))
        self.mean_denoising.setToolTip(_translate("MainWindow", "均值去噪"))
        self.mean_denoising.setShortcut(_translate("MainWindow", "Shift+J"))

        self.gauss_denoising.setText(_translate("MainWindow", "高斯去噪"))
        self.gauss_denoising.setShortcut(_translate("MainWindow", "Shift+G"))

        self.sf_fusing.setText(_translate("MainWindow", "空间频率融合"))
        self.sf_fusing.setToolTip(_translate("MainWindow", "空间频率融合"))
        self.sf_fusing.setShortcut(_translate("MainWindow", "Shift+H"))

        self.multi_band_blending_fusing.setText(_translate("MainWindow", "多尺度融合"))
        self.multi_band_blending_fusing.setToolTip(_translate("MainWindow", "多尺度融合"))
        self.multi_band_blending_fusing.setShortcut(_translate("MainWindow", "Shift+T"))

        self.sf_and_mbb_fusing.setText(_translate("MainWindow", "空间频率及多尺度融合"))
        self.sf_and_mbb_fusing.setToolTip(_translate("MainWindow", "空间频率及多尺度融合"))
        self.sf_and_mbb_fusing.setShortcut(_translate("MainWindow", "Shift+D"))

