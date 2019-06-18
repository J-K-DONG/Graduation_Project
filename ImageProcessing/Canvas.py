# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets


class Canvas(QtWidgets.QGraphicsView):
    sync_wheel_signal = QtCore.pyqtSignal(str, object, tuple)
    sync_mouse_signal = QtCore.pyqtSignal(str, str, object)

    def __init__(self, parent):
        super().__init__(parent)
        self.side = None
        self._zoom = 0
        self._empty = True
        self._scene = QtWidgets.QGraphicsScene(self)
        self._photo = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._photo)
        self.setScene(self._scene)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.NoAnchor)
        self.setResizeAnchor(QtWidgets.QGraphicsView.NoAnchor)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(245, 245, 245)))
        self.setFrameShape(QtWidgets.QFrame.NoFrame)

        # self.setMouseTracking(True)

    def hasPhoto(self):
        return not self._empty

    def fitInView(self, scale=True):
        rect = QtCore.QRectF(self._photo.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.hasPhoto():
                unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect)
                factor = max(viewrect.width() / scenerect.width(),
                             viewrect.height() / scenerect.height())
                self.scale(factor, factor)
            self._zoom = 0

    def setPhoto(self, pixmap=None):
        self._zoom = 0
        if pixmap and not pixmap.isNull():
            self._empty = False
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
            self._photo.setPixmap(pixmap)
        else:
            self._empty = True
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            self._photo.setPixmap(QtGui.QPixmap())
        self.fitInView()

    def wheelEvent(self, event: QtGui.QWheelEvent, tag=None):
        if tag is None:
            self.sync_wheel_signal.emit(self.side, event, (self.x(), self.y())
                                        )
        if self.hasPhoto():
            if event.angleDelta().y() > 0:
                factor = 1.25
                self._zoom += 1
            else:
                factor = 0.8
                self._zoom -= 1
            if self._zoom != 0:
                cur_pos = self.mapToScene(event.pos())
                self.scale(factor, factor)
                new_pos = self.mapToScene(event.pos())
                delta_zoomed = new_pos - cur_pos
                shift_x, shift_y = delta_zoomed.x(), delta_zoomed.y()
                self.translate(shift_x, shift_x)
            elif self._zoom == 0:
                self.fitInView()
            else:
                self._zoom = 0

    @QtCore.pyqtSlot(str, object, tuple, name='wheel_receiver')
    def on_sync_wheel_signal(self, msg, event: QtGui.QWheelEvent, shift):
        self.wheelEvent(event, 'sync')

    @QtCore.pyqtSlot(str, str, object, name='mouse_receiver')
    def on_sync_mouse_signal(self, msg, type_, event):
        # print(msg, type_, event)
        if type_ == 'mouse_press':
            self.mousePressEvent(event, 'sync')
        if type_ == 'mouse_release':
            self.mouseReleaseEvent(event, 'sync')
        if type_ == 'mouse_move':
            self.mouseMoveEvent(event, 'sync')

    def toggleDragMode(self):
        if self.dragMode() == QtWidgets.QGraphicsView.ScrollHandDrag:
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        elif not self._photo.pixmap().isNull():
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

    def resizeEvent(self, event):
        self.fitInView()

    def make_connection(self, other):
        other.sync_wheel_signal.connect(self.on_sync_wheel_signal)
        other.sync_mouse_signal.connect(self.on_sync_mouse_signal)

    def mousePressEvent(self, event: QtGui.QMouseEvent, tag=None):
        if tag is None and self._photo.isUnderMouse():
            self.sync_mouse_signal.emit(self.side, 'mouse_press', event)
        super(Canvas, self).mousePressEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent, tag=None):
        if tag is None and self._photo.isUnderMouse():
            self.sync_mouse_signal.emit(self.side, 'mouse_release', event)
        super(Canvas, self).mouseReleaseEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent, tag=None):
        if tag is None and self._photo.isUnderMouse():
            self.sync_mouse_signal.emit(self.side, 'mouse_move', event)
        super(Canvas, self).mouseMoveEvent(event)

