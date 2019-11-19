import sys, os
if hasattr(sys, 'frozen'):
    os.environ['PATH'] = sys._MEIPASS + ";" + os.environ['PATH']
from PyQt5 import QtWidgets, QtGui
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib
import math
matplotlib.use("Qt5Agg")
from PyQt5.QtGui import QPixmap

from operator1 import Ui_Form
from PyQt5.QtWidgets import QFileDialog, QMessageBox


class mywindow(QtWidgets.QWidget,Ui_Form):
    def __init__(self):
        super(mywindow,self).__init__()
        self.setupUi(self)
        self.file_path=""

    def RobertsOperator(self, roi):
        operator_first = np.array([[-1, 0], [0, 1]])
        operator_second = np.array([[0, -1], [1, 0]])
        return np.abs(np.sum(roi[1:, 1:] * operator_first)) + np.abs(np.sum(roi[1:, 1:] * operator_second))

    def RobertsAlogrithm(self, image):
        image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        for i in range(1, image.shape[0]):
            for j in range(1, image.shape[1]):
                image[i, j] = self.RobertsOperator(image[i - 1:i + 2, j - 1:j + 2])
        return image[1:image.shape[0], 1:image.shape[1]]

    def PreWittOperator(self, roi):
        prewitt_x = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        prewitt_y = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        # result = np.abs(np.sum(roi * prewitt_x)) + np.abs(np.sum(roi * prewitt_y))
        # result = np.abs(np.sum(roi * prewitt_x))*0.5+np.abs(np.sum(roi * prewitt_y))*0.5
        result = (np.abs(np.sum(roi * prewitt_x)) ** 2 + np.abs(np.sum(roi * prewitt_y)) ** 2) ** 0.5
        return result

    def PreWittAlogrithm(self, image):
        new_image = np.zeros(image.shape)
        image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        for i in range(1, image.shape[0] - 1):
            for j in range(1, image.shape[1] - 1):
                new_image[i - 1, j - 1] = self.PreWittOperator(image[i - 1:i + 2, j - 1:j + 2])
        return new_image.astype(np.uint8)

    def SobelOperator(self, roi):
        sobel_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        sobel_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        result = (np.abs(np.sum(roi * sobel_x)) ** 2 + np.abs(np.sum(roi * sobel_y)) ** 2) ** 0.5
        return result

    def SobelAlogrithm(self, image):
        new_image = np.zeros(image.shape)
        image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        for i in range(1, image.shape[0] - 1):
            for j in range(1, image.shape[1] - 1):
                new_image[i - 1, j - 1] = self.SobelOperator(image[i - 1:i + 2, j - 1:j + 2])
        return new_image.astype(np.uint8)

    def median(self, img, x, y, size):
        # print('in m\n')
        sum_s = []
        size1 = int(size / 2)
        for k in range(-size1, size1 + 1):
            for m in range(-size1, size1 + 1):
                sum_s.append(img[x + k][y + m])
        sum_s.sort()
        return sum_s[(int(size * size / 2) + 1)]

    # def put(self):
    #     print('into')

    def median_filter(self, im_copy_med, img, size):
        size1 = int(size / 2)
        img = cv2.copyMakeBorder(img, size1, size1, size1, size1, cv2.BORDER_DEFAULT)
        print('01')
        im_copy_med = cv2.copyMakeBorder(im_copy_med, size1, size1, size1, size1, cv2.BORDER_DEFAULT)
        # for i in range(int(size / 2), img.shape[0] - int(size / 2)):
        #     for j in range(int(size / 2), img.shape[1] - int(size / 2)):
        #         im_copy_med[i][j] = self.median(img, i, j, size)
        for i in range(size1, img.shape[0] - size1):
            for j in range(size1, img.shape[1] - size1):
                # print('a')
                # print(im_copy_med[i][j])
                im_copy_med[i][j] = self.median(img, i, j, size)
                # print('a')
        print('3')
        return im_copy_med[size1:im_copy_med.shape[0] - size1, size1:im_copy_med.shape[1] - size1]

    def mean(self, img, x, y, size):
        sum_s = 0
        size1 = int(size / 2)
        for k in range(-size1, size1 + 1):
            for m in range(-size1, size1 + 1):
                sum_s += img[x + k][y + m] / (size * size)
        return sum_s

    def mean_filter(self, im_copy_mean, img, size):
        size1 = int(size / 2)
        img = cv2.copyMakeBorder(img, size1, size1, size1, size1, cv2.BORDER_DEFAULT)
        im_copy_mean = cv2.copyMakeBorder(im_copy_mean, size1, size1, size1, size1, cv2.BORDER_DEFAULT)
        # for i in range(int(size / 2), img.shape[0] - int(size / 2)):
        #     for j in range(int(size / 2), img.shape[1] - int(size / 2)):
        #         im_copy_mean[i][j] = self.mean(img, i, j, size)
        # return im_copy_mean
        for i in range(size1, img.shape[0] - size1):
            for j in range(size1, img.shape[1] - size1):
                im_copy_mean[i][j] = self.mean(img, i, j, size)
        return im_copy_mean[size1:im_copy_mean.shape[0] - size1, size1:im_copy_mean.shape[1] - size1]

    def gaussian(self, kernel_size, sigma):
        print('gaussion kernel')
        center = kernel_size // 2
        print('hi')
        kernel = np.zeros((kernel_size, kernel_size))
        print('here')
        if sigma <= 0:
            sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8
        s = sigma ** 2
        sum_val = 0
        for i in range(kernel_size):
            for j in range(kernel_size):
                x, y = i - center, j - center
                kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / 2 * s)
                sum_val += kernel[i, j]
        kernel = kernel / sum_val
        return kernel

    def gaussian_filter(self, img, kernel):
        print('gaussian_filter')
        res_h = img.shape[0] - kernel.shape[0] + 1
        res_w = img.shape[1] - kernel.shape[1] + 1
        res = np.zeros((res_h, res_w))
        dh = kernel.shape[0]
        dw = kernel.shape[1]
        for i in range(res_h):
            for j in range(res_w):
                res[i, j] = np.sum(img[i:i + dh, j:j + dw] * kernel)
        return res

    def openImg(self):
        imgName,imgType= QFileDialog.getOpenFileName(self,
                                    '打开图片',
                                    'c:\\',
                                    'Image files(*.jpg *.gif *.png)')

        print(imgName)
        self.file_path=imgName
        #利用qlabel显示图片
        png = QtGui.QPixmap(imgName).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(png)

    def read_img(self, path):
        try:
            img = cv2.imread(path)
        except:
            print('cannot find img')
            img = cv2.imread('01.jpg')
        grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return grayImage

    def operator(self, path, type):
        grayImage = self.read_img(path)
        if type == 'r':
            print('r start')
            Roberts_saber = self.RobertsAlogrithm(grayImage)
            print('finish r cal')
            plt.imshow(Roberts_saber, cmap=plt.cm.gray), plt.title(u'Roberts'), plt.axis('off')
            plt.show()
            return
        if type == 'p':
            PreWitt_saber = self.PreWittAlogrithm(grayImage)
            plt.imshow(PreWitt_saber, cmap=plt.cm.gray), plt.title(u'PreWitt'), plt.axis('off')
            plt.show()
            return
        if type == 's':
            Sobel_saber = self.SobelAlogrithm(grayImage)
            plt.imshow(Sobel_saber, cmap=plt.cm.gray), plt.title(u'Sobel'), plt.axis('off')
            plt.show()
            return
        print('type err')
        return

    def Filter(self, path, type, kernel_size, sigma=1):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        if type == 'med':
            # print('med in')
            # print(path)
            # print(kernel_size)
            grayImage = self.read_img(path)
            im_copy_med = grayImage
            medimg = self.median_filter(im_copy_med, grayImage, kernel_size)
            print(medimg.shape[0])
            print(medimg.shape[1])
            # plt.subplot(121), plt.imshow(grayImage, cmap=plt.cm.gray), plt.title(u'原图'), plt.axis('off')
            # plt.subplot(122),
            print('med out')
            plt.rcParams['figure.figsize'] = (medimg.shape[0], medimg.shape[1])
            plt.imshow(medimg, cmap=plt.cm.gray), plt.title(u'中值'), plt.axis('off')
            plt.show()
            return
        if type == 'mean':
            grayImage = self.read_img(path)
            im_copy_mean = grayImage
            meanimg = self.mean_filter(im_copy_mean, grayImage, kernel_size)
            plt.imshow(meanimg, cmap=plt.cm.gray), plt.title(u'均值'), plt.axis('off')
            plt.show()
            return
        if type == 'guass':
            grayImage = self.read_img(path)
            # plt.subplot(121), plt.imshow(grayImage, cmap=plt.cm.gray), plt.title(u'原图'), plt.axis('off')
            gausimg = self.gaussian_filter(grayImage, self.gaussian(kernel_size, sigma))
            # plt.subplot(122),
            plt.imshow(gausimg, cmap=plt.cm.gray), plt.title(u'高斯'), plt.axis('off')
            plt.show()
            return
        print('Type Erro')
        return

    def process_img(self, type):
        if type == 'med' or type=='mean':
            kernel_size=self.lineEdit.text()
            print("kernel size:")
            print(kernel_size)
            if kernel_size == '':
                kernel_size = 3
            else:
                try:
                    kernel_size = int(kernel_size)
                    if kernel_size <= 0 :
                        QMessageBox.critical(self.centralwidget, "错误", "请重新设置kernel size")
                        return
                except:
                    QMessageBox.critical(self.centralwidget, "错误", "请重新设置kernel size")
                    return
            self.Filter(self.file_path,type,kernel_size)
            return
        if type=='guass':
            kernel_size=self.lineEdit_2.text()
            print('kernel')
            print(kernel_size)
            if kernel_size == '':
                kernel_size = 3
            else:
                try:
                    kernel_size = int(kernel_size)
                    if kernel_size <= 0:
                        QMessageBox.critical(self.centralwidget, "错误", "请重新设置kernel size")
                        return
                except:
                    QMessageBox.critical(self.centralwidget, "错误", "请重新设置kernel size")
                    return
            sigma=self.lineEdit_3.text()
            if sigma == '':
                sigma = 1.0
            else:
                try:
                    sigma = float(sigma)
                    if sigma <= 0.0:
                        QMessageBox.critical(self.centralwidget, "错误", "请重新设置sigma")
                        return
                except:
                    QMessageBox.critical(self.centralwidget, "错误", "请重新设置sigma")
                    return
            print('sigma')
            print(sigma)
            if sigma == '':
                sigma = 1
            else:
                sigma = int(sigma)
            self.Filter(self.file_path, type, kernel_size, sigma)
            return
        # operators
        if type == 'r':
            self.operator(self.file_path, 'r')
            return
        if type == 'p':
            self.operator(self.file_path, 'p')
            return
        if type == 's':
            self.operator(self.file_path, 's')
            return

    def robots(self):
        self.openImg()
        self.process_img('r')

    def Sobel(self):
        self.openImg()
        print('s open')
        self.process_img('s')

    def Prewitt(self):
        self.openImg()
        self.process_img('p')

    def mean_btn(self):
        self.openImg()
        self.process_img('mean')

    def median_btn(self):
        self.openImg()
        self.process_img('med')

    def guassian_btn(self):
        self.openImg()
        self.process_img('guass')


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = mywindow()
    window.show()
    sys.exit(app.exec_())
