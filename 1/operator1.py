# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'operator1.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!

import sys, os
if hasattr(sys, 'frozen'):
    os.environ['PATH'] = sys._MEIPASS + ";" + os.environ['PATH']
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(549, 296)
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(10, 30, 281, 201))
        self.label.setObjectName("label")
        self.pushButton_2 = QtWidgets.QPushButton(Form)
        self.pushButton_2.setGeometry(QtCore.QRect(20, 260, 61, 21))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(7)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(Form)
        self.pushButton_3.setGeometry(QtCore.QRect(90, 260, 61, 21))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(7)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(Form)
        self.pushButton_4.setGeometry(QtCore.QRect(160, 260, 51, 21))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(7)
        self.pushButton_4.setFont(font)
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(410, 70, 61, 31))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(7)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.pushButton_5 = QtWidgets.QPushButton(Form)
        self.pushButton_5.setGeometry(QtCore.QRect(320, 70, 61, 31))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(7)
        self.pushButton_5.setFont(font)
        self.pushButton_5.setObjectName("pushButton_5")
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setGeometry(QtCore.QRect(300, 40, 61, 31))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(Form)
        self.label_3.setGeometry(QtCore.QRect(300, 110, 61, 21))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(Form)
        self.label_4.setGeometry(QtCore.QRect(300, 140, 41, 16))
        self.label_4.setObjectName("label_4")
        self.pushButton_6 = QtWidgets.QPushButton(Form)
        self.pushButton_6.setGeometry(QtCore.QRect(405, 180, 61, 31))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(7)
        self.pushButton_6.setFont(font)
        self.pushButton_6.setObjectName("pushButton_6")
        self.lineEdit = QtWidgets.QLineEdit(Form)
        self.lineEdit.setGeometry(QtCore.QRect(360, 40, 113, 20))
        self.lineEdit.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.lineEdit.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit.setPlaceholderText(str(3))
        self.lineEdit_2 = QtWidgets.QLineEdit(Form)
        self.lineEdit_2.setGeometry(QtCore.QRect(360, 110, 113, 20))
        self.lineEdit_2.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.lineEdit_2.setPlaceholderText(str(3))
        self.lineEdit_3 = QtWidgets.QLineEdit(Form)
        self.lineEdit_3.setGeometry(QtCore.QRect(360, 140, 113, 20))
        self.lineEdit_3.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.lineEdit_3.setPlaceholderText(str(1))

        self.retranslateUi(Form)
        self.pushButton_2.clicked.connect(Form.robots)
        self.pushButton_4.clicked.connect(Form.Sobel)
        self.pushButton_3.clicked.connect(Form.Prewitt)
        self.pushButton_5.clicked.connect(Form.median_btn)
        self.pushButton.clicked.connect(Form.mean_btn)
        self.pushButton_6.clicked.connect(Form.guassian_btn)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "作业1"))
        self.label.setText(_translate("Form", "原图"))
        self.pushButton_2.setText(_translate("Form", "Roberts"))
        self.pushButton_3.setText(_translate("Form", "Prewitt"))
        self.pushButton_4.setText(_translate("Form", "Sobel"))
        self.pushButton.setText(_translate("Form", "均值滤波"))
        self.pushButton_5.setText(_translate("Form", "中值滤波"))
        self.label_2.setText(_translate("Form", "kernel size"))
        self.label_3.setText(_translate("Form", "kernel size"))
        self.label_4.setText(_translate("Form", "sigma"))
        self.pushButton_6.setText(_translate("Form", "高斯滤波"))
