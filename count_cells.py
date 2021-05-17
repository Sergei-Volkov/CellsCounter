from PyQt5 import QtWidgets
import os
import sys
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.widgets import RectangleSelector

import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K


def dice_loss(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (
        (2. * intersection + smooth) /
        (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)
    )


model = keras.models.load_model('model.h5', custom_objects={'dice_loss': dice_loss})


class CustomDialog(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Сообщение')

        QBtn = QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel

        self.buttonBox = QtWidgets.QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.layout = QtWidgets.QVBoxLayout()
        message = QtWidgets.QLabel("Область выбрана верно?")
        self.layout.addWidget(message)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)


class Widget(QtWidgets.QWidget):

    def __init__(self):
        super(Widget, self).__init__()
        self.figure = plt.figure(dpi=300, figsize=(10, 10))
        self.ax = None
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.fileNames = None
        self.curFileName = None
        self.curImage = None
        self.rs = None
        self.initUI()

    def initUI(self):
        self.setGeometry(1000, 1000, 1200, 800)
        self.center()
        self.setWindowTitle('Cells Counter')

        # Grid Layout
        grid = QtWidgets.QGridLayout()
        self.setLayout(grid)

        # Canvas and Toolbar
        grid.addWidget(self.canvas, 3, 0, 1, 2)
        grid.addWidget(self.toolbar, 0, 0, 1, 2)

        # Import Files Button
        btn1 = QtWidgets.QPushButton('Выбрать файлы', self)
        btn1.resize(btn1.sizeHint())
        btn1.clicked.connect(self.getFilesNames)
        grid.addWidget(btn1, 1, 0)

        # Show Image Button
        btn2 = QtWidgets.QPushButton('Показать изображение', self)
        btn2.resize(btn2.sizeHint())
        btn2.clicked.connect(self.showImage)
        grid.addWidget(btn2, 1, 1)

        # Crop Image Button
        btn3 = QtWidgets.QPushButton('Обрезать изображение', self)
        btn3.resize(btn2.sizeHint())
        btn3.clicked.connect(self.cropImage)
        grid.addWidget(btn3, 2, 0)

        # Count Cells Button
        btn4 = QtWidgets.QPushButton('Посчитать число клеток', self)
        btn4.resize(btn2.sizeHint())
        btn4.clicked.connect(self.countCells)
        grid.addWidget(btn4, 2, 1)

        # # Show Previous Image Button
        # btn5 = QtWidgets.QPushButton('Показать предыдущее изображение', self)
        # btn5.resize(btn5.sizeHint())
        # btn5.clicked.connect(self.showImage)
        # grid.addWidget(btn5, 2, 0)
        #
        # # Show Next Image Button
        # btn6 = QtWidgets.QPushButton('Показать следующее изображение', self)
        # btn6.resize(btn6.sizeHint())
        # btn6.clicked.connect(self.showImage)
        # grid.addWidget(btn6, 2, 1)

        self.show()

    def getFilesNames(self):
        self.fileNames, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            'File System',
            os.getcwd(),
            'Images (*.bmp *.dib *.jpeg *.jpg *.jpe *.jp2 *.png *.webp *.exr '
            '*.pbm *.pgm *.ppm *.pxm *.pnm *.sr *.ras *.tiff *.tif *.hdr *.pic)'
        )

    def showImage(self):
        try:
            self.curFileName = self.fileNames.pop(0)
            self.curImage = cv.imread(self.curFileName)
            self.curImage = cv.cvtColor(self.curImage, cv.COLOR_BGR2RGB)
            plt.cla()
            self.ax = self.figure.add_subplot(111)
            self.ax.imshow(self.curImage)
            plt.axis('off')
            self.ax.set_title(self.curFileName.split('/')[-1])
            self.canvas.draw()
        except AttributeError:
            QtWidgets.QMessageBox.about(self, "Сообщение", "В памяти еще нет файлов. Для начала обработки "
                                                           'выберите файлы с помощью кнопки "Выбрать файлы"')
        except IndexError:
            QtWidgets.QMessageBox.about(self, "Сообщение", 'Вы обработали все файлы. Выберите новые '
                                                           'с помощью кнопки "Выбрать файлы"')

    def cropImage(self):
        if self.ax is not None:
            QtWidgets.QMessageBox.about(self, "Сообщение", "Выберите левой кнопкой мыши верхнюю левую "
                                                           "и нижнюю правую точки прямоугольника, "
                                                           "который требуется вырезать")
            self.canvas.mpl_connect("button_press_event", self.mouseEventRegister)
            self.canvas.mpl_connect("button_release_event", self.mouseEventRegister)
            self.rs = RectangleSelector(
                self.ax,
                self.lineSelectCallback,
                drawtype='box',
                button=[1, 3],  # don't use middle button
                minspanx=5,
                minspany=5,
                spancoords='pixels',
                rectprops=dict(edgecolor="green", fill=False),
                interactive=True
            )
        else:
            QtWidgets.QMessageBox.about(self, "Сообщение", "На экране еще нет изображения. Для начала обработки "
                                                           'выведите его на экран с помощью кнопки '
                                                           '"Показать изображение"')

    def mouseEventRegister(self, event):
        if event.button == 1 or event.button == 3 and not self.rs.active:
            self.rs.set_active(True)
        else:
            self.rs.set_active(False)

    def lineSelectCallback(self, click, release):
        x1, y1 = click.xdata, click.ydata
        x2, y2 = release.xdata, release.ydata
        print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
        print("(%3.0f, %3.0f) --> (%3.0f, %3.0f)" % (x1, y1, x2, y2))
        try:
            dlg = CustomDialog()
            if dlg.exec():
                self.curImage = self.curImage[round(y1):round(y2), round(x1):round(x2)]
                self.ax.imshow(self.curImage)
                plt.axis('off')
                self.ax.set_title(self.curFileName.split('/')[-1])
                self.canvas.draw()
        except IndexError:
            pass

    def countCells(self, threshold=0.8):
        global model
        if self.curImage is not None:
            self.curImage = cv.resize(self.curImage, (512, 512)) / 255.
            mask = model.predict(np.expand_dims(self.curImage, 0))[0][..., 0]
            print(mask)
            self.ax.imshow(
                mask
                # np.ma.masked_array(
                #     self.curImage,
                #     np.repeat(np.expand_dims(mask, -1), 3, axis=-1)
                # ),
                # alpha=0.7
            )
            plt.axis('off')
            self.ax.set_title(self.curFileName.split('/')[-1])
            self.canvas.draw()
        else:
            QtWidgets.QMessageBox.about(self, "Сообщение", "На экране еще нет изображения. Для начала обработки "
                                                           'выведите его на экран с помощью кнопки '
                                                           '"Показать изображение"')

    def center(self):
        qr = self.frameGeometry()
        cp = QtWidgets.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = Widget()
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
