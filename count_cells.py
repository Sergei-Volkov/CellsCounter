from PyQt5 import QtWidgets
import os
import sys
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.widgets import RectangleSelector
import matplotlib.pylab as pylab
params = {
    'figure.figsize': (10, 10),
    'axes.titlesize': 'xx-small'
}
pylab.rcParams.update(params)

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
count_model = keras.models.load_model('count_model.h5')


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
            self.ax.titlesize = 14
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

    def countCells(self):
        global model, count_model
        if self.curImage is not None:
            self.curImage = cv.resize(self.curImage, (512, 512)) / 255
            mask = (model.predict(np.expand_dims(self.curImage, 0)) > 0.1)[0]

            counter = 0
            coord = pd.DataFrame(data=np.argwhere(mask[..., 0]), columns=['x', 'y'])
            dbscan_repr = DBSCAN(min_samples=2, eps=1).fit_predict(coord)
            coord['label'] = dbscan_repr
            for i in range(0, dbscan_repr.max()+1):
                df = coord[coord["label"] == i]
                df = df.transform(lambda x: x - x.min() + 32)
                img = np.zeros((64, 64))
                for row in df.iterrows():
                    img[row[1][0], row[1][1]] = 1
                counter += round(count_model(
                    np.expand_dims(np.repeat(np.expand_dims(img, -1), 3, axis=2), 0)
                ).numpy()[0, 0])

            masked = self.curImage.copy()
            masked[np.repeat(mask, 3, axis=-1) == 0] = 0
            self.ax.imshow(self.curImage)
            self.ax.imshow(masked, alpha=0.7)
            plt.text(
                self.curImage.shape[0], 0, f'{counter} клеток',
                bbox=dict(fill=False, edgecolor='green', linewidth=2)
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
