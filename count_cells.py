# код, использованный в работе с CV2, взят с сайта
# https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/

from os import path
import cv2 as cv
import numpy as np

refPt = []  # список для двух точек изображения
cropping = False  # флаг того, обрезано ли изображение уже или нет


# функция для вырезания
def click_and_crop(event, x, y, flags, param):
    global refPt, cropping
    # если левая кнопка мыши нажата, регистрируем первую точку
    # прямоугольника (верхнюю левую);
    if event == cv.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
    # если левая кнопка мыши нажата во второй раз, регистрируем
    # вторую точку прямоугольника (нижнюю правую)
    elif event == cv.EVENT_LBUTTONUP:
        refPt.append((x, y))
        cropping = False
        # рисуем прямоугольник, изображение внутри которого хотим вырезать
        cv.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv.imshow("Image", image)


# считывание пути к изображению
image_path = input('Введите путь к изображению. Для выхода из приложения напишите "Выход"\n')
# требование выхода
if image_path.lower() == 'выход':
    exit(0)
# проверка полученного пути
while not isinstance(image_path, str):
    print('Путь к изображению - строка (вводится в кавычках). '
          'Пожалуйста, введите корректный путь к изображению.\n')
    image_path = input('Введите путь к изображению. Для выхода из приложения напишите "Выход"\n')
while not path.exists(image_path):
    if image_path.lower() == 'выход':
        exit(0)
    print('Указанный путь не существует в системе. '
          'Пожалуйста, проверьте правильность написания пути.')
    image_path = input('Введите путь к изображению. Для выхода из приложения напишите "Выход"\n')

image = cv.imread(image_path)
while image is None:
    if image_path.lower() == 'выход':
        exit(0)
    print('Указанный путь ведет не к файлу изображения. '
          'Пожалуйста, проверьте правильность написания пути.')
    image_path = input('Введите путь к изображению. Для выхода из приложения напишите "Выход"\n')
    image = cv.imread(image_path)


# создаем копию изображения (обрезать будем ее), готовим изображение к выводу на экран
clone = image.copy()
cv.namedWindow("Image", cv.WINDOW_NORMAL)
# после каждого клика запускаем функцию click_and_crop()
cv.setMouseCallback("Image", click_and_crop)

# бесконечный цикл, в котором происходит запись координат прямоугольника для обрезания
# correct_rectangle = False
while True:
    # выводим изображение на экран
    cv.imshow("Image", image)
    key = cv.waitKey(1) & 0xFF
    # если нажата клавиша 'r' (reset), регион обрезания сбрасывается
    # до последнего состояния
    if key == ord("r"):
        image = clone.copy()
    # если нажата клавиша 'c' (close), сбрасывается цикл (окончание)
    elif key == ord("c"):
        # проверка на то, что полученная область - прямоугольник
        if refPt[0][1] != refPt[1][1] and refPt[0][0] != refPt[1][0]:
            roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
            break
        else:
            print('Выделенная область имеет размерность 1 (полоса/точка). '
                  'Пожалуйста, выделите область размерности 2 (прямоугольник).')


# выводим на экран получившийся регион прямоугольной формы
cv.imshow("ROI", roi)
cv.waitKey(0)

# после обрезания закрываем все окна
cv.destroyAllWindows()

# TODO!!!
# добавить подгрузку нейронной сети из папки


def get_image_mask(img, threshold):
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    crp_img = cv.resize(img, (256, 256)) / 255
    return model.predict(np.expand_dims(crp_img, 0))[0][..., 0] > threshold


def check_convex(region) -> bool:
    pass


def get_clusters_from_mask(mask):
    pass


def count_cells(img):
    mask = get_image_mask(img, threshold=0.1)
    amount = 0
    for region in get_clusters_from_mask(mask):
        if check_convex(region):
            amount += 1
        else:
            amount += model_ellipses.predict(region)
    pass
