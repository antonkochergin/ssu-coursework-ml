import cv2
import numpy as np
from rembg import remove
from PIL import Image

img = cv2.imread("./foots/1/IMG_0315.jpg")
img = cv2.resize(img, (img.shape[1] // 5, img.shape[0] // 5))


# cv2.imshow("Ishodnik", img)


def draw_grid(image, cell_size=50, color=(0, 255, 0), thickness=1):
    """
    Рисует сетку на изображении
    """
    img_copy = image.copy()
    h, w = img_copy.shape[:2]

    # Рисуем вертикальные линии
    for x in range(0, w, cell_size):
        cv2.line(img_copy, (x, 0), (x, h), color, thickness)

    # Рисуем горизонтальные линии
    for y in range(0, h, cell_size):
        cv2.line(img_copy, (0, y), (w, y), color, thickness)

    return img_copy


img = draw_grid(img, cell_size=25, color=(40, 40, 40), thickness=1)


# ВОПРОС: тут надо подавать изображение с параметрами или можно просто подать изображение и уже из него извлечь данные
# или тут надо чтобы исходное изображение подгонялось под определенный размер
def draw_contour(image, width, height, channels):
    """
    Данная функция: удаляет фон, преобразовывает в оттенки серого
    """
    # Проверяем размеры
    if image.shape[1] != width or image.shape[0] != height:
        image = cv2.resize(image, (width, height))
    # Конвртируем в RGB для PIL
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    # Удаляем фон
    img_no_bg = remove(img_pil)
    img_no_bg_cv = cv2.cvtColor(np.array(img_no_bg), cv2.COLOR_RGB2BGR)
    # Перевод в оттенки серого
    gray = cv2.cvtColor(img_no_bg_cv, cv2.COLOR_BGR2GRAY)
    return gray


gray = draw_contour(img, img.shape[1], img.shape[0], img.shape[2])
# cv2.imshow("Result", gray)

ret, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imshow("Result", result)
cv2.waitKey(0)
