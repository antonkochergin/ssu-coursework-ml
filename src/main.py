import cv2
import numpy as np
from rembg import remove
from PIL import Image

img = cv2.imread("./foots/1/IMG_0315.jpg")
img = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4))


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


cv2.imshow("Result", draw_contour(img, img.shape[1], img.shape[0], img.shape[2]))
cv2.waitKey(0)
