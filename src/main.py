import cv2
import numpy as np


from rembg import remove
from PIL import Image



# img = cv2.imread("./foots/Test/i.jpg")
img = cv2.imread("./foots/1/IMG_0315.jpg")
img = cv2.resize(img , (img.shape[1] // 4 , img.shape[0]//4))
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# blurred = cv2.GaussianBlur(img , (9,9), 0)
# edges = cv2.Canny(img, 130, 140)
#
# # Обводка
# kernel = np.ones((7, 7), np.uint8)
# img = cv2.dilate(img, kernel, iterations=1)
# cv2.imshow("Edges (Canny)", edges)




result_pil = remove(img)

# Конвертируем для OpenCV
result_np = np.array(result_pil)

cv2.imshow('Result', result_np)
cv2.waitKey(0)
