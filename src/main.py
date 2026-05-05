import cv2
import numpy as np
from rembg import remove


def resize_with_padding(image, target_size=(512, 512)):
    """
    Изменяет размер, сохраняя пропорции, и добавляет черные полосы.
    """
    h, w = image.shape[:2]
    tw, th = target_size
    scale = min(tw / w, th / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if len(image.shape) == 2:
        final_image = np.zeros((th, tw), dtype=np.uint8)
    else:
        final_image = np.zeros((th, tw, 3), dtype=np.uint8)

    dx = (tw - new_w) // 2
    dy = (th - new_h) // 2
    final_image[dy:dy + new_h, dx:dx + new_w] = resized
    return final_image


def split_feet_smart(src_image, debug=False):
    """
    Разделяет изображение с двумя стопами по промежутку между ними.
    Левая стопа: поворот на 180 градусов.
    Правая стопа: поворот на 180 градусов + отражение по горизонтали.
    """
    h, w = src_image.shape[:2]

    no_bg = remove(src_image)
    mask = no_bg[:, :, 3]
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Контуры
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    # Bounding boxes
    bboxes = sorted([cv2.boundingRect(c) for c in contours], key=lambda b: b[0])
    left_edge = bboxes[0][0] + bboxes[0][2]  # правый край левой стопы
    right_edge = bboxes[1][0]  # левый край правой стопы

    split_x = (left_edge + right_edge) // 2

    # Левая стопа:
    left_foot = src_image[:, :split_x]
    left_foot = cv2.rotate(left_foot, cv2.ROTATE_180)

    # Правая стопа:
    right_foot = src_image[:, split_x:]
    right_foot = cv2.rotate(right_foot, cv2.ROTATE_180)
    right_foot = cv2.flip(right_foot, 1)

    if debug:
        debug_img = src_image.copy()
        cv2.line(debug_img, (split_x, 0), (split_x, h), (0, 255, 0), 3)
        for x, y, bw, bh in bboxes:
            cv2.rectangle(debug_img, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
        cv2.drawContours(debug_img, contours, -1, (255, 0, 0), 2)
        return left_foot, right_foot, debug_img

    return left_foot, right_foot

def get_clean_foot_step(src_image, target_size=(512, 512)):
    """
    Получение чистой маски и серого изображения с сеткой.
    """
    no_bg_rgba = remove()

    mask = no_bg_rgba[:, :, 3]

    _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    refined_mask = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel)
    refined_mask = cv2.medianBlur(refined_mask, 3)  # Сглаживаем "ступеньки" - улучшение

    # Сетка
    img_with_grid = src_image.copy()
    h_orig, w_orig = img_with_grid.shape[:2]
    cell_size = 15
    for x in range(0, w_orig, cell_size):
        cv2.line(img_with_grid, (x, 0), (x, h_orig), (0, 0, 0), 1)
    for y in range(0, h_orig, cell_size):
        cv2.line(img_with_grid, (0, y), (w_orig, y), (0, 0, 0), 1)

    # Перевод в Grayscale и ресайз
    gray_grid = cv2.cvtColor(img_with_grid, cv2.COLOR_BGR2GRAY)

    final_gray = resize_with_padding(gray_grid, target_size)
    final_mask = resize_with_padding(refined_mask, target_size)

    return final_gray, final_mask


def get_maximum_contrast_threshold(gray_image, mask_refined):
    """
    Создание финального бинарного изображения
    """
    # Жесткий порог для выделения сетки внутри стопы
    _, binary = cv2.threshold(gray_image, 45, 255, cv2.THRESH_BINARY)

    # Накладываем маску, чтобы вернуть идеальный контур
    result = cv2.bitwise_and(binary, mask_refined)

    # Финальная бинаризация для четкости краев
    _, result = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)

    return result


if __name__ == "__main__":
    path = "./foots/1/IMG_0315.jpg"
    src = cv2.imread(path)

    left_src, right_src, debug_split = split_feet_smart(src, debug=True)
    left_resized = resize_with_padding(left_src, target_size=(512, 512))
    right_resized = resize_with_padding(right_src, target_size=(512, 512))

    left_gray, left_mask = get_clean_foot_step(left_resized, target_size=(512, 512))
    right_gray, right_mask = get_clean_foot_step(right_resized, target_size=(512, 512))

    right_contrast_threshold = get_maximum_contrast_threshold(right_gray, right_mask)
    left_contrast_threshold = get_maximum_contrast_threshold(left_gray, left_mask)


    cv2.imshow("Left Foot - Contrast", left_contrast_threshold)
    cv2.imshow("Right Foot - Contrast", right_contrast_threshold)
    cv2.imshow("Left Foot - Resized 512x512", left_resized)
    cv2.imshow("Right Foot - Resized 512x512", right_resized)
    cv2.waitKey(0)
