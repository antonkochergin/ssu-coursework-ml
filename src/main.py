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

    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    bboxes = sorted([cv2.boundingRect(c) for c in contours], key=lambda b: b[0])
    left_edge = bboxes[0][0] + bboxes[0][2]  # правый край левой стопы
    right_edge = bboxes[1][0]  # левый край правой стопы

    split_x = (left_edge + right_edge) // 2

    left_foot = src_image[:, :split_x]
    left_foot = cv2.rotate(left_foot, cv2.ROTATE_180)

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


def get_clean_foot_step(src_image, target_size=None):
    """
    Получение чистой маски и серого изображения с сеткой.
    Если target_size=None — возвращает без ресайза.
    """
    no_bg_rgba = remove(src_image)

    mask = no_bg_rgba[:, :, 3]

    _, mask_binary = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    refined_mask = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel)
    refined_mask = cv2.medianBlur(refined_mask, 5)

    img_with_grid = src_image.copy()
    h_orig, w_orig = img_with_grid.shape[:2]
    cell_size = 60
    for x in range(0, w_orig, cell_size):
        cv2.line(img_with_grid, (x, 0), (x, h_orig), (0, 0, 0), 3)
    for y in range(0, h_orig, cell_size):
        cv2.line(img_with_grid, (0, y), (w_orig, y), (0, 0, 0), 3)

    gray_grid = cv2.cvtColor(img_with_grid, cv2.COLOR_BGR2GRAY)

    if target_size is not None:
        final_gray = resize_with_padding(gray_grid, target_size)
        final_mask = resize_with_padding(refined_mask, target_size)
    else:
        final_gray = gray_grid
        final_mask = refined_mask

    return final_gray, final_mask


def get_maximum_contrast_threshold(gray_image, mask_refined):
    """
    Создание финального бинарного изображения
    """

    _, binary = cv2.threshold(gray_image, 45, 255, cv2.THRESH_BINARY)

    result = cv2.bitwise_and(binary, mask_refined)

    _, result = cv2.threshold(result, 100, 255, cv2.THRESH_BINARY)

    return result


def imshow_fit(winname, image, max_width=900, max_height=700):
    """
    Показывает изображение, автоматически уменьшая под размер экрана.
    """
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        display = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        display = image

    cv2.imshow(winname, display)


if __name__ == "__main__":
    path = "./foots/1/IMG_0315.jpg"
    # path = "./foots/2/IMG_0320.jpg"
    # path = "./foots/4/IMG_0302.jpg"
    # path = "./foots/5/IMG_0256.JPG"
    # path = "./foots/6/IMG_0356.jpg"
    TARGET = (512, 512)

    try:
        src = cv2.imread(path)

        left_src, right_src, debug_split = split_feet_smart(src, debug=True)

        left_rembg = remove(left_src)
        # imshow_fit("DEBUG: raw rembg alpha", raw_rembg[:, :, 3])
        imshow_fit("left rembg color", left_rembg[:, :, :3])

        right_rembg = remove(right_src)
        imshow_fit("right rembg color", right_rembg[:, :, :3])

        left_gray, left_mask = get_clean_foot_step(left_src, target_size=None)
        right_gray, right_mask = get_clean_foot_step(right_src, target_size=None)

        left_resized = resize_with_padding(left_src, target_size=TARGET)
        left_gray = resize_with_padding(left_gray, target_size=TARGET)
        left_mask = resize_with_padding(left_mask, target_size=TARGET)

        right_resized = resize_with_padding(right_src, target_size=TARGET)
        right_gray = resize_with_padding(right_gray, target_size=TARGET)
        right_mask = resize_with_padding(right_mask, target_size=TARGET)

        left_contrast = get_maximum_contrast_threshold(left_gray, left_mask)
        right_contrast = get_maximum_contrast_threshold(right_gray, right_mask)

        # cv2.imshow("Left Foot - Mask", left_mask)
        cv2.imshow("Left Foot - Contrast", left_contrast)
        cv2.imshow("Left Foot - Resized 512x512", left_resized)

        # cv2.imshow("Right Foot - Mask", right_mask)
        cv2.imshow("Right Foot - Contrast", right_contrast)
        cv2.imshow("Right Foot - Resized 512x512", right_resized)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("Не получилось - проверь файл ")
