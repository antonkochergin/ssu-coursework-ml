import cv2
import numpy as np
from rembg import remove


def resize_with_padding(image, target_size=(512, 512)):
    h, w = image.shape[:2]
    tw, th = target_size

    # Вычисляем коэффициент масштабирования
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


def get_clean_foot_step(image_path, target_size=(512, 512)):
    src = cv2.imread(image_path)

    # Удаление фона (получаем RGBA)
    no_bg_rgba = remove(src)

    # Извлечение маски и цветного изображения
    mask = no_bg_rgba[:, :, 3]
    foot_color = no_bg_rgba[:, :, :3]

    gray_foot = cv2.cvtColor(foot_color, cv2.COLOR_BGR2GRAY)

    # Очистка маски (Морфология)
    kernel = np.ones((5, 5), np.uint8)
    refined_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    final_gray = resize_with_padding(gray_foot, target_size)
    final_mask = resize_with_padding(refined_mask, target_size)

    return final_gray, final_mask

if __name__ == "__main__":
    path = "./foots/1/IMG_0315.jpg"

    gray, mask = get_clean_foot_step(path)

    if gray is not None:
        cv2.imshow("Grayscale Foot (Padded)", gray)
        cv2.imshow("Mask (Padded)", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
