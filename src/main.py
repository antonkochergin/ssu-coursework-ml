import cv2
import numpy as np
from rembg import remove


def resize_with_padding(image, target_size=(512, 512)):
    """Изменяет размер, сохраняя пропорции, и добавляет черные полосы."""
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


def get_clean_foot_step(src_image, target_size=(512, 512)):
    """
    ЭТАП 1: Получение чистой маски и серого изображения с сеткой.
    """
    # 1. Удаляем фон с оригинала (без сетки)
    print("Удаление фона...")
    no_bg_rgba = remove(src_image)

    # 2. Обработка маски для максимальной четкости (до ресайза!)
    mask = no_bg_rgba[:, :, 3]
    # Делаем маску жесткой (убираем размытость rembg)
    _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Легкая морфология (ядро 3x3 не съест пальцы)
    kernel = np.ones((3, 3), np.uint8)
    refined_mask = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel)
    refined_mask = cv2.medianBlur(refined_mask, 3)  # Сглаживаем "ступеньки"

    # 3. Подготовка изображения с сеткой
    img_with_grid = src_image.copy()
    h_orig, w_orig = img_with_grid.shape[:2]
    cell_size = 150
    # Рисуем жирную сетку (thickness=3), чтобы она не стерлась при сжатии
    for x in range(0, w_orig, cell_size):
        cv2.line(img_with_grid, (x, 0), (x, h_orig), (0, 0, 0), 4)
    for y in range(0, h_orig, cell_size):
        cv2.line(img_with_grid, (0, y), (w_orig, y), (0, 0, 0), 4)

    # 4. Перевод в Grayscale и ресайз
    gray_grid = cv2.cvtColor(img_with_grid, cv2.COLOR_BGR2GRAY)

    final_gray = resize_with_padding(gray_grid, target_size)
    final_mask = resize_with_padding(refined_mask, target_size)

    return final_gray, final_mask


def get_maximum_contrast_threshold(gray_image, mask_refined):
    """
    ЭТАП 2: Создание финального бинарного изображения (Рис. 5).
    """
    print("Применение контрастности...")
    # Жесткий порог для выделения сетки внутри стопы
    _, binary = cv2.threshold(gray_image, 45, 255, cv2.THRESH_BINARY)

    # Накладываем маску, чтобы вернуть идеальный контур
    result = cv2.bitwise_and(binary, mask_refined)

    # Финальная бинаризация для "звенящей" четкости краев
    _, result = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)

    return result


# --- ОСНОВНОЙ БЛОК (MAIN) ---
if __name__ == "__main__":
    path = "./foots/1/IMG_0315.jpg"
    src = cv2.imread(path)

    if src is None:
        print(f"Ошибка: Файл {path} не найден.")
        exit()

    # 1. Вызываем функцию (передаем чистый src)
    # Она возвращает: серое фото с сеткой и аккуратную маску
    gray_padded, mask_padded = get_clean_foot_step(src)

    # 2. Создаем Рис. 5 (Максимальная контрастность + сетка)
    contrast_padded = get_maximum_contrast_threshold(gray_padded, mask_padded)

    # 3. Визуализация
    cv2.imshow("Fig 4. Grayscale + Grid", gray_padded)
    cv2.imshow("Fig 5. Final Binary Contour", contrast_padded)

    print("Готово. Нажмите любую клавишу для выхода.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()