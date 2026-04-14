import cv2
import numpy as np
from rembg import remove


# Используем твои функции без изменений (resize_with_padding, get_maximum_contrast_threshold)
# Но немного оптимизируем основной цикл

def process_video(video_path, output_path, target_size=(512, 512)):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Ошибка: Не удалось открыть видео.")
        return

    # Получаем параметры исходного видео
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Итоговое видео будет квадратным, как мы и задали в target_size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, target_size, isColor=False)

    print("Начинается обработка видео... Это может занять время.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Обработка кадра (используем твою логику)
        # ВАЖНО: rembg внутри get_clean_foot_step будет работать долго
        gray_padded, mask_padded = get_clean_foot_step(frame, target_size)

        # 2. Финальный результат (Рис. 5 для видео)
        contrast_frame = get_maximum_contrast_threshold(gray_padded, mask_padded)

        # Показываем процесс
        cv2.imshow("Processing Video...", contrast_frame)

        # Записываем кадр в файл
        out.write(contrast_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Видео сохранено в {output_path}")

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


def get_clean_foot_step(src_image, target_size=(512, 512)):
    """
    Получение чистой маски и серого изображения с сеткой.
    """
    no_bg_rgba = remove(src_image)


    mask = no_bg_rgba[:, :, 3]

    _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)


    kernel = np.ones((3, 3), np.uint8)
    refined_mask = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel)
    refined_mask = cv2.medianBlur(refined_mask, 3)  # Сглаживаем "ступеньки"


    img_with_grid = src_image.copy()
    h_orig, w_orig = img_with_grid.shape[:2]
    cell_size = 150

    for x in range(0, w_orig, cell_size):
        cv2.line(img_with_grid, (x, 0), (x, h_orig), (0, 0, 0), 4)
    for y in range(0, h_orig, cell_size):
        cv2.line(img_with_grid, (0, y), (w_orig, y), (0, 0, 0), 4)

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

    # Финальная бинаризация для "звенящей" четкости краев
    _, result = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)

    return result


if __name__ == "__main__":
    video_input = "./foots/1/MVI_0318.MOV"  # Путь к твоему видео
    video_output = "foot_analysis_result.mp4"

    process_video(video_input, video_output)

