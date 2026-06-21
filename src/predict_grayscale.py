"""
predict_grayscale.py - Применение лучших параметров (grayscale)
Использует параметры из ga_grayscale_results.json
"""

import cv2
import numpy as np
import json
import os
import glob

from auto_analysis import (
    get_foot_mask,
    get_largest_contour,
    find_AB_points_final,
    calculate_strieter_index_full,
    draw_contours,
    get_foot_type
)


# =============================================================================
# ЗАГРУЗКА ЛУЧШИХ ПАРАМЕТРОВ
# =============================================================================

def load_best_params():
    """Загружает лучшие параметры."""
    if os.path.exists('ga_grayscale_results.json'):
        with open('ga_grayscale_results.json', 'r') as f:
            data = json.load(f)
        return data['best_params']

    if os.path.exists('best_grayscale.json'):
        with open('best_grayscale.json', 'r') as f:
            data = json.load(f)
        return data['params']

    print("❌ Файл с параметрами не найден!")
    return None


# =============================================================================
# ФУНКЦИЯ ДЛЯ GRAYSCALE
# =============================================================================

def get_plantogram_mask_grayscale(image, foot_mask, threshold, min_area):
    """Выделение плантограммы через grayscale."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(mask, foot_mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(mask)

    plant_contour = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            cv2.drawContours(clean_mask, [cnt], -1, 255, -1)
            if plant_contour is None or area > cv2.contourArea(plant_contour):
                plant_contour = cnt

    return clean_mask, plant_contour


# =============================================================================
# ОБРАБОТКА ОДНОЙ СТОПЫ
# =============================================================================

def process_single_foot(image, params):
    """
    Обрабатывает одну стопу с grayscale параметрами.
    """
    threshold = params['THRESHOLD']
    min_area = params['MIN_AREA']

    foot_mask = get_foot_mask(image)
    clean_mask, plant_contour = get_plantogram_mask_grayscale(
        image, foot_mask, threshold, min_area
    )

    if plant_contour is None:
        vis = draw_contours(image, foot_mask, clean_mask)
        cv2.putText(vis, "НЕТ КОНТУРА", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return None, None, vis

    A, B = find_AB_points_final(plant_contour)
    vis = draw_contours(image, foot_mask, clean_mask)
    index, V, G, D, foot_type = calculate_strieter_index_full(plant_contour, A, B, vis)

    return index, foot_type, vis


# =============================================================================
# ОСНОВНАЯ ПРОГРАММА
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🦶 ПРИМЕНЕНИЕ GRAYSCALE ПАРАМЕТРОВ")
    print("=" * 60)

    # Загружаем параметры
    params = load_best_params()
    if params is None:
        exit()

    print(f"\n📊 Параметры:")
    print(f"  THRESHOLD: {params['THRESHOLD']}")
    print(f"  MIN_AREA: {params['MIN_AREA']}")
    print("=" * 60 + "\n")

    # Получаем все стопы
    image_files = glob.glob("./split_feet/*.png")
    image_files = sorted(image_files)

    print(f"📊 Найдено стоп: {len(image_files)}\n")

    success = 0
    failed = 0

    for img_path in image_files:
        name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"🔄 {name}:")

        image = cv2.imread(img_path)
        if image is None:
            print(f"  ❌ Не удалось загрузить")
            failed += 1
            continue

        index, foot_type, vis = process_single_foot(image, params)

        if index is not None:
            success += 1
            print(f"  ✅ Индекс: {index:.1f} ({foot_type})")
        else:
            failed += 1
            print(f"  ❌ Контур не найден")

        # Показываем
        cv2.namedWindow("RESULT", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("RESULT", 700, 800)
        if vis is not None:
            cv2.imshow("RESULT", vis)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("📊 СТАТИСТИКА")
    print("=" * 60)
    print(f"  ✅ Успешно: {success}/{len(image_files)}")
    print(f"  ❌ Ошибок: {failed}/{len(image_files)}")
    print("=" * 60)