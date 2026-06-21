"""
interactive_grayscale.py - Интерактивный режим (ПРАВИЛЬНЫЙ)
Точка B ищется на внутреннем крае ПЛАНТОГРАММЫ (зеленой области)
"""

import cv2
import numpy as np
import json
import os
import glob
import hashlib

from auto_analysis import (
    get_foot_mask,
    get_largest_contour,
    find_AB_points_final,
    calculate_strieter_index_full,
    draw_contours,
    get_foot_type,
    FOOT_TYPE_BOUNDARIES,
    find_A_at_widest_toe,
    find_B_at_leftmost_heel
)

# =============================================================================
# КОНСТАНТЫ
# =============================================================================

SPLIT_DIR = "./split_feet/"
PROGRESS_FILE = "progress_grayscale.json"
RESULTS_VIS_DIR = "./results_vis_grayscale/"
MAX_DISPLAY_SIZE = 800

# Кэш для масок стоп
FOOT_MASK_CACHE = {}


# =============================================================================
# ФУНКЦИЯ ДЛЯ GRAYSCALE (с улучшенным соединением)
# =============================================================================

def get_foot_mask_cached(image):
    """Кэшированная версия get_foot_mask"""
    img_hash = hashlib.md5(image.tobytes()).hexdigest()
    if img_hash not in FOOT_MASK_CACHE:
        FOOT_MASK_CACHE[img_hash] = get_foot_mask(image)
    return FOOT_MASK_CACHE[img_hash]


def get_plantogram_mask_grayscale(image, foot_mask, threshold, min_area, close_size):
    """
    Выделение плантограммы через grayscale + порог.
    close_size — размер ядра для морфологического замыкания.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # CLAHE для усиления контраста
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Пороговая обработка (ищем СВЕТЛЫЕ области)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Расширение для соединения
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel_dilate, iterations=1)

    # Морфологическое замыкание (соединяет разрывы)
    if close_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Ограничиваем маской стопы
    mask = cv2.bitwise_and(mask, foot_mask)

    # Оставляем только крупные области
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(mask)

    plant_contour = None
    max_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            cv2.drawContours(clean_mask, [cnt], -1, 255, -1)
            if area > max_area:
                max_area = area
                plant_contour = cnt

    # Визуализация
    vis = draw_contours(image, foot_mask, clean_mask)
    cv2.putText(vis, f"Area: {max_area:.0f}", (10, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    if close_size > 0:
        cv2.putText(vis, f"CLOSE: {close_size}", (10, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return clean_mask, plant_contour, vis, max_area


# =============================================================================
# ПОИСК ТОЧКИ B НА ПЛАНТОГРАММЕ (ПРАВИЛЬНЫЙ ПОДХОД)
# =============================================================================

def find_B_on_plantogram(plant_contour):
    """
    Находит точку B на внутреннем крае ПЛАНТОГРАММЫ.
    Используется стандартная функция из auto_analysis.
    """
    if plant_contour is None:
        return None
    return find_B_at_leftmost_heel(plant_contour)


# =============================================================================
# ОБРАБОТКА ОДНОЙ СТОПЫ
# =============================================================================

def process_single_foot(image, threshold, min_area, close_size):
    # Маска стопы (с кэшированием)
    foot_mask = get_foot_mask_cached(image)

    # Контур плантограммы
    clean_mask, plant_contour, vis, max_area = get_plantogram_mask_grayscale(
        image, foot_mask, threshold, min_area, close_size
    )

    if plant_contour is None:
        return None, None, None

    # Точка A — на контуре плантограммы
    A = find_A_at_widest_toe(plant_contour)

    # Точка B — на внутреннем крае плантограммы (ПРАВИЛЬНО!)
    B = find_B_on_plantogram(plant_contour)

    if B is None:
        return None, None, vis

    # Расчет индекса
    index, V, G, D, foot_type = calculate_strieter_index_full(plant_contour, A, B, vis)

    return index, foot_type, vis


# =============================================================================
# ЗАГРУЗКА/СОХРАНЕНИЕ ПРОГРЕССА
# =============================================================================

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {'completed': [], 'skipped': [], 'results': {}}


def save_progress(progress):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


# =============================================================================
# ИНТЕРАКТИВНЫЙ РЕЖИМ
# =============================================================================

def interactive_grayscale():
    print("\n" + "=" * 60)
    print("🖥️ ИНТЕРАКТИВНЫЙ РЕЖИМ (ПРАВИЛЬНЫЙ)")
    print("=" * 60)
    print("📂 Исходная папка: ./split_feet/")
    print("=" * 60 + "\n")

    os.makedirs(RESULTS_VIS_DIR, exist_ok=True)

    all_feet = []
    for file in sorted(os.listdir(SPLIT_DIR)):
        if file.endswith('.png') and '_' in file:
            all_feet.append(file.replace('.png', ''))

    if not all_feet:
        print("❌ Нет разделенных стоп")
        return

    progress = load_progress()
    results = progress.get('results', {})
    completed = set(progress.get('completed', []))
    skipped = set(progress.get('skipped', []))

    to_process = []
    for foot in all_feet:
        if foot not in completed:
            to_process.append(foot)

    if not to_process:
        print("\n✅ Все стопы обработаны!")
        print(f"Всего: {len(completed)}")
        return

    print(f"\nОсталось: {len(to_process)} стоп")
    print(f"Всего: {len(all_feet)} стоп")
    print(f"Готово: {len(completed)}")
    print(f"Пропущено: {len(skipped)}")

    # Окно управления
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controls", 450, 200)
    cv2.moveWindow("Controls", 0, 0)

    cv2.createTrackbar("THRESHOLD", "Controls", 127, 255, lambda x: None)
    cv2.createTrackbar("MIN_AREA (x1000)", "Controls", 30, 200, lambda x: None)
    cv2.createTrackbar("CLOSE_SIZE", "Controls", 0, 30, lambda x: None)

    for foot_name in to_process:
        print(f"\n{'=' * 50}")
        print(f"Обработка: {foot_name}")
        print(f"Прогресс: {len(completed) + 1}/{len(all_feet)}")
        print(f"{'=' * 50}")
        print("Управление:")
        print("  S - Сохранить")
        print("  Пробел - Пропустить")
        print("  R - Сбросить")
        print("  ESC - Выйти")
        print("  (Точка B на ПЛАНТОГРАММЕ)")

        img_path = os.path.join(SPLIT_DIR, f"{foot_name}.png")
        image = cv2.imread(img_path)
        if image is None:
            print(f"❌ Не удалось загрузить")
            continue

        h, w = image.shape[:2]
        scale = min(MAX_DISPLAY_SIZE / w, MAX_DISPLAY_SIZE / h, 1.0)
        display_size = (int(w * scale), int(h * scale))

        # Первый показ
        threshold = cv2.getTrackbarPos("THRESHOLD", "Controls")
        min_area = cv2.getTrackbarPos("MIN_AREA (x1000)", "Controls") * 1000
        close_size = cv2.getTrackbarPos("CLOSE_SIZE", "Controls")

        index, foot_type, display_img = process_single_foot(
            image, threshold, min_area, close_size
        )

        if display_img is not None:
            display_img = cv2.resize(display_img, display_size)
            cv2.imshow("Interactive Grayscale", display_img)

        while True:
            threshold = cv2.getTrackbarPos("THRESHOLD", "Controls")
            min_area = cv2.getTrackbarPos("MIN_AREA (x1000)", "Controls") * 1000
            close_size = cv2.getTrackbarPos("CLOSE_SIZE", "Controls")

            index, foot_type, display_img = process_single_foot(
                image, threshold, min_area, close_size
            )

            if display_img is not None:
                display_img = cv2.resize(display_img, display_size)

                # Информация на изображении
                info_text = [
                    f"Стопа: {foot_name}",
                    f"TH: {threshold} | MIN: {min_area} | CLOSE: {close_size}",
                    f"Индекс: {index:.1f}" if index else "Индекс: -",
                    f"Тип: {foot_type}" if foot_type else "Тип: -",
                    f"Прогресс: {len(completed) + 1}/{len(all_feet)}"
                ]

                y_offset = 25
                for text in info_text:
                    cv2.putText(display_img, text, (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y_offset += 25

                cv2.imshow("Interactive Grayscale", display_img)

            key = cv2.waitKey(30) & 0xFF

            if key == ord('s'):
                if index is not None:
                    results[foot_name] = {
                        'index': float(index),
                        'foot_type': foot_type,
                        'threshold': threshold,
                        'min_area': min_area,
                        'close_size': close_size
                    }
                    completed.add(foot_name)
                    if foot_name in skipped:
                        skipped.remove(foot_name)

                    vis_path = os.path.join(RESULTS_VIS_DIR, f"{foot_name}_vis.png")
                    cv2.imwrite(vis_path, display_img)
                    print(f"  ✅ Сохранено: {foot_name} (индекс: {index:.1f})")

                    progress['results'] = results
                    progress['completed'] = list(completed)
                    progress['skipped'] = list(skipped)
                    save_progress(progress)
                    break
                else:
                    print("  ⚠️ Нет контура. Попробуйте другие параметры.")

            elif key == ord(' '):
                skipped.add(foot_name)
                progress['results'] = results
                progress['completed'] = list(completed)
                progress['skipped'] = list(skipped)
                save_progress(progress)
                print(f"  ⏭️ Пропущено: {foot_name}")
                break

            elif key == ord('r'):
                cv2.setTrackbarPos("THRESHOLD", "Controls", 127)
                cv2.setTrackbarPos("MIN_AREA (x1000)", "Controls", 30)
                cv2.setTrackbarPos("CLOSE_SIZE", "Controls", 0)
                print("  🔄 Сброс")

            elif key == 27:
                print("\n⚠️ Выход по ESC")
                cv2.destroyAllWindows()
                return

    print("\n" + "=" * 60)
    print("✅ ВСЕ СТОПЫ ОБРАБОТАНЫ!")
    print(f"Обработано: {len(completed)}")
    print(f"Пропущено: {len(skipped)}")
    print("=" * 60)


if __name__ == "__main__":
    interactive_grayscale()