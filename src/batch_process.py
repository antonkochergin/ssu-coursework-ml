"""
process_best.py - Обработка всех стоп с лучшими параметрами из ГА (фитнес 88.22%)
Использует параметры из best_autosave.json
"""

import cv2
import numpy as np
from rembg import remove
import os
import glob
import json

# =============================================================================
# ЛУЧШИЕ ПАРАМЕТРЫ ИЗ ГА (best_autosave.json)
# =============================================================================

# Параметры для маски стопы
FOOT_MASK_THRESHOLD = 100
MORPHOLOGY_KERNEL_SIZE = 5

# Параметры для маски плантограммы (зеленый цвет в HSV)
# ИЗ best_autosave.json - фитнес 88.22%
HSV_LOWER_GREEN = np.array([72, 179, 101])
HSV_UPPER_GREEN = np.array([99, 244, 183])
MIN_PLANTOGRAM_AREA = 54104

# Параметры для поиска точек A и B
# ИЗ best_autosave.json
TOE_AREA_RATIO = 0.4361466882431499
HEEL_AREA_RATIO = 0.6509073113319302
TOE_SEARCH_STEP = 5
TOE_Y_TOLERANCE = 3

# Параметры для расчета индекса Штриттера
# ИЗ best_autosave.json
PERP_DIST_TOLERANCE = 23
PERP_DIST_TOLERANCE_FALLBACK = 12

# Параметры для визуализации
PERP_LINE_EXTEND = 2000
FONT_SCALE = 0.8
FONT_THICKNESS = 2
POINT_SIZE_LARGE = 12
POINT_SIZE_MEDIUM = 10
CROSS_SIZE = 15

# Границы для классификации типа стопы
FOOT_TYPE_BOUNDARIES = [
    (36.0, "Высокосводчатая (полая)"),
    (43.0, "Повышенный свод"),
    (50.0, "Нормальная"),
    (60.0, "Уплощенная"),
    (float('inf'), "Плоскостопие")
]


# =============================================================================
# ОБРАБОТКА ИЗОБРАЖЕНИЯ
# =============================================================================

def get_foot_mask(src_image):
    """Получает бинарную маску стопы."""
    no_bg = remove(src_image)
    foot_mask = no_bg[:, :, 3]
    _, foot_mask = cv2.threshold(foot_mask, FOOT_MASK_THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel = np.ones((MORPHOLOGY_KERNEL_SIZE, MORPHOLOGY_KERNEL_SIZE), np.uint8)
    foot_mask = cv2.morphologyEx(foot_mask, cv2.MORPH_CLOSE, kernel)
    return foot_mask


def get_plantogram_mask(src_image, foot_mask):
    """Получает маску плантограммы с лучшими параметрами."""
    hsv = cv2.cvtColor(src_image, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, HSV_LOWER_GREEN, HSV_UPPER_GREEN)
    green_mask = cv2.bitwise_and(green_mask, foot_mask)

    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(green_mask)
    for cnt in contours:
        if cv2.contourArea(cnt) > MIN_PLANTOGRAM_AREA:
            cv2.drawContours(clean_mask, [cnt], -1, 255, -1)

    return clean_mask


def get_largest_contour(mask):
    """Находит самый большой контур в маске."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def draw_contours(image, foot_mask, plant_mask):
    """Отрисовывает контуры стопы и плантограммы."""
    result = image.copy()
    foot_contours, _ = cv2.findContours(foot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, foot_contours, -1, (255, 0, 0), 2)
    plant_contours, _ = cv2.findContours(plant_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, plant_contours, -1, (0, 0, 255), 2)
    return result


# =============================================================================
# ПОИСК ТОЧЕК A И B
# =============================================================================

def find_A_at_widest_toe(plant_contour):
    """Находит точку A как самую широкую точку в носочной части стопы."""
    pts = plant_contour[:, 0, :]
    y_min, y_max = np.min(pts[:, 1]), np.max(pts[:, 1])
    height = y_max - y_min

    toe_threshold = y_min + height * TOE_AREA_RATIO
    toe_pts = pts[pts[:, 1] < toe_threshold]

    if len(toe_pts) == 0:
        toe_threshold = y_min + height * 0.25
        toe_pts = pts[pts[:, 1] < toe_threshold]

    max_width = 0
    best_left = None

    for y in np.arange(int(y_min), int(toe_threshold), TOE_SEARCH_STEP):
        level_pts = pts[np.abs(pts[:, 1] - y) < TOE_Y_TOLERANCE]
        if len(level_pts) > 1:
            left_x, right_x = np.min(level_pts[:, 0]), np.max(level_pts[:, 0])
            width = right_x - left_x
            if width > max_width:
                max_width = width
                left_pts = level_pts[level_pts[:, 0] == left_x]
                if len(left_pts) > 0:
                    best_left = tuple(left_pts[0])

    if best_left is None:
        min_x = np.min(toe_pts[:, 0])
        best_left_pts = toe_pts[toe_pts[:, 0] == min_x]
        best_left = tuple(best_left_pts[0]) if len(best_left_pts) > 0 else (0, 0)

    return best_left


def find_B_at_leftmost_heel(plant_contour):
    """Находит точку B как крайнюю левую точку в области пятки."""
    pts = plant_contour[:, 0, :]
    y_min, y_max = np.min(pts[:, 1]), np.max(pts[:, 1])
    height = y_max - y_min

    heel_threshold = y_min + height * HEEL_AREA_RATIO
    heel_pts = pts[pts[:, 1] > heel_threshold]

    if len(heel_pts) == 0:
        heel_threshold = y_min + height * 0.7
        heel_pts = pts[pts[:, 1] > heel_threshold]

    B_idx = np.argmin(heel_pts[:, 0])
    return tuple(heel_pts[B_idx])


def find_AB_points_final(plant_contour):
    """Находит точки A и B."""
    if plant_contour is None:
        return None, None
    A = find_A_at_widest_toe(plant_contour)
    B = find_B_at_leftmost_heel(plant_contour)
    return A, B


# =============================================================================
# РАСЧЕТ ИНДЕКСА ШТРИТТЕРА
# =============================================================================

def draw_points_on_image(debug_img, A, B, V, G, D, perp_dx, perp_dy):
    """Отрисовка всех точек, линий и подписей."""
    cv2.line(debug_img, A, B, (0, 255, 255), 2)

    h, w = debug_img.shape[:2]
    perp_len = max(PERP_LINE_EXTEND, max(h, w) * 2)
    start_point = (int(V[0] - perp_dx * perp_len), int(V[1] - perp_dy * perp_len))
    end_point = (int(V[0] + perp_dx * perp_len), int(V[1] + perp_dy * perp_len))
    cv2.line(debug_img, start_point, end_point, (200, 0, 255), 3)

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = FONT_THICKNESS
    font_scale = FONT_SCALE

    for point, color, label, offset in [
        (G, (255, 255, 0), "G", (-40, -25)),
        (D, (0, 255, 255), "D", (20, -20)),
        (V, (0, 255, 0), "V", (20, -15)),
    ]:
        cv2.circle(debug_img, point, POINT_SIZE_LARGE, color, -1)
        cv2.circle(debug_img, point, POINT_SIZE_LARGE, (0, 0, 0), 2)
        cv2.putText(debug_img, label,
                    (point[0] + offset[0], point[1] + offset[1]),
                    font, font_scale, (0, 0, 0), thickness + 1)
        cv2.putText(debug_img, label,
                    (point[0] + offset[0], point[1] + offset[1]),
                    font, font_scale, color, thickness)

    cv2.circle(debug_img, A, POINT_SIZE_MEDIUM, (255, 0, 0), -1)
    cv2.circle(debug_img, A, POINT_SIZE_MEDIUM, (0, 0, 0), 2)
    cv2.putText(debug_img, "A", (A[0] - 35, A[1] - 20),
                font, font_scale, (0, 0, 0), thickness + 1)
    cv2.putText(debug_img, "A", (A[0] - 35, A[1] - 20),
                font, font_scale, (255, 0, 0), thickness)

    cv2.circle(debug_img, B, POINT_SIZE_MEDIUM, (0, 0, 255), -1)
    cv2.circle(debug_img, B, POINT_SIZE_MEDIUM, (0, 0, 0), 2)
    cv2.putText(debug_img, "B", (B[0] - 35, B[1] + 30),
                font, font_scale, (0, 0, 0), thickness + 1)
    cv2.putText(debug_img, "B", (B[0] - 35, B[1] + 30),
                font, font_scale, (0, 0, 255), thickness)

    cv2.line(debug_img, (V[0] - CROSS_SIZE, V[1]), (V[0] + CROSS_SIZE, V[1]), (0, 0, 0), 2)
    cv2.line(debug_img, (V[0], V[1] - CROSS_SIZE), (V[0], V[1] + CROSS_SIZE), (0, 0, 0), 2)


def get_foot_type(index):
    """Определяет тип стопы по индексу Штриттера."""
    if index is None:
        return None
    for threshold, name in FOOT_TYPE_BOUNDARIES:
        if index <= threshold:
            return name
    return "Плоскостопие"


def calculate_strieter_index_full(plant_contour, A, B, debug_img=None):
    """Полный расчет индекса Штриттера."""
    if plant_contour is None or A is None or B is None:
        return None, None, None, None, None

    V = ((A[0] + B[0]) // 2, (A[1] + B[1]) // 2)

    dx, dy = B[0] - A[0], B[1] - A[1]
    length = np.sqrt(dx * dx + dy * dy)
    if length == 0:
        return None, None, None, None, None

    perp_dx, perp_dy = -dy / length, dx / length

    pts = plant_contour[:, 0, :]
    projections = []

    for p in pts:
        vx, vy = p[0] - V[0], p[1] - V[1]
        proj = vx * perp_dx + vy * perp_dy
        dist_to_perp = abs(vx * (-perp_dy) + vy * perp_dx)

        if dist_to_perp < PERP_DIST_TOLERANCE:
            projections.append((proj, p))

    if len(projections) < 2:
        for p in pts:
            vx, vy = p[0] - V[0], p[1] - V[1]
            dist_to_perp = abs(vx * (-perp_dy) + vy * perp_dx)
            if dist_to_perp < PERP_DIST_TOLERANCE_FALLBACK:
                proj = vx * perp_dx + vy * perp_dy
                projections.append((proj, p))

    if len(projections) < 2:
        return None, None, None, None, None

    projections.sort(key=lambda x: x[0])
    min_proj, point_min = projections[0]
    max_proj, point_max = projections[-1]

    if point_min[0] < point_max[0]:
        G, D = point_min, point_max
        GD, VD = max_proj - min_proj, abs(max_proj)
    else:
        G, D = point_max, point_min
        GD, VD = max_proj - min_proj, abs(min_proj)

    index = (GD * 100) / VD if VD > 0 else None
    foot_type = get_foot_type(index)

    if debug_img is not None and index is not None:
        draw_points_on_image(debug_img, A, B, V, G, D, perp_dx, perp_dy)

    return index, V, G, D, foot_type


# =============================================================================
# ОБРАБОТКА ОДНОЙ СТОПЫ
# =============================================================================

def process_single_foot(image, name, save_vis=False, output_dir=None):
    """
    Обрабатывает одну стопу (уже разделенную).
    """
    if image is None:
        return None, None, None

    try:
        foot_mask = get_foot_mask(image)
        plant_mask = get_plantogram_mask(image, foot_mask)
        vis = draw_contours(image, foot_mask, plant_mask)

        plant_contour = get_largest_contour(plant_mask)
        if plant_contour is None:
            cv2.putText(vis, "НЕТ ПЛАНТОГРАММЫ", (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return vis, None, None

        A, B = find_AB_points_final(plant_contour)
        index, V, G, D, foot_type = calculate_strieter_index_full(plant_contour, A, B, vis)

        if index is not None and save_vis and output_dir and name:
            os.makedirs(output_dir, exist_ok=True)
            vis_path = os.path.join(output_dir, f"{name}_vis.png")
            cv2.imwrite(vis_path, vis)

        return vis, index, foot_type

    except Exception as e:
        print(f"  ❌ Ошибка: {str(e)}")
        return None, None, None


# =============================================================================
def show(name, img, max_w=900, max_h=700):
    """Отображает изображение с масштабированием."""
    if img is None:
        return
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    cv2.imshow(name, img)


# =============================================================================
# ОСНОВНАЯ ПРОГРАММА
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🦶 АНАЛИЗ СТОП С ЛУЧШИМИ ПАРАМЕТРАМИ")
    print("🎯 Фитнес: 88.22%")
    print("=" * 60)
    print("📂 Исходная папка: ./split_feet/")
    print("=" * 60 + "\n")

    print("ПАРАМЕТРЫ (из best_autosave.json):")
    print(f"  HSV_LOWER: {HSV_LOWER_GREEN}")
    print(f"  HSV_UPPER: {HSV_UPPER_GREEN}")
    print(f"  MIN_AREA: {MIN_PLANTOGRAM_AREA}")
    print(f"  TOE_RATIO: {TOE_AREA_RATIO:.4f}")
    print(f"  HEEL_RATIO: {HEEL_AREA_RATIO:.4f}")
    print(f"  PERP_TOL: {PERP_DIST_TOLERANCE}")
    print("=" * 60 + "\n")

    # Получаем все изображения из папки split_feet/
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(glob.glob(os.path.join("./split_feet/", ext)))
    image_files = sorted(list(set(image_files)))

    if not image_files:
        print("❌ Нет изображений в папке ./split_feet/")
        exit()

    print(f"📊 Найдено стоп: {len(image_files)}\n")

    all_results = []
    success_count = 0
    no_contour_count = 0

    for idx, img_path in enumerate(image_files, 1):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"🔄 {idx}/{len(image_files)}: {base_name}")

        src = cv2.imread(img_path)
        if src is None:
            print(f"  ⚠️ Не удалось загрузить")
            continue

        vis, index, foot_type = process_single_foot(
            src, base_name,
            save_vis=True,
            output_dir="./process_best_results/"
        )

        if index is not None:
            success_count += 1
            print(f"  ✅ Индекс: {index:.1f} ({foot_type})")
            all_results.append({
                'file': base_name,
                'index': index,
                'type': foot_type
            })
        else:
            no_contour_count += 1
            if vis is not None:
                cv2.putText(vis, f"НЕТ КОНТУРА", (50, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # Показываем
        if vis is not None:
            cv2.namedWindow("RESULT", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("RESULT", 700, 800)
            show("RESULT", vis)

        key = cv2.waitKey(0)
        if key == 27:  # ESC
            print("\n⚠️ Выход")
            break

    cv2.destroyAllWindows()

    # Статистика
    print("\n" + "=" * 60)
    print("📊 СТАТИСТИКА")
    print("=" * 60)
    print(f"  ✅ Найдено контуров: {success_count}/{len(image_files)}")
    print(f"  ❌ Нет контура: {no_contour_count}/{len(image_files)}")

    if all_results:
        indices = [r['index'] for r in all_results]
        print(f"\n📈 ИНДЕКСЫ:")
        print(f"  Средний: {np.mean(indices):.1f}")
        print(f"  Минимальный: {np.min(indices):.1f}")
        print(f"  Максимальный: {np.max(indices):.1f}")

        types = {}
        for r in all_results:
            t = r['type']
            types[t] = types.get(t, 0) + 1
        print(f"\n📊 ТИПЫ:")
        for t, c in sorted(types.items()):
            print(f"  {t}: {c}")

    print("=" * 60)