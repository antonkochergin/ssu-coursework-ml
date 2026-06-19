import cv2
import numpy as np
from rembg import remove
import os
from datetime import datetime

# =============================================================================
# КОНСТАНТЫ
# =============================================================================

FOOT_MASK_THRESHOLD = 100
MORPHOLOGY_KERNEL_SIZE = 5

HSV_LOWER_GREEN = np.array([74, 149, 141])
HSV_UPPER_GREEN = np.array([86, 255, 194])
MIN_PLANTOGRAM_AREA = 50000

TOE_AREA_RATIO = 0.3
HEEL_AREA_RATIO = 0.65
TOE_SEARCH_STEP = 5
TOE_Y_TOLERANCE = 3

PERP_DIST_TOLERANCE = 10
PERP_DIST_TOLERANCE_FALLBACK = 20

PERP_LINE_EXTEND = 2000
FONT_SCALE = 4.8            # было 2.4 — увеличено в 2 раза
FONT_THICKNESS = 10         # было 5
POINT_SIZE_LARGE = 56       # было 28
POINT_SIZE_MEDIUM = 44      # было 22
CROSS_SIZE = 60             # было 30

# Толщина линий (×9 от исходной: ×6 × 1.5)
LINE_THICKNESS_CONTOUR = 18       # было 2 → 12 → 18
LINE_THICKNESS_DIVIDER = 18       # было 2 → 12 → 18
LINE_THICKNESS_RECT = 18          # было 2 → 12 → 18
LINE_THICKNESS_AB = 27            # было 3 → 18 → 27
LINE_THICKNESS_PERP = 36          # было 4 → 24 → 36
LINE_THICKNESS_CROSS = 18         # было 2 → 12 → 18
CIRCLE_OUTLINE = 27               # было 3 → 18 → 27
TEXT_OUTLINE = 18                 # было 2 → 12 → 18

FOOT_TYPE_BOUNDARIES = [
    (36.0, "Высокосводчатая (полая)"),
    (43.0, "Повышенный свод"),
    (50.0, "Нормальная"),
    (60.0, "Уплощенная"),
    (float('inf'), "Плоскостопие")
]


# =============================================================================
# СОЗДАНИЕ ПАПКИ ДЛЯ СКРИНШОТОВ
# =============================================================================

def create_screenshots_dir():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = "screenshots"
    session_dir = os.path.join(base_dir, f"session_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)
    return session_dir


# =============================================================================
# СОХРАНЕНИЕ ИЗОБРАЖЕНИЙ
# =============================================================================

def save_step(session_dir, filename, img):
    filepath = os.path.join(session_dir, filename)
    cv2.imwrite(filepath, img)
    print(f"  -> {filepath}")
    return filepath


def mask_to_color(mask, color=(255, 255, 255)):
    result = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    result[mask == 255] = color
    return result


# =============================================================================
# РАЗДЕЛЕНИЕ СТОП
# =============================================================================

def split_feet_smart(src_image):
    h, w = src_image.shape[:2]

    no_bg = remove(src_image)
    mask = no_bg[:, :, 3]
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    bboxes = sorted([cv2.boundingRect(c) for c in contours], key=lambda b: b[0])
    split_x = (bboxes[0][0] + bboxes[0][2] + bboxes[1][0]) // 2

    left_foot = cv2.rotate(src_image[:, :split_x].copy(), cv2.ROTATE_180)
    right_foot = cv2.rotate(src_image[:, split_x:].copy(), cv2.ROTATE_180)
    right_foot = cv2.flip(right_foot, 1)

    debug_img = src_image.copy()
    cv2.line(debug_img, (split_x, 0), (split_x, h), (0, 255, 0), LINE_THICKNESS_DIVIDER)
    cv2.drawContours(debug_img, contours, -1, (255, 0, 0), LINE_THICKNESS_CONTOUR)
    for x, y, bw, bh in bboxes:
        cv2.rectangle(debug_img, (x, y), (x + bw, y + bh), (0, 0, 255), LINE_THICKNESS_RECT)

    return left_foot, right_foot, debug_img


# =============================================================================
# ПОЛУЧЕНИЕ МАСОК
# =============================================================================

def get_foot_mask(src_image):
    no_bg = remove(src_image)
    no_bg_visual = no_bg[:, :, :3].copy()

    foot_mask = no_bg[:, :, 3]
    _, foot_mask = cv2.threshold(foot_mask, FOOT_MASK_THRESHOLD, 255, cv2.THRESH_BINARY)

    kernel = np.ones((MORPHOLOGY_KERNEL_SIZE, MORPHOLOGY_KERNEL_SIZE), np.uint8)
    foot_mask = cv2.morphologyEx(foot_mask, cv2.MORPH_CLOSE, kernel)

    return foot_mask, no_bg_visual


def get_plantogram_mask(src_image, foot_mask):
    hsv = cv2.cvtColor(src_image, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, HSV_LOWER_GREEN, HSV_UPPER_GREEN)
    green_mask_raw = green_mask.copy()
    green_mask = cv2.bitwise_and(green_mask, foot_mask)

    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(green_mask)
    for cnt in contours:
        if cv2.contourArea(cnt) > MIN_PLANTOGRAM_AREA:
            cv2.drawContours(clean_mask, [cnt], -1, 255, -1)

    return green_mask_raw, green_mask, clean_mask


def get_largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return None
    return max(contours, key=cv2.contourArea)


def draw_contours(image, foot_mask, plant_mask):
    result = image.copy()

    foot_contours, _ = cv2.findContours(foot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, foot_contours, -1, (255, 0, 0), LINE_THICKNESS_CONTOUR)

    plant_contours, _ = cv2.findContours(plant_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, plant_contours, -1, (0, 0, 255), LINE_THICKNESS_CONTOUR)

    return result


# =============================================================================
# ПОИСК ТОЧЕК A И B
# =============================================================================

def find_A_at_widest_toe(plant_contour):
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
    A = find_A_at_widest_toe(plant_contour)
    B = find_B_at_leftmost_heel(plant_contour)
    return A, B


def get_foot_type(index):
    if index is None:
        return None
    for threshold, name in FOOT_TYPE_BOUNDARIES:
        if index <= threshold:
            return name
    return "Плоскостопие"


def draw_points_with_labels(debug_img, A, B, V, G, D, perp_dx, perp_dy):
    """
    Отрисовка всех точек, линий и буквенных обозначений.
    """
    # Линия AB (желтый)
    cv2.line(debug_img, A, B, (0, 255, 255), LINE_THICKNESS_AB)

    # Перпендикуляр (фиолетовый)
    h, w = debug_img.shape[:2]
    perp_len = max(PERP_LINE_EXTEND, max(h, w) * 2)
    start_point = (int(V[0] - perp_dx * perp_len), int(V[1] - perp_dy * perp_len))
    end_point = (int(V[0] + perp_dx * perp_len), int(V[1] + perp_dy * perp_len))
    cv2.line(debug_img, start_point, end_point, (200, 0, 255), LINE_THICKNESS_PERP)

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = FONT_THICKNESS
    font_scale = FONT_SCALE

    # Точки G, D, V с буквами
    for point, color, label, offset in [
        (G, (255, 255, 0), "G", (-130, -70)),
        (D, (0, 255, 255), "D", (60, -60)),
        (V, (0, 255, 0), "V", (60, -45)),
    ]:
        cv2.circle(debug_img, point, POINT_SIZE_LARGE, color, -1)
        cv2.circle(debug_img, point, POINT_SIZE_LARGE, (0, 0, 0), CIRCLE_OUTLINE)
        cv2.putText(debug_img, label,
                    (point[0] + offset[0], point[1] + offset[1]),
                    font, font_scale, (255, 255, 255), thickness + TEXT_OUTLINE)
        cv2.putText(debug_img, label,
                    (point[0] + offset[0], point[1] + offset[1]),
                    font, font_scale, color, thickness)

    # Точка A с буквой
    cv2.circle(debug_img, A, POINT_SIZE_MEDIUM, (255, 0, 0), -1)
    cv2.circle(debug_img, A, POINT_SIZE_MEDIUM, (0, 0, 0), CIRCLE_OUTLINE)
    cv2.putText(debug_img, "A", (A[0] - 100, A[1] - 65),
                font, font_scale, (255, 255, 255), thickness + TEXT_OUTLINE)
    cv2.putText(debug_img, "A", (A[0] - 100, A[1] - 65),
                font, font_scale, (255, 0, 0), thickness)

    # Точка B с буквой
    cv2.circle(debug_img, B, POINT_SIZE_MEDIUM, (0, 0, 255), -1)
    cv2.circle(debug_img, B, POINT_SIZE_MEDIUM, (0, 0, 0), CIRCLE_OUTLINE)
    cv2.putText(debug_img, "B", (B[0] - 100, B[1] + 95),
                font, font_scale, (255, 255, 255), thickness + TEXT_OUTLINE)
    cv2.putText(debug_img, "B", (B[0] - 100, B[1] + 95),
                font, font_scale, (0, 0, 255), thickness)

    # Крестик в точке V
    cv2.line(debug_img, (V[0] - CROSS_SIZE, V[1]), (V[0] + CROSS_SIZE, V[1]), (0, 0, 0), LINE_THICKNESS_CROSS)
    cv2.line(debug_img, (V[0], V[1] - CROSS_SIZE), (V[0], V[1] + CROSS_SIZE), (0, 0, 0), LINE_THICKNESS_CROSS)


def calculate_strieter_index_full(plant_contour, A, B, debug_img=None):
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
        print("ВНИМАНИЕ: Не удалось найти точки пересечения")
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
        draw_points_with_labels(debug_img, A, B, V, G, D, perp_dx, perp_dy)

    return index, V, G, D, foot_type


# =============================================================================
# ОСНОВНОЙ ПРОЦЕСС ПОЭТАПНОЙ ОБРАБОТКИ
# =============================================================================

def process_foot_step_by_step(foot, side, session_dir, start_step=1):
    step = start_step
    label = "ЛЕВАЯ СТОПА" if side == 0 else "ПРАВАЯ СТОПА"

    print(f"\n{'=' * 60}")
    print(f"ОБРАБОТКА: {label}")
    print(f"{'=' * 60}")

    # --- Шаг 1: Исходное изображение ---
    print(f"[Шаг {step}] Исходное изображение стопы")
    save_step(session_dir, f"{side}-step-{step}-original.png", foot)
    step += 1

    # --- Шаг 2: Удаление фона (rembg) ---
    print(f"[Шаг {step}] Удаление фона (rembg)")
    foot_mask, no_bg_visual = get_foot_mask(foot)
    save_step(session_dir, f"{side}-step-{step}-remove-background.png", no_bg_visual)
    step += 1

    # --- Шаг 3: Бинарная маска стопы ---
    print(f"[Шаг {step}] Бинарная маска стопы")
    foot_mask_color = mask_to_color(foot_mask, (255, 255, 255))
    save_step(session_dir, f"{side}-step-{step}-foot-mask.png", foot_mask_color)
    step += 1

    # --- Шаг 4: Контур стопы ---
    print(f"[Шаг {step}] Контур стопы")
    foot_contour_img = foot.copy()
    foot_contours, _ = cv2.findContours(foot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(foot_contour_img, foot_contours, -1, (255, 0, 0), LINE_THICKNESS_CONTOUR)
    save_step(session_dir, f"{side}-step-{step}-foot-contour.png", foot_contour_img)
    step += 1

    # --- Шаг 5: Маска зеленого ---
    print(f"[Шаг {step}] Маска зеленого цвета (HSV-диапазон)")
    green_mask_raw, green_mask_with_foot, clean_mask = get_plantogram_mask(foot, foot_mask)
    green_mask_raw_color = mask_to_color(green_mask_raw, (0, 255, 0))
    save_step(session_dir, f"{side}-step-{step}-green-mask-raw.png", green_mask_raw_color)
    step += 1

    # --- Шаг 6: Фильтрация по площади ---
    print(f"[Шаг {step}] Фильтрация по площади (> {MIN_PLANTOGRAM_AREA} px)")
    clean_mask_color = mask_to_color(clean_mask, (0, 255, 0))
    save_step(session_dir, f"{side}-step-{step}-plantogram-clean.png", clean_mask_color)
    step += 1

    # --- Шаг 7: Итоговое наложение контуров ---
    print(f"[Шаг {step}] Итоговое наложение контуров")
    vis = draw_contours(foot, foot_mask, clean_mask)
    save_step(session_dir, f"{side}-step-{step}-all-contours.png", vis)
    step += 1

    # --- Получение контура плантограммы для шагов 8 и 9 ---
    plant_contour = get_largest_contour(clean_mask)
    if plant_contour is None:
        print("ОШИБКА: контур плантограммы не найден!")
        return foot, None, None

    # --- Шаг 8: Точки A и B ---
    print(f"[Шаг {step}] Точки A и B, линия AB")
    A, B = find_AB_points_final(plant_contour)

    points_img = vis.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = FONT_THICKNESS
    font_scale = FONT_SCALE

    # Точка A с буквой
    cv2.circle(points_img, A, POINT_SIZE_MEDIUM, (255, 0, 0), -1)
    cv2.circle(points_img, A, POINT_SIZE_MEDIUM, (0, 0, 0), CIRCLE_OUTLINE)
    cv2.putText(points_img, "A", (A[0] - 100, A[1] - 65),
                font, font_scale, (255, 255, 255), thickness + TEXT_OUTLINE)
    cv2.putText(points_img, "A", (A[0] - 100, A[1] - 65),
                font, font_scale, (255, 0, 0), thickness)

    # Точка B с буквой
    cv2.circle(points_img, B, POINT_SIZE_MEDIUM, (0, 0, 255), -1)
    cv2.circle(points_img, B, POINT_SIZE_MEDIUM, (0, 0, 0), CIRCLE_OUTLINE)
    cv2.putText(points_img, "B", (B[0] - 100, B[1] + 95),
                font, font_scale, (255, 255, 255), thickness + TEXT_OUTLINE)
    cv2.putText(points_img, "B", (B[0] - 100, B[1] + 95),
                font, font_scale, (0, 0, 255), thickness)

    cv2.line(points_img, A, B, (0, 255, 255), LINE_THICKNESS_AB)
    save_step(session_dir, f"{side}-step-{step}-points-AB.png", points_img)
    step += 1

    # --- Шаг 9: Индекс Штриттера ---
    print(f"[Шаг {step}] Расчет индекса Штриттера")
    index, V, G, D, foot_type = calculate_strieter_index_full(plant_contour, A, B, vis)

    if index is not None:
        save_step(session_dir, f"{side}-step-{step}-strieter-index.png", vis)
    else:
        save_step(session_dir, f"{side}-step-{step}-strieter-failed.png", vis)

    # Вывод результатов в консоль
    print(f"\n{'=' * 40}")
    print(f"РЕЗУЛЬТАТЫ: {label}")
    print(f"{'=' * 40}")
    print(f"  Точка A: {A}")
    print(f"  Точка B: {B}")
    if index is not None:
        print(f"  Точка V (центр AB): {V}")
        print(f"  Точка G (внутренний край): {G}")
        print(f"  Точка D (наружный край): {D}")
        print(f"  Индекс Штриттера: {index:.1f}")
        print(f"  Тип стопы: {foot_type}")
    else:
        print("  Расчет не выполнен")

    return vis, index, foot_type


# =============================================================================
# ТОЧКА ВХОДА
# =============================================================================

if __name__ == "__main__":
    path = "./foots/4/IMG_0302.jpg"
    src = cv2.imread(path)

    if src is None:
        raise Exception(f"Изображение не найдено: {path}")

    session_dir = create_screenshots_dir()
    print("=" * 60)
    print("ПОЭТАПНЫЙ АНАЛИЗ ПЛАНТОГРАММЫ СТОПЫ")
    print("=" * 60)
    print(f"\nПапка для сохранения: {session_dir}\n")

    # Шаг 0: Разделение стоп
    left, right, debug = split_feet_smart(src)
    print("[Шаг 0] Разделение на левую и правую стопы")
    save_step(session_dir, "split-feet.png", debug)

    print("\nВыберите стопу для анализа:")
    print("1 - Левая стопа")
    print("2 - Правая стопа")
    print("3 - Обе стопы последовательно")
    choice = input("Ваш выбор (1-3): ").strip()

    if choice == "1":
        process_foot_step_by_step(left, 0, session_dir)
    elif choice == "2":
        process_foot_step_by_step(right, 1, session_dir)
    elif choice == "3":
        process_foot_step_by_step(left, 0, session_dir)
        print("\n" + "=" * 60)
        print("Переход к обработке правой стопы...")
        print("=" * 60)
        process_foot_step_by_step(right, 1, session_dir)
    else:
        print("Некорректный выбор. Обрабатываю левую стопу по умолчанию.")
        process_foot_step_by_step(left, 0, session_dir)

    print("\n" + "=" * 60)
    print("АНАЛИЗ ЗАВЕРШЕН")
    print(f"Все изображения сохранены в: {session_dir}")
    print("=" * 60)