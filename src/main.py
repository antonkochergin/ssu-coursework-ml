import cv2
import numpy as np
from rembg import remove

# =============================================================================
# КОНСТАНТЫ
# =============================================================================

# Параметры для маски стопы
FOOT_MASK_THRESHOLD = 100
MORPHOLOGY_KERNEL_SIZE = 5

# Параметры для маски плантограммы (зеленый цвет в HSV)
HSV_LOWER_GREEN = np.array([74, 149, 141])
HSV_UPPER_GREEN = np.array([86, 255, 194])
MIN_PLANTOGRAM_AREA = 50000

# Параметры для поиска точек A и B
INNER_EDGE_OFFSET = 20
TOE_AREA_RATIO = 0.3  # верхние 30% стопы - носочная часть
HEEL_AREA_RATIO = 0.65  # нижние 35% стопы - пятка
TOE_SEARCH_STEP = 5  # шаг поиска по Y в носочной части
TOE_Y_TOLERANCE = 3  # допуск по Y для группировки точек

# Параметры для расчета индекса Штриттера
PERP_DIST_TOLERANCE = 10  # допуск для поиска точек на перпендикуляре
PERP_DIST_TOLERANCE_FALLBACK = 20

# Параметры для визуализации
PERP_LINE_EXTEND = 2000  # длина перпендикуляра
FONT_SCALE = 1.2
FONT_THICKNESS = 3
POINT_SIZE_LARGE = 18
POINT_SIZE_MEDIUM = 14
CROSS_SIZE = 20

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

def split_feet_smart(src_image, debug=False):
    """
    Разделяет изображение на две стопы: левую и правую.

    Args:
        src_image: исходное изображение
        debug: флаг для отладки

    Returns:
        left_foot, right_foot, debug_img
    """
    h, w = src_image.shape[:2]

    # Удаляем фон и получаем маску
    no_bg = remove(src_image)
    mask = no_bg[:, :, 3]
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Находим контуры стоп
    contours, _ = cv2.findContours(
        mask_bin,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    # Определяем bounding boxes и точку разделения
    bboxes = sorted([cv2.boundingRect(c) for c in contours], key=lambda b: b[0])
    split_x = (bboxes[0][0] + bboxes[0][2] + bboxes[1][0]) // 2

    # Разделяем и поворачиваем стопы
    left_foot = cv2.rotate(src_image[:, :split_x].copy(), cv2.ROTATE_180)
    right_foot = cv2.rotate(src_image[:, split_x:].copy(), cv2.ROTATE_180)
    right_foot = cv2.flip(right_foot, 1)

    # Отладочное изображение
    debug_img = src_image.copy()
    cv2.line(debug_img, (split_x, 0), (split_x, h), (0, 255, 0), 2)
    cv2.drawContours(debug_img, contours, -1, (255, 0, 0), 2)
    for x, y, bw, bh in bboxes:
        cv2.rectangle(debug_img, (x, y), (x + bw, y + bh), (0, 0, 255), 2)

    return left_foot, right_foot, debug_img


def get_foot_mask(src_image):
    """
    Получает бинарную маску стопы.

    Args:
        src_image: изображение стопы

    Returns:
        бинарная маска стопы
    """
    no_bg = remove(src_image)
    foot_mask = no_bg[:, :, 3]
    _, foot_mask = cv2.threshold(foot_mask, FOOT_MASK_THRESHOLD, 255, cv2.THRESH_BINARY)

    kernel = np.ones((MORPHOLOGY_KERNEL_SIZE, MORPHOLOGY_KERNEL_SIZE), np.uint8)
    foot_mask = cv2.morphologyEx(foot_mask, cv2.MORPH_CLOSE, kernel)

    return foot_mask


def get_plantogram_mask(src_image, foot_mask):
    """
    Получает маску плантограммы (зеленой области внутри стопы).

    Args:
        src_image: изображение стопы
        foot_mask: маска стопы

    Returns:
        бинарная маска плантограммы
    """
    hsv = cv2.cvtColor(src_image, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, HSV_LOWER_GREEN, HSV_UPPER_GREEN)
    green_mask = cv2.bitwise_and(green_mask, foot_mask)

    # Оставляем только крупные области
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(green_mask)
    for cnt in contours:
        if cv2.contourArea(cnt) > MIN_PLANTOGRAM_AREA:
            cv2.drawContours(clean_mask, [cnt], -1, 255, -1)

    return clean_mask


def get_largest_contour(mask):
    """
    Находит самый большой контур в маске.

    Args:
        mask: бинарная маска

    Returns:
        самый большой контур
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(contours, key=cv2.contourArea)


def draw_contours(image, foot_mask, plant_mask):
    """
    Отрисовывает контуры стопы и плантограммы на изображении.

    Args:
        image: исходное изображение
        foot_mask: маска стопы
        plant_mask: маска плантограммы

    Returns:
        изображение с нарисованными контурами
    """
    result = image.copy()

    # Контур стопы (синий)
    foot_contours, _ = cv2.findContours(foot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, foot_contours, -1, (255, 0, 0), 2)

    # Контур плантограммы (красный)
    plant_contours, _ = cv2.findContours(plant_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, plant_contours, -1, (0, 0, 255), 2)

    return result


# =============================================================================
# ПОИСК ТОЧЕК A И B
# =============================================================================

def find_A_at_widest_toe(plant_contour):
    """
    Находит точку A как самую широкую точку в носочной части стопы.

    Args:
        plant_contour: контур плантограммы

    Returns:
        координаты точки A (внутренний край в самой широкой части носка)
    """
    pts = plant_contour[:, 0, :]
    y_min, y_max = np.min(pts[:, 1]), np.max(pts[:, 1])
    height = y_max - y_min

    # Носочная часть (верхние 30%)
    toe_threshold = y_min + height * TOE_AREA_RATIO
    toe_pts = pts[pts[:, 1] < toe_threshold]

    if len(toe_pts) == 0:
        toe_threshold = y_min + height * 0.25
        toe_pts = pts[pts[:, 1] < toe_threshold]

    # Поиск самой широкой точки
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

    # Fallback: если не нашли, берем крайнюю левую точку
    if best_left is None:
        min_x = np.min(toe_pts[:, 0])
        best_left_pts = toe_pts[toe_pts[:, 0] == min_x]
        best_left = tuple(best_left_pts[0]) if len(best_left_pts) > 0 else (0, 0)

    return best_left


def find_B_at_leftmost_heel(plant_contour):
    """
    Находит точку B как крайнюю левую точку в области пятки.

    Args:
        plant_contour: контур плантограммы

    Returns:
        координаты точки B (крайняя левая точка пятки)
    """
    pts = plant_contour[:, 0, :]
    y_min, y_max = np.min(pts[:, 1]), np.max(pts[:, 1])
    height = y_max - y_min

    # Область пятки (нижние 35%)
    heel_threshold = y_min + height * HEEL_AREA_RATIO
    heel_pts = pts[pts[:, 1] > heel_threshold]

    if len(heel_pts) == 0:
        heel_threshold = y_min + height * 0.7
        heel_pts = pts[pts[:, 1] > heel_threshold]

    # Крайняя левая точка
    B_idx = np.argmin(heel_pts[:, 0])
    return tuple(heel_pts[B_idx])


def find_AB_points_final(plant_contour):
    """
    Находит точки A и B для расчета индекса Штриттера.

    Args:
        plant_contour: контур плантограммы

    Returns:
        A, B - координаты точек
    """
    A = find_A_at_widest_toe(plant_contour)
    B = find_B_at_leftmost_heel(plant_contour)
    return A, B


# =============================================================================
# РАСЧЕТ ИНДЕКСА ШТРИТТЕРА
# =============================================================================

def draw_points_on_image(debug_img, A, B, V, G, D, perp_dx, perp_dy):
    """
    Отрисовка всех точек, линий и подписей на изображении.

    Args:
        debug_img: изображение для отрисовки
        A, B, V, G, D: координаты точек
        perp_dx, perp_dy: направляющий вектор перпендикуляра
    """
    # Отрезок AB (желтый)
    cv2.line(debug_img, A, B, (0, 255, 255), 3)

    # Перпендикуляр (фиолетовый)
    h, w = debug_img.shape[:2]
    perp_len = max(PERP_LINE_EXTEND, max(h, w) * 2)
    start_point = (int(V[0] - perp_dx * perp_len), int(V[1] - perp_dy * perp_len))
    end_point = (int(V[0] + perp_dx * perp_len), int(V[1] + perp_dy * perp_len))
    cv2.line(debug_img, start_point, end_point, (200, 0, 255), 4)

    # Точки
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = FONT_THICKNESS
    font_scale = FONT_SCALE

    # Точки G и D (пересечения с контуром)
    for point, color, label, offset in [
        (G, (255, 255, 0), "G", (-60, -35)),
        (D, (0, 255, 255), "D", (25, -25)),
        (V, (0, 255, 0), "V", (25, -20)),
    ]:
        cv2.circle(debug_img, point, POINT_SIZE_LARGE, color, -1)
        cv2.circle(debug_img, point, POINT_SIZE_LARGE, (0, 0, 0), 3)
        cv2.putText(debug_img, label,
                    (point[0] + offset[0], point[1] + offset[1]),
                    font, font_scale, (0, 0, 0), thickness + 2)
        cv2.putText(debug_img, label,
                    (point[0] + offset[0], point[1] + offset[1]),
                    font, font_scale, color, thickness)

    # Точки A и B
    cv2.circle(debug_img, A, POINT_SIZE_MEDIUM, (255, 0, 0), -1)
    cv2.circle(debug_img, A, POINT_SIZE_MEDIUM, (0, 0, 0), 2)
    cv2.putText(debug_img, "A", (A[0] - 50, A[1] - 30),
                font, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(debug_img, "A", (A[0] - 50, A[1] - 30),
                font, font_scale, (255, 0, 0), thickness)

    cv2.circle(debug_img, B, POINT_SIZE_MEDIUM, (0, 0, 255), -1)
    cv2.circle(debug_img, B, POINT_SIZE_MEDIUM, (0, 0, 0), 2)
    cv2.putText(debug_img, "B", (B[0] - 50, B[1] + 45),
                font, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(debug_img, "B", (B[0] - 50, B[1] + 45),
                font, font_scale, (0, 0, 255), thickness)

    # Крестик в точке V
    cv2.line(debug_img, (V[0] - CROSS_SIZE, V[1]), (V[0] + CROSS_SIZE, V[1]), (0, 0, 0), 2)
    cv2.line(debug_img, (V[0], V[1] - CROSS_SIZE), (V[0], V[1] + CROSS_SIZE), (0, 0, 0), 2)


def get_foot_type(index):
    """
    Определяет тип стопы по индексу Штриттера.

    Args:
        index: значение индекса Штриттера

    Returns:
        строковое название типа стопы
    """
    if index is None:
        return None
    for threshold, name in FOOT_TYPE_BOUNDARIES:
        if index <= threshold:
            return name
    return "Плоскостопие"


def calculate_strieter_index_full(plant_contour, A, B, debug_img=None):
    """
    Полный расчет индекса Штриттера.

    Этапы:
    1. Находит центр отрезка AB (точка V)
    2. Строит перпендикуляр к AB через точку V
    3. Находит точки пересечения с контуром (G и D)
    4. Рассчитывает индекс по формуле: I = (GD * 100) / VD

    Args:
        plant_contour: контур плантограммы
        A, B: координаты точек A и B
        debug_img: изображение для визуализации (опционально)

    Returns:
        index, V, G, D, foot_type
    """
    # 1. Центр отрезка AB (точка V)
    V = ((A[0] + B[0]) // 2, (A[1] + B[1]) // 2)

    # 2. Вектор перпендикуляра к AB
    dx, dy = B[0] - A[0], B[1] - A[1]
    length = np.sqrt(dx * dx + dy * dy)
    if length == 0:
        return None, None, None, None, None

    perp_dx, perp_dy = -dy / length, dx / length

    # 3. Поиск точек пересечения перпендикуляра с контуром
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
        print("Не удалось найти точки пересечения")
        return None, None, None, None, None

    projections.sort(key=lambda x: x[0])
    min_proj, point_min = projections[0]
    max_proj, point_max = projections[-1]

    # Определяем внутренний и наружный край
    if point_min[0] < point_max[0]:
        G, D = point_min, point_max
        GD, VD = max_proj - min_proj, abs(max_proj)
    else:
        G, D = point_max, point_min
        GD, VD = max_proj - min_proj, abs(min_proj)

    # 4. Расчет индекса
    index = (GD * 100) / VD if VD > 0 else None
    foot_type = get_foot_type(index)

    # Визуализация
    if debug_img is not None and index is not None:
        draw_points_on_image(debug_img, A, B, V, G, D, perp_dx, perp_dy)

    return index, V, G, D, foot_type


# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

def show(name, img, max_w=1200, max_h=800):
    """
    Отображает изображение в окне с автоматическим масштабированием.

    Args:
        name: название окна
        img: изображение
        max_w, max_h: максимальные размеры окна
    """
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    cv2.imshow(name, img)


def process_foot(foot, label):
    """
    Обрабатывает одну стопу: строит маски, находит точки и рассчитывает индекс.

    Args:
        foot: изображение стопы
        label: название для вывода

    Returns:
        vis, index, foot_type
    """
    foot_mask = get_foot_mask(foot)
    plant_mask = get_plantogram_mask(foot, foot_mask)
    vis = draw_contours(foot, foot_mask, plant_mask)

    plant_contour = get_largest_contour(plant_mask)
    A, B = find_AB_points_final(plant_contour)
    index, V, G, D, foot_type = calculate_strieter_index_full(plant_contour, A, B, vis)

    print(f"\n=== {label} ===")
    print(f"  Точка A: {A}")
    print(f"  Точка B: {B}")
    print(f"  Центр V: {V}")
    print(f"  Точка G (внутренний край): {G}")
    print(f"  Точка D (наружный край): {D}")
    print(f"  GD = {np.sqrt((D[0] - G[0]) ** 2 + (D[1] - G[1]) ** 2):.1f} px")
    print(f"  VD = {np.sqrt((D[0] - V[0]) ** 2 + (D[1] - V[1]) ** 2):.1f} px")
    print(f"  Индекс Штриттера: {index:.1f}")
    print(f"  Тип стопы: {foot_type}")

    return vis, index, foot_type


# =============================================================================
# ОСНОВНАЯ ПРОГРАММА
# =============================================================================

if __name__ == "__main__":
    path = "foots/4/IMG_0302.jpg"
    src = cv2.imread(path)

    if src is None:
        raise Exception("Image not found")

    left, right, debug = split_feet_smart(src, debug=True)

    # Обработка левой стопы
    left_vis, idx_left, type_left = process_foot(left, "ЛЕВАЯ СТОПА")

    # Обработка правой стопы
    right_vis, idx_right, type_right = process_foot(right, "ПРАВАЯ СТОПА")

    # Отображение результатов
    show("LEFT RESULT", left_vis)
    show("RIGHT RESULT", right_vis)
    show("DEBUG SPLIT", debug)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
