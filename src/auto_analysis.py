import hashlib

import cv2
import numpy as np
from rembg import remove
import os
import json
import glob

# =============================================================================
# КОНСТАНТЫ
# =============================================================================

# Параметры для маски стопы
FOOT_MASK_THRESHOLD = 100
MORPHOLOGY_KERNEL_SIZE = 5

# Параметры для маски плантограммы (зеленый цвет в HSV)
# ОПТИМИЗИРОВАНЫ ГЕНЕТИЧЕСКИМ АЛГОРИТМОМ (фитнес 88.22%)
HSV_LOWER_GREEN = np.array([72, 179, 101])
HSV_UPPER_GREEN = np.array([99, 244, 183])
MIN_PLANTOGRAM_AREA = 54104

# Параметры для поиска точек A и B
INNER_EDGE_OFFSET = 20
TOE_AREA_RATIO = 0.4361466882431499
HEEL_AREA_RATIO = 0.6509073113319302
TOE_SEARCH_STEP = 5
TOE_Y_TOLERANCE = 3

# Параметры для расчета индекса Штриттера
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

# Пути к папкам
INPUT_DIR = "./foots-for-analysis/"
ORIGINAL_DIR = "./original/"
SPLIT_DIR = "./split_feet/"
RESULTS_VIS_DIR = "./results_vis/"
PROGRESS_FILE = "progress.json"
SPLIT_PROGRESS_FILE = "split_progress.json"

# Максимальный размер для отображения
MAX_DISPLAY_SIZE = 800


# =============================================================================
# ОБРАБОТКА ИЗОБРАЖЕНИЯ
# =============================================================================
# В auto_analysis.py добавить:
CACHE_DIR = "./cache/"
os.makedirs(CACHE_DIR, exist_ok=True)


def get_foot_mask_cached(src_image, image_hash=None):
    """Кэшированная версия get_foot_mask"""
    if image_hash is None:
        # Создаем хеш изображения
        image_hash = hashlib.md5(src_image.tobytes()).hexdigest()

    cache_file = os.path.join(CACHE_DIR, f"{image_hash}_foot_mask.npy")

    if os.path.exists(cache_file):
        return np.load(cache_file)

    result = get_foot_mask(src_image)
    np.save(cache_file, result)
    return result

def split_feet_smart(src_image, debug=False):
    """Разделяет изображение на две стопы: левую и правую."""
    h, w = src_image.shape[:2]

    no_bg = remove(src_image)
    mask = no_bg[:, :, 3]
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if len(contours) < 2:
        print(f"⚠️ Найдено только {len(contours)} контур(ов). Ожидается 2 стопы.")
        if len(contours) == 1:
            print("   Пытаемся разделить изображение пополам...")
            split_x = w // 2
            left_foot = cv2.rotate(src_image[:, :split_x].copy(), cv2.ROTATE_180)
            right_foot = cv2.rotate(src_image[:, split_x:].copy(), cv2.ROTATE_180)
            right_foot = cv2.flip(right_foot, 1)
            debug_img = src_image.copy()
            cv2.line(debug_img, (split_x, 0), (split_x, h), (0, 255, 0), 2)
            return left_foot, right_foot, debug_img
        else:
            print("   ⚠️ Пропускаем изображение (нет стоп)")
            return None, None, src_image.copy()

    contours = contours[:2]
    bboxes = sorted([cv2.boundingRect(c) for c in contours], key=lambda b: b[0])
    split_x = (bboxes[0][0] + bboxes[0][2] + bboxes[1][0]) // 2

    left_foot = cv2.rotate(src_image[:, :split_x].copy(), cv2.ROTATE_180)
    right_foot = cv2.rotate(src_image[:, split_x:].copy(), cv2.ROTATE_180)
    right_foot = cv2.flip(right_foot, 1)

    debug_img = src_image.copy()
    cv2.line(debug_img, (split_x, 0), (split_x, h), (0, 255, 0), 2)
    cv2.drawContours(debug_img, contours, -1, (255, 0, 0), 2)
    for x, y, bw, bh in bboxes:
        cv2.rectangle(debug_img, (x, y), (x + bw, y + bh), (0, 0, 255), 2)

    return left_foot, right_foot, debug_img


def get_foot_mask(src_image):
    """Получает бинарную маску стопы."""
    no_bg = remove(src_image)
    foot_mask = no_bg[:, :, 3]
    _, foot_mask = cv2.threshold(foot_mask, FOOT_MASK_THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel = np.ones((MORPHOLOGY_KERNEL_SIZE, MORPHOLOGY_KERNEL_SIZE), np.uint8)
    foot_mask = cv2.morphologyEx(foot_mask, cv2.MORPH_CLOSE, kernel)
    return foot_mask


def get_plantogram_mask(src_image, foot_mask, params=None):
    """Получает маску плантограммы."""
    if params is None:
        hsv_lower = HSV_LOWER_GREEN
        hsv_upper = HSV_UPPER_GREEN
        min_area = MIN_PLANTOGRAM_AREA
    else:
        hsv_lower = np.array([params['H_LOW'], params['S_LOW'], params['V_LOW']])
        hsv_upper = np.array([params['H_HIGH'], params['S_HIGH'], params['V_HIGH']])
        min_area = params['MIN_AREA']

    hsv = cv2.cvtColor(src_image, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    green_mask = cv2.bitwise_and(green_mask, foot_mask)

    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(green_mask)
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
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

def find_A_at_widest_toe(plant_contour, params=None):
    """Находит точку A как самую широкую точку в носочной части."""
    if params is None:
        toe_ratio = TOE_AREA_RATIO
        search_step = TOE_SEARCH_STEP
        y_tolerance = TOE_Y_TOLERANCE
    else:
        toe_ratio = params.get('TOE_RATIO', TOE_AREA_RATIO)
        search_step = params.get('TOE_SEARCH_STEP', TOE_SEARCH_STEP)
        y_tolerance = params.get('TOE_Y_TOLERANCE', TOE_Y_TOLERANCE)

    pts = plant_contour[:, 0, :]
    y_min, y_max = np.min(pts[:, 1]), np.max(pts[:, 1])
    height = y_max - y_min

    toe_threshold = y_min + height * toe_ratio
    toe_pts = pts[pts[:, 1] < toe_threshold]

    if len(toe_pts) == 0:
        toe_threshold = y_min + height * 0.25
        toe_pts = pts[pts[:, 1] < toe_threshold]

    max_width = 0
    best_left = None

    for y in np.arange(int(y_min), int(toe_threshold), search_step):
        level_pts = pts[np.abs(pts[:, 1] - y) < y_tolerance]
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


def find_B_at_leftmost_heel(plant_contour, params=None):
    """Находит точку B как крайнюю левую точку в области пятки."""
    if params is None:
        heel_ratio = HEEL_AREA_RATIO
    else:
        heel_ratio = params.get('HEEL_RATIO', HEEL_AREA_RATIO)

    pts = plant_contour[:, 0, :]
    y_min, y_max = np.min(pts[:, 1]), np.max(pts[:, 1])
    height = y_max - y_min

    heel_threshold = y_min + height * heel_ratio
    heel_pts = pts[pts[:, 1] > heel_threshold]

    if len(heel_pts) == 0:
        heel_threshold = y_min + height * 0.7
        heel_pts = pts[pts[:, 1] > heel_threshold]

    B_idx = np.argmin(heel_pts[:, 0])
    return tuple(heel_pts[B_idx])


def find_AB_points_final(plant_contour, params=None):
    """Находит точки A и B."""
    A = find_A_at_widest_toe(plant_contour, params)
    B = find_B_at_leftmost_heel(plant_contour, params)
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


def calculate_strieter_index_full(plant_contour, A, B, debug_img=None, params=None):
    """Полный расчет индекса Штриттера."""
    if params is None:
        perp_tolerance = PERP_DIST_TOLERANCE
        perp_tolerance_fallback = PERP_DIST_TOLERANCE_FALLBACK
    else:
        perp_tolerance = params.get('PERP_TOLERANCE', PERP_DIST_TOLERANCE)
        perp_tolerance_fallback = params.get('PERP_TOLERANCE_FALLBACK', PERP_DIST_TOLERANCE_FALLBACK)

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
        if dist_to_perp < perp_tolerance:
            projections.append((proj, p))

    if len(projections) < 2:
        for p in pts:
            vx, vy = p[0] - V[0], p[1] - V[1]
            dist_to_perp = abs(vx * (-perp_dy) + vy * perp_dx)
            if dist_to_perp < perp_tolerance_fallback:
                proj = vx * perp_dx + vy * perp_dy
                projections.append((proj, p))

    if len(projections) < 2:
        print("Не удалось найти точки пересечения")
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
# ФУНКЦИИ ДЛЯ РАЗДЕЛЕНИЯ С ПРОВЕРКОЙ ПРОГРЕССА
# =============================================================================

def create_directories():
    """Создает все необходимые папки."""
    for dir_path in [ORIGINAL_DIR, SPLIT_DIR, RESULTS_VIS_DIR]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"📁 Создана папка: {dir_path}")


def get_image_files():
    """Получает список всех изображений из папки foots-for-analysis."""
    extensions = ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
    image_files = sorted(list(set(image_files)), key=lambda x: os.path.basename(x))
    return image_files


def load_split_progress():
    """Загружает прогресс разделения."""
    if os.path.exists(SPLIT_PROGRESS_FILE):
        with open(SPLIT_PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {'processed': [], 'skipped': [], 'total_feet': []}


def save_split_progress(progress):
    """Сохраняет прогресс разделения."""
    with open(SPLIT_PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def split_and_save_all():
    """Разделяет все изображения на стопы и сохраняет их."""
    print("\n" + "=" * 60)
    print("📸 РАЗДЕЛЕНИЕ ИЗОБРАЖЕНИЙ НА СТОПЫ")
    print("=" * 60)
    print(f"📂 Исходная папка: {INPUT_DIR}")

    create_directories()

    progress = load_split_progress()
    processed = set(progress.get('processed', []))
    skipped = set(progress.get('skipped', []))
    all_feet = set(progress.get('total_feet', []))

    all_images = get_image_files()

    if not all_images:
        print(f"\n❌ Изображения не найдены в папке {INPUT_DIR}!")
        return []

    to_process = []
    for img_path in all_images:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        if base_name in processed:
            continue
        if base_name in skipped:
            print(f"⏭️ {base_name} пропущен ранее, пропускаем")
            continue
        to_process.append(img_path)

    if not to_process:
        print("\n✅ Все изображения уже обработаны!")
        print(f"Всего обработано: {len(processed)}")
        print(f"Всего пропущено: {len(skipped)}")
        print(f"Всего стоп: {len(all_feet)}")
        return list(all_feet)

    print(f"\n📊 Статистика:")
    print(f"  ✅ Уже обработано: {len(processed)}")
    print(f"  ⏭️ Пропущено: {len(skipped)}")
    print(f"  🆕 Новых для обработки: {len(to_process)}")

    result = list(all_feet)

    for idx, img_path in enumerate(to_process, 1):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\n{'─' * 50}")
        print(f"Обработка {idx}/{len(to_process)}: {os.path.basename(img_path)}")

        src = cv2.imread(img_path)
        if src is None:
            print(f"  ⚠️ Не удалось загрузить: {img_path}")
            skipped.add(base_name)
            save_split_progress({'processed': list(processed), 'skipped': list(skipped), 'total_feet': list(result)})
            continue

        next_idx = len(processed) + 1

        original_name = f"{next_idx}.png"
        original_path = os.path.join(ORIGINAL_DIR, original_name)
        cv2.imwrite(original_path, src)
        print(f"  ✅ Оригинал сохранен: {original_name}")

        left_foot, right_foot, debug_img = split_feet_smart(src, debug=True)

        if left_foot is None or right_foot is None:
            print(f"  ⚠️ Пропускаем изображение {base_name} (не удалось разделить на стопы)")
            skipped.add(base_name)
            save_split_progress({'processed': list(processed), 'skipped': list(skipped), 'total_feet': list(result)})
            continue

        feet_names = []
        for foot_type, foot_img in [("_1", left_foot), ("_2", right_foot)]:
            foot_name = f"{next_idx}{foot_type}"
            feet_names.append(foot_name)
            split_path = os.path.join(SPLIT_DIR, f"{foot_name}.png")
            cv2.imwrite(split_path, foot_img)
            print(f"  ✅ {foot_name}.png сохранен в split_feet/ (без контуров)")

        result.extend(feet_names)
        debug_path = os.path.join(ORIGINAL_DIR, f"{next_idx}_debug.png")
        cv2.imwrite(debug_path, debug_img)

        processed.add(base_name)
        save_split_progress({'processed': list(processed), 'skipped': list(skipped), 'total_feet': list(result)})

        foot1_path = os.path.join(SPLIT_DIR, f"{next_idx}_1.png")
        foot2_path = os.path.join(SPLIT_DIR, f"{next_idx}_2.png")
        if os.path.exists(foot1_path) and os.path.exists(foot2_path):
            print(f"  ✅ Проверка: обе стопы сохранены ({next_idx}_1.png, {next_idx}_2.png)")

    print("\n" + "=" * 60)
    print("📊 ИТОГОВАЯ СТАТИСТИКА:")
    print(f"  ✅ Обработано изображений: {len(processed)}")
    print(f"  ⚠️ Пропущено изображений: {len(skipped)}")
    print(f"  ✅ Всего стоп: {len(result)}")
    print("=" * 60)

    return result


# =============================================================================
# ИНТЕРАКТИВНЫЙ РЕЖИМ ДЛЯ ПОДБОРА ПАРАМЕТРОВ
# =============================================================================

def load_progress():
    """Загружает прогресс из файла."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {'completed': [], 'skipped': [], 'results': {}}


def save_progress(progress):
    """Сохраняет прогресс в файл."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def get_default_params():
    """Возвращает параметры по умолчанию."""
    return {
        'H_LOW': 74,
        'H_HIGH': 86,
        'S_LOW': 149,
        'S_HIGH': 255,
        'V_LOW': 141,
        'V_HIGH': 194,
        'MIN_AREA': 50000,
        'TOE_RATIO': 0.3,
        'HEEL_RATIO': 0.65,
        'PERP_TOLERANCE': 10,
    }


def interactive_tuning():
    """Интерактивный режим с ползунками для подбора параметров."""
    print("\n" + "=" * 60)
    print("🎯 ИНТЕРАКТИВНЫЙ РЕЖИМ ПОДБОРА ПАРАМЕТРОВ")
    print("=" * 60)

    create_directories()

    all_feet = []
    for file in sorted(os.listdir(SPLIT_DIR)):
        if file.endswith('.png') and '_' in file:
            all_feet.append(file.replace('.png', ''))

    if not all_feet:
        print("❌ Нет разделенных стоп в папке split_feet/")
        print("Сначала запустите разделение изображений.")
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

    print(f"\nОсталось обработать: {len(to_process)} стоп")
    print(f"Всего стоп: {len(all_feet)}")
    print(f"Готово: {len(completed)}")
    print(f"Пропущено: {len(skipped)}")

    # СОКРАЩЕННЫЕ ПОЛЗУНКИ - только основные параметры
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controls", 400, 350)
    cv2.moveWindow("Controls", 0, 0)

    # Только самые важные параметры
    cv2.createTrackbar("H_LOW", "Controls", 74, 179, lambda x: None)
    cv2.createTrackbar("H_HIGH", "Controls", 86, 179, lambda x: None)
    cv2.createTrackbar("S_LOW", "Controls", 149, 255, lambda x: None)
    cv2.createTrackbar("S_HIGH", "Controls", 255, 255, lambda x: None)  # ← ДОБАВЛЯЕМ
    cv2.createTrackbar("V_LOW", "Controls", 141, 255, lambda x: None)
    cv2.createTrackbar("V_HIGH", "Controls", 194, 255, lambda x: None)  # ← ДОБАВЛЯЕМ
    cv2.createTrackbar("MIN_AREA", "Controls", 50000, 150000, lambda x: None)

    for foot_name in to_process:
        print(f"\n{'=' * 50}")
        print(f"Обработка: {foot_name}")
        print(f"Прогресс: {len(completed) + 1}/{len(all_feet)}")
        print(f"{'=' * 50}")
        print("Управление:")
        print("  S - Сохранить и перейти к следующей")
        print("  Пробел - Пропустить")
        print("  R - Сбросить параметры")
        print("  ESC - Выйти")

        img_path = os.path.join(SPLIT_DIR, f"{foot_name}.png")
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Не удалось загрузить: {img_path}")
            continue

        # МАСШТАБИРУЕМ изображение для отображения
        h, w = img.shape[:2]
        scale = min(MAX_DISPLAY_SIZE / w, MAX_DISPLAY_SIZE / h, 1.0)
        display_size = (int(w * scale), int(h * scale))

        foot_mask = get_foot_mask(img)

        while True:
            params = {
                'H_LOW': cv2.getTrackbarPos("H_LOW", "Controls"),
                'H_HIGH': cv2.getTrackbarPos("H_HIGH", "Controls"),
                'S_LOW': cv2.getTrackbarPos("S_LOW", "Controls"),
                'S_HIGH': cv2.getTrackbarPos("S_HIGH", "Controls"),      # ← ДОБАВЛЯЕМ
                'V_LOW': cv2.getTrackbarPos("V_LOW", "Controls"),
                'V_HIGH': cv2.getTrackbarPos("V_HIGH", "Controls"),      # ← ДОБАВЛЯЕМ
                'MIN_AREA': cv2.getTrackbarPos("MIN_AREA", "Controls"),
                'TOE_RATIO': TOE_AREA_RATIO,
                'HEEL_RATIO': HEEL_AREA_RATIO,
                'PERP_TOLERANCE': PERP_DIST_TOLERANCE,
                'PERP_TOLERANCE_FALLBACK': PERP_DIST_TOLERANCE_FALLBACK,
            }

            plant_mask = get_plantogram_mask(img, foot_mask, params)
            plant_contour = get_largest_contour(plant_mask)

            # Если нет контура - показываем сообщение
            if plant_contour is None:
                display_img = draw_contours(img, foot_mask, plant_mask)
                cv2.putText(display_img, "НЕТ ПЛАНТОГРАММЫ!", (50, display_size[1]//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                display_img = cv2.resize(display_img, display_size)
                cv2.imshow("Interactive Tuning", display_img)

                key = cv2.waitKey(30) & 0xFF
                if key == ord(' '):
                    skipped.add(foot_name)
                    progress['results'] = results
                    progress['completed'] = list(completed)
                    progress['skipped'] = list(skipped)
                    save_progress(progress)
                    print(f"  ⏭️ Пропущено: {foot_name} (нет плантограммы)")
                    break
                elif key == 27:
                    print("\n⚠️ Выход по ESC")
                    cv2.destroyAllWindows()
                    return
                continue

            A, B = find_AB_points_final(plant_contour, params)
            display_img = draw_contours(img, foot_mask, plant_mask)
            index, V, G, D, foot_type = calculate_strieter_index_full(
                plant_contour, A, B, display_img, params
            )

            # МАСШТАБИРУЕМ для отображения
            display_img = cv2.resize(display_img, display_size)

            # Информация на изображении
            info_text = [
                f"Стопа: {foot_name}",
                f"Индекс: {index:.1f}" if index else "Индекс: -",
                f"Тип: {foot_type}" if foot_type else "Тип: -",
                f"Прогресс: {len(completed) + 1}/{len(all_feet)}"
            ]
            y_offset = 25
            for text in info_text:
                cv2.putText(display_img, text, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30

            cv2.imshow("Interactive Tuning", display_img)

            key = cv2.waitKey(30) & 0xFF

            if key == ord('s'):
                if index is not None:
                    results[foot_name] = {
                        'index': float(index),
                        'foot_type': foot_type,
                        'params': params
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
                    print("  ⚠️ Не удалось рассчитать индекс. Попробуйте другие параметры.")

            elif key == ord(' '):
                skipped.add(foot_name)
                progress['results'] = results
                progress['completed'] = list(completed)
                progress['skipped'] = list(skipped)
                save_progress(progress)
                print(f"  ⏭️ Пропущено: {foot_name}")
                break

            elif key == ord('r'):
                default_params = get_default_params()
                cv2.setTrackbarPos("H_LOW", "Controls", default_params['H_LOW'])
                cv2.setTrackbarPos("H_HIGH", "Controls", default_params['H_HIGH'])
                cv2.setTrackbarPos("S_LOW", "Controls", default_params['S_LOW'])
                cv2.setTrackbarPos("S_HIGH", "Controls", default_params['S_HIGH'])
                cv2.setTrackbarPos("V_LOW", "Controls", default_params['V_LOW'])
                cv2.setTrackbarPos("V_HIGH", "Controls", default_params['V_HIGH'])
                cv2.setTrackbarPos("MIN_AREA", "Controls", default_params['MIN_AREA'])
                print("  🔄 Параметры сброшены")

            elif key == 27:
                print("\n⚠️ Выход по ESC")
                cv2.destroyAllWindows()
                return

    print("\n" + "=" * 60)
    print("✅ ВСЕ СТОПЫ ОБРАБОТАНЫ!")
    print(f"Обработано: {len(completed)}")
    print(f"Пропущено: {len(skipped)}")
    print("=" * 60)
# =============================================================================
# ОСНОВНАЯ ПРОГРАММА
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🦶 АНАЛИЗ СТОП - РАСЧЕТ ИНДЕКСА ШТРИТТЕРА")
    print("=" * 60)
    print(f"📂 Исходные файлы: {INPUT_DIR}")
    print(f"📂 Результаты: {SPLIT_DIR}")
    print("=" * 60)

    split_and_save_all()
    interactive_tuning()

    print("\n" + "=" * 60)
    print("🏁 ПРОГРАММА ЗАВЕРШЕНА")
    print("=" * 60)