import cv2
import numpy as np
from rembg import remove


# ==================== ВАШИ ИСХОДНЫЕ ФУНКЦИИ ====================

def resize_with_padding(image, target_size=(512, 512)):
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
    h, w = src_image.shape[:2]
    no_bg = remove(src_image)
    mask = no_bg[:, :, 3]
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    bboxes = sorted([cv2.boundingRect(c) for c in contours], key=lambda b: b[0])
    left_edge = bboxes[0][0] + bboxes[0][2]
    right_edge = bboxes[1][0]
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


def imshow_fit(winname, image, max_width=900, max_height=700):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        display = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        display = image
    cv2.imshow(winname, display)


# ==================== ФУНКЦИИ ФИЛЬТРАЦИИ АРТЕФАКТОВ ====================

def filter_foot_contour(contours, image_shape, debug=False):
    if not contours:
        return None

    h, w = image_shape[:2]
    valid_contours = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect_ratio = bh / max(bw, 1)

        if aspect_ratio < 1.3:
            continue

        edge_margin = 30
        if x < edge_margin or y < edge_margin or \
                x + bw > w - edge_margin or y + bh > h - edge_margin:
            if area < 5000:
                continue

        valid_contours.append(cnt)

    if not valid_contours:
        return None

    return max(valid_contours, key=cv2.contourArea)


def filter_toe_artifacts(points, mask_shape, debug=False):
    h, w = mask_shape[:2] if len(mask_shape) == 3 else mask_shape

    if len(points) < 10:
        return points

    y_coords = points[:, 1]
    y_min, y_max = np.min(y_coords), np.max(y_coords)

    toe_zone_top = y_min
    toe_zone_bottom = y_min + int((y_max - y_min) * 0.15)

    filtered_points = []

    for y in range(toe_zone_top, min(toe_zone_bottom + 1, h - 1)):
        row_points = points[points[:, 1] == y]
        if len(row_points) > 0:
            width = np.max(row_points[:, 0]) - np.min(row_points[:, 0])
            if width > 40:
                filtered_points.extend(row_points)

    remaining = points[points[:, 1] >= toe_zone_bottom]
    filtered_points.extend(remaining)

    if len(filtered_points) < 10:
        return points

    return np.array(filtered_points)


def filter_artifact_by_shape(mask, debug=False):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask

    main_contour = filter_foot_contour(contours, mask.shape, debug)

    if main_contour is None:
        main_contour = max(contours, key=cv2.contourArea)

    mask_filtered = np.zeros_like(mask)
    cv2.drawContours(mask_filtered, [main_contour], -1, 255, -1)

    return mask_filtered


# ==================== ФУНКЦИИ ДЛЯ ЗЕЛЁНОЙ ПЛАНТОГРАММЫ ====================

def extract_green_plantogram_with_params(image_bgr, lower_green, upper_green, kernel_size=5, median_blur=5,
                                         filter_artifacts=True):
    """
    Выделяет зелёную область с заданными параметрами.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)

    if kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    if median_blur > 1:
        mask = cv2.medianBlur(mask, median_blur)

    if filter_artifacts:
        mask = filter_artifact_by_shape(mask, debug=False)

    return mask


def extract_green_plantogram(image_bgr):
    """
    Выделяет зелёную область (отпечаток стопы) из BGR изображения
    с фильтрацией артефактов.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # ПОДОБРАННЫЕ ПАРАМЕТРЫ (сохранённые)
    lower_green = np.array([0, 235, 123])
    upper_green = np.array([132, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Морфологическая очистка
    kernel = np.ones((21, 21), np.uint8)  # kernel_size = 21
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.medianBlur(mask, 15)  # median_blur = 15

    # Фильтрация артефактов по форме
    mask_filtered = filter_artifact_by_shape(mask, debug=False)

    return mask_filtered

def normalize_coordinates(point, image_shape):
    if len(image_shape) == 3:
        h, w = image_shape[:2]
    else:
        h, w = image_shape
    x = max(0, min(w - 1, point[0]))
    y = max(0, min(h - 1, point[1]))
    return (x, y)


def find_inner_tangent(mask):
    """
    Находит касательную к внутреннему (левому) краю стопы.

    Логика:
    - Находим левый край стопы (минимальный X для каждого Y)
    - Находим центральную Y (середина по высоте)
    - Движемся от центра ВВЕРХ: ищем первую точку с минимальным X
    - Движемся от центра ВНИЗ: ищем первую точку с минимальным X
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None

    cnt = filter_foot_contour(contours, mask.shape)
    if cnt is None:
        cnt = max(contours, key=cv2.contourArea)

    if cnt is None or len(cnt) < 10:
        return None, None, None

    try:
        points = cnt.reshape(-1, 2)
        if len(points) < 50:
            return None, None, None
    except Exception as e:
        print(f"Ошибка при reshape контура: {e}")
        return None, None, None

    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
    h_total = y_max - y_min

    left_edge = []
    for y in range(y_min, y_max + 1):
        if y >= mask.shape[0]:
            break
        white_x = np.where(mask[y, :] > 0)[0]
        if len(white_x) > 0:
            left_edge.append([white_x[0], y])

    if len(left_edge) < 20:
        return None, None, None

    left_edge = np.array(left_edge)

    xs = left_edge[:, 0]
    window = 5
    kernel = np.ones(window) / window
    xs_smoothed = np.convolve(xs, kernel, mode='same')
    left_edge[:, 0] = xs_smoothed.astype(int)

    center_idx = len(left_edge) // 2
    center_y = left_edge[center_idx][1]
    center_x = left_edge[center_idx][0]

    # Движение от центра ВВЕРХ
    A = None
    min_x_so_far = center_x
    for i in range(center_idx, -1, -1):
        current_x = left_edge[i][0]
        if current_x < min_x_so_far:
            min_x_so_far = current_x
            A = (int(current_x), int(left_edge[i][1]))
        if center_y - left_edge[i][1] > h_total * 0.3:
            break

    if A is None:
        A_idx = len(left_edge) // 4
        A = (int(left_edge[A_idx][0]), int(left_edge[A_idx][1]))

    # Движение от центра ВНИЗ
    B = None
    min_x_so_far = center_x
    for i in range(center_idx, len(left_edge)):
        current_x = left_edge[i][0]
        if current_x < min_x_so_far:
            min_x_so_far = current_x
            B = (int(current_x), int(left_edge[i][1]))
        if left_edge[i][1] - center_y > h_total * 0.3:
            break

    if B is None:
        B_idx = len(left_edge) * 3 // 4
        B = (int(left_edge[B_idx][0]), int(left_edge[B_idx][1]))

    h, w = mask.shape
    A = (max(0, min(w - 1, A[0])), max(0, min(h - 1, A[1])))
    B = (max(0, min(w - 1, B[0])), max(0, min(h - 1, B[1])))

    angle = np.arctan2(B[1] - A[1], B[0] - A[0])

    return A, B, angle


def find_wide_points(contour, y_percent_start=0.2, y_percent_end=0.6):
    if contour is None or len(contour) < 10:
        return (0, 0), (0, 0), 0

    try:
        points = contour.reshape(-1, 2)
        if len(points) < 20:
            return (0, 0), (0, 0), 0
    except Exception as e:
        print(f"Ошибка при reshape контура: {e}")
        return (0, 0), (0, 0), 0

    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])

    if y_max == y_min:
        return (0, 0), (0, 0), 0

    y_start = y_min + int((y_max - y_min) * (y_percent_start + 0.05))
    y_end = y_min + int((y_max - y_min) * (y_percent_end - 0.05))

    y_start = max(y_start, y_min + 10)
    y_end = min(y_end, y_max - 10)

    max_width = 0
    best_y = y_start
    best_left, best_right = 0, 0

    for y in range(y_start, y_end + 1):
        row_points = points[points[:, 1] == y]
        if len(row_points) >= 4:
            left = np.min(row_points[:, 0])
            right = np.max(row_points[:, 0])
            width = right - left
            if width > max_width:
                max_width = width
                best_y = y
                best_left, best_right = left, right

    return (best_left, best_y), (best_right, best_y), max_width


def calculate_k2(plantogram_mask):
    contours, _ = cv2.findContours(plantogram_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("   Контуры не найдены")
        return None, None

    cnt = filter_foot_contour(contours, plantogram_mask.shape)
    if cnt is None:
        print("   Не найден подходящий контур стопы")
        return None, None

    if len(cnt) < 10:
        print("   Контур слишком маленький")
        return None, None

    try:
        points = cnt.reshape(-1, 2)
        if len(points) < 50:
            print(f"   Слишком мало точек в контуре: {len(points)}")
            return None, None
    except Exception as e:
        print(f"   Ошибка при reshape: {e}")
        return None, None

    points = filter_toe_artifacts(points, plantogram_mask.shape)

    A, B, width_AB = find_wide_points(cnt)

    if width_AB == 0:
        print("   Не удалось найти ширину AB")
        return None, None

    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])

    if y_max == y_min:
        print("   Вырожденный контур")
        return None, None

    toe_zone = points[points[:, 1] <= y_min + int((y_max - y_min) * 0.1)]
    heel_zone = points[points[:, 1] >= y_max - int((y_max - y_min) * 0.1)]

    if len(toe_zone) > 0:
        toe_x = int(np.median(toe_zone[:, 0]))
        toe_y = int(np.mean(toe_zone[:, 1]))
        toe_point = (toe_x, toe_y)
    else:
        toe_point = (int(np.mean(points[:, 0])), y_min)

    if len(heel_zone) > 0:
        heel_x = int(np.median(heel_zone[:, 0]))
        heel_y = int(np.mean(heel_zone[:, 1]))
        heel_point = (heel_x, heel_y)
    else:
        heel_point = (int(np.mean(points[:, 0])), y_max)

    length_EF = np.linalg.norm(np.array(heel_point) - np.array(toe_point))
    if length_EF == 0:
        print("   Длина EF равна 0")
        return None, None

    k2 = width_AB / length_EF
    return k2, (A, B, toe_point, heel_point, width_AB, length_EF)


def visualize_tangent(foot_image, mask):
    A, B, angle = find_inner_tangent(mask)

    result = foot_image.copy()

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

    if A is None or B is None:
        cv2.putText(result, "No tangent found", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return result

    A_norm = normalize_coordinates(A, result.shape)
    B_norm = normalize_coordinates(B, result.shape)

    h, w = result.shape[:2]

    dx = B_norm[0] - A_norm[0]
    dy = B_norm[1] - A_norm[1]

    if dx != 0 or dy != 0:
        if dy != 0:
            t_top = -A_norm[1] / dy
            top_x = int(A_norm[0] + t_top * dx)
            top_point = (max(0, min(w - 1, top_x)), 0)

            t_bottom = (h - 1 - A_norm[1]) / dy
            bottom_x = int(A_norm[0] + t_bottom * dx)
            bottom_point = (max(0, min(w - 1, bottom_x)), h - 1)
        else:
            top_point = (A_norm[0], 0)
            bottom_point = (A_norm[0], h - 1)

        cv2.line(result, top_point, bottom_point, (255, 0, 0), 2)

    cv2.circle(result, A_norm, 8, (0, 0, 255), -1)
    cv2.circle(result, B_norm, 8, (0, 0, 255), -1)
    cv2.putText(result, "A", (A_norm[0] - 30, A_norm[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(result, "B", (B_norm[0] - 30, B_norm[1] + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        try:
            points = cnt.reshape(-1, 2)
            if len(points) > 0:
                y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
                mid_y = int((y_min + y_max) / 2)
                cv2.line(result, (0, mid_y), (w, mid_y), (0, 255, 255), 1)
                cv2.putText(result, f"mid_Y = {mid_y}", (10, mid_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        except:
            pass

    return result


def visualize_k2(image_bgr, plantogram_mask, k2, points):
    result = image_bgr.copy()

    mask_colored = np.zeros_like(image_bgr)
    mask_colored[:, :, 1] = plantogram_mask
    result = cv2.addWeighted(result, 0.6, mask_colored, 0.4, 0)

    A, B, toe, heel, width_AB, length_EF = points

    A_norm = normalize_coordinates(A, result.shape)
    B_norm = normalize_coordinates(B, result.shape)
    toe_norm = normalize_coordinates(toe, result.shape)
    heel_norm = normalize_coordinates(heel, result.shape)

    cv2.line(result, A_norm, B_norm, (255, 0, 0), 3)
    cv2.circle(result, A_norm, 8, (0, 255, 0), -1)
    cv2.circle(result, B_norm, 8, (0, 255, 0), -1)
    cv2.putText(result, "A", (A_norm[0] - 20, A_norm[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(result, "B", (B_norm[0] + 5, B_norm[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.line(result, toe_norm, heel_norm, (0, 255, 255), 3)
    cv2.circle(result, toe_norm, 8, (255, 0, 255), -1)
    cv2.circle(result, heel_norm, 8, (255, 0, 255), -1)
    cv2.putText(result, "E", (toe_norm[0] - 20, toe_norm[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    cv2.putText(result, "F", (heel_norm[0] - 20, heel_norm[1] + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    cv2.putText(result, f"AB = {width_AB:.1f} px", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(result, f"EF = {length_EF:.1f} px", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(result, f"k2 = AB/EF = {k2:.3f}", (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if k2 > 0.35:
        status = "ПОПЕРЕЧНОЕ ПЛОСКОСТОПИЕ (k2 > 0.35)"
        color = (0, 0, 255)
    elif k2 >= 0.3:
        status = "НОРМА (0.3 - 0.35)"
        color = (0, 255, 0)
    else:
        status = "ВЫСОКИЙ СВОД (k2 < 0.3)"
        color = (255, 255, 0)
    cv2.putText(result, status, (10, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return result


# ==================== ФУНКЦИИ ДЛЯ ИНДЕКСА ШТРИТЕРА ====================
def calculate_strieter_index(plantogram_mask):
    """
    Рассчитывает индекс Штритера по маске плантограммы.

    Методика:
    1. Находим самые левые точки на контуре (минимальный X)
    2. Проводим через них касательную АБ
    3. Находим середину отрезка АБ
    4. Из середины отрезка АБ возводится перпендикуляр (ВД) до пересечения
       с наружным краем отпечатка
    5. Отмечаются точки Г и Д – пересечения перпендикуляра с внутренней и
       наружной частями отпечатка стопы соответственно
    6. Индекс: I = ГД * 100 / ВД
    """
    contours, _ = cv2.findContours(plantogram_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("   Контуры не найдены")
        return None, None

    # Берём самый большой контур
    cnt = max(contours, key=cv2.contourArea)

    try:
        points = cnt.reshape(-1, 2)
        if len(points) < 50:
            print(f"   Слишком мало точек: {len(points)}")
            return None, None
    except Exception as e:
        print(f"   Ошибка при reshape: {e}")
        return None, None

    # --- НАХОДИМ САМЫЕ ЛЕВЫЕ ТОЧКИ КОНТУРА (минимальный X) ---
    # Сортируем точки по X (по возрастанию)
    sorted_by_x = points[np.argsort(points[:, 0])]

    # Берём 10% самых левых точек
    leftmost_count = max(10, int(len(sorted_by_x) * 0.1))
    leftmost_points = sorted_by_x[:leftmost_count]

    # Находим верхнюю и нижнюю точки среди самых левых
    # Верхняя точка А - минимальный Y среди левых точек
    idx_a = np.argmin(leftmost_points[:, 1])
    A = (int(leftmost_points[idx_a][0]), int(leftmost_points[idx_a][1]))

    # Нижняя точка Б - максимальный Y среди левых точек
    idx_b = np.argmax(leftmost_points[:, 1])
    B = (int(leftmost_points[idx_b][0]), int(leftmost_points[idx_b][1]))

    # Проверяем, что точки не совпадают
    if A[1] == B[1]:
        print("   Точки А и Б совпадают по Y")
        return None, None

    # --- ШАГ 2: Находим середину отрезка АБ ---
    mid_x = (A[0] + B[0]) // 2
    mid_y = (A[1] + B[1]) // 2
    mid_point = (mid_x, mid_y)

    # --- ШАГ 3: Строим перпендикуляр к АБ в середине ---
    # Вектор направления АБ
    dx = B[0] - A[0]
    dy = B[1] - A[1]

    # Перпендикулярный вектор (нормализованный)
    perp_x = -dy
    perp_y = dx

    # Нормализуем
    length = np.sqrt(perp_x ** 2 + perp_y ** 2)
    if length == 0:
        print("   Нулевая длина вектора")
        return None, None
    perp_x = perp_x / length
    perp_y = perp_y / length

    # --- ИЩЕМ ТОЧКУ Д (наружный край) ---
    # Идём от середины в сторону наружного края (вправо)
    step = 0.5
    max_dist = max(plantogram_mask.shape) * 3

    point_D = None
    for t in np.arange(0, max_dist, step):
        x = int(mid_x + perp_x * t)
        y = int(mid_y + perp_y * t)
        if x < 0 or x >= plantogram_mask.shape[1] or y < 0 or y >= plantogram_mask.shape[0]:
            break
        if plantogram_mask[y, x] > 0:
            point_D = (x, y)
            break

    # --- ИЩЕМ ТОЧКУ Г (внутренний край) ---
    # Идём от середины в сторону внутреннего края (влево)
    point_G = None
    for t in np.arange(0, max_dist, step):
        x = int(mid_x - perp_x * t)
        y = int(mid_y - perp_y * t)
        if x < 0 or x >= plantogram_mask.shape[1] or y < 0 or y >= plantogram_mask.shape[0]:
            break
        if plantogram_mask[y, x] > 0:
            point_G = (x, y)
            break

    if point_G is None or point_D is None:
        print("   Не найдены точки пересечения")
        return None, None

    # --- ШАГ 4: Рассчитываем индекс ---
    # ВД - расстояние от середины АБ до наружного края (точка Д)
    dist_VD = np.sqrt((point_D[0] - mid_x) ** 2 + (point_D[1] - mid_y) ** 2)

    # ГД - расстояние между внутренней и наружной точками
    dist_GD = np.sqrt((point_D[0] - point_G[0]) ** 2 + (point_D[1] - point_G[1]) ** 2)

    if dist_VD == 0:
        print("   ВД = 0")
        return None, None

    strieter_index = (dist_GD / dist_VD) * 100

    return strieter_index, (A, B, mid_point, point_G, point_D, dist_GD, dist_VD)

def visualize_strieter_index(image_bgr, plantogram_mask, strieter_index, points):
    """
    Визуализирует индекс Штритера.
    """
    result = image_bgr.copy()

    # Полупрозрачная маска
    mask_colored = np.zeros_like(image_bgr)
    mask_colored[:, :, 1] = plantogram_mask
    result = cv2.addWeighted(result, 0.5, mask_colored, 0.5, 0)

    A, B, mid_point, point_G, point_D, dist_GD, dist_VD = points

    # Рисуем касательную АБ (красная)
    cv2.line(result, A, B, (0, 0, 255), 2)
    cv2.circle(result, A, 8, (0, 0, 255), -1)
    cv2.circle(result, B, 8, (0, 0, 255), -1)
    cv2.putText(result, "A", (A[0] - 30, A[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(result, "B", (B[0] - 30, B[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Рисуем середину (жёлтая точка)
    cv2.circle(result, mid_point, 8, (0, 255, 255), -1)
    cv2.putText(result, "O", (mid_point[0] + 10, mid_point[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Рисуем перпендикуляр ВД (жёлтая линия от середины до наружного края)
    cv2.line(result, mid_point, point_D, (0, 255, 255), 2)

    # Рисуем отрезок ГД (зелёная линия)
    cv2.line(result, point_G, point_D, (0, 255, 0), 3)

    # Точки Г и Д
    cv2.circle(result, point_G, 8, (255, 0, 255), -1)
    cv2.circle(result, point_D, 8, (255, 0, 255), -1)
    cv2.putText(result, "G", (point_G[0] - 30, point_G[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    cv2.putText(result, "D", (point_D[0] + 10, point_D[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # Информация
    cv2.putText(result, f"GD = {dist_GD:.1f} px", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(result, f"VD = {dist_VD:.1f} px", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(result, f"I = GD/VD * 100 = {strieter_index:.1f}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255), 2)

    # Диагноз
    if strieter_index <= 36:
        status = "ВЫСОКОСВОДЧАТАЯ (полая) стопа"
        color = (255, 255, 0)
    elif strieter_index <= 43:
        status = "ПОВЫШЕННЫЙ СВОД"
        color = (0, 255, 255)
    elif strieter_index <= 50:
        status = "НОРМАЛЬНАЯ стопа"
        color = (0, 255, 0)
    elif strieter_index <= 60:
        status = "УПЛОЩЕННАЯ стопа"
        color = (0, 165, 255)
    else:
        status = "ПЛОСКОСТОПИЕ"
        color = (0, 0, 255)

    cv2.putText(result, status, (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return result
# ==================== ИНТЕРАКТИВНАЯ ОТЛАДКА С КАСАТЕЛЬНОЙ ====================

def debug_green_range_with_tangent(image_bgr):
    """
    Интерактивная отладка с ползунками.
    Маска и касательная обновляются в реальном времени.
    Нажмите 's' для сохранения параметров, 'q' для выхода.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Масштабируем для отображения
    screen_height = 600
    screen_width = 800
    h, w = image_bgr.shape[:2]
    scale = min(screen_width / w, screen_height / h, 1.0)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        display_img = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        display_hsv = cv2.resize(hsv, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        display_img = image_bgr
        display_hsv = hsv
        new_w, new_h = w, h

    # Создаём окна
    cv2.namedWindow("Green Mask", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Green Mask", min(500, new_w), min(500, new_h))
    cv2.namedWindow("Tangent Result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tangent Result", min(800, new_w), min(600, new_h))
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controls", 500, 400)

    def nothing(x):
        pass

    # HSV трекбары
    cv2.createTrackbar("H min", "Controls", 24, 179, nothing)
    cv2.createTrackbar("H max", "Controls", 81, 179, nothing)
    cv2.createTrackbar("S min", "Controls", 104, 255, nothing)
    cv2.createTrackbar("S max", "Controls", 255, 255, nothing)
    cv2.createTrackbar("V min", "Controls", 78, 255, nothing)
    cv2.createTrackbar("V max", "Controls", 255, 255, nothing)

    # Морфология
    cv2.createTrackbar("Kernel size", "Controls", 5, 20, nothing)
    cv2.createTrackbar("Median blur", "Controls", 5, 15, nothing)
    cv2.createTrackbar("Filter artifacts", "Controls", 1, 1, nothing)

    print("\n=== ИНТЕРАКТИВНАЯ НАСТРОЙКА ЗЕЛЁНОЙ МАСКИ ===")
    print("Окно 'Controls' - двигайте трекбары")
    print("Окно 'Green Mask' - показывает маску (белое = выделено)")
    print("Окно 'Tangent Result' - показывает касательную и k2")
    print("Цель: сделать белой только зелёную область стопы")
    print("\nНажмите 's' для сохранения параметров")
    print("Нажмите 'q' для выхода")

    while True:
        # Получаем значения
        h_min = cv2.getTrackbarPos("H min", "Controls")
        h_max = cv2.getTrackbarPos("H max", "Controls")
        s_min = cv2.getTrackbarPos("S min", "Controls")
        s_max = cv2.getTrackbarPos("S max", "Controls")
        v_min = cv2.getTrackbarPos("V min", "Controls")
        v_max = cv2.getTrackbarPos("V max", "Controls")
        kernel_size = cv2.getTrackbarPos("Kernel size", "Controls")
        median_blur = cv2.getTrackbarPos("Median blur", "Controls")
        filter_artifacts = cv2.getTrackbarPos("Filter artifacts", "Controls")

        if kernel_size % 2 == 0:
            kernel_size += 1
        if median_blur % 2 == 0:
            median_blur += 1

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])

        # Создаём маску
        mask = cv2.inRange(display_hsv, lower, upper)

        if kernel_size > 1:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        if median_blur > 1:
            mask = cv2.medianBlur(mask, median_blur)

        if filter_artifacts == 1:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                max_contour = max(contours, key=cv2.contourArea)
                mask_filtered = np.zeros_like(mask)
                cv2.drawContours(mask_filtered, [max_contour], -1, 255, -1)
                mask = mask_filtered

        # Показываем маску
        cv2.imshow("Green Mask", mask)

        # Рисуем касательную на исходном (немасштабированном) изображении
        if np.sum(mask) > 1000:
            # Применяем маску к исходному изображению для расчётов
            full_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            A, B, angle = find_inner_tangent(full_mask)
            k2_result = calculate_k2(full_mask)

            result = image_bgr.copy()
            mask_colored = np.zeros_like(result)
            mask_colored[:, :, 1] = full_mask
            result = cv2.addWeighted(result, 0.6, mask_colored, 0.4, 0)

            if A and B:
                cv2.circle(result, A, 8, (0, 0, 255), -1)
                cv2.circle(result, B, 8, (0, 0, 255), -1)
                cv2.putText(result, "A", (A[0] - 30, A[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(result, "B", (B[0] - 30, B[1] + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Рисуем линию через A и B
                h_img, w_img = result.shape[:2]
                dx = B[0] - A[0]
                dy = B[1] - A[1]
                if dx != 0 or dy != 0:
                    if dy != 0:
                        t_top = -A[1] / dy
                        top_x = int(A[0] + t_top * dx)
                        t_bottom = (h_img - 1 - A[1]) / dy
                        bottom_x = int(A[0] + t_bottom * dx)
                        cv2.line(result, (max(0, min(w_img - 1, top_x)), 0),
                                 (max(0, min(w_img - 1, bottom_x)), h_img - 1), (255, 0, 0), 2)

            if k2_result[0] is not None:
                k2, points = k2_result
                cv2.putText(result, f"k2 = {k2:.3f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Масштабируем результат для отображения
            if scale < 1.0:
                result_display = cv2.resize(result, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                result_display = result

            cv2.imshow("Tangent Result", result_display)
        else:
            # Маска пустая
            result_display = display_img.copy()
            cv2.putText(result_display, "No green area detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Tangent Result", result_display)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            print("\n=== СОХРАНЁННЫЕ ЗНАЧЕНИЯ ===")
            print(f"lower_green = np.array([{h_min}, {s_min}, {v_min}])")
            print(f"upper_green = np.array([{h_max}, {s_max}, {v_max}])")
            print(f"\nРекомендуемые параметры:")
            print(f"kernel_size = {kernel_size}")
            print(f"median_blur = {median_blur}")
            print(f"filter_artifacts = {filter_artifacts == 1}")

    cv2.destroyAllWindows()


# ==================== ОСНОВНОЙ БЛОК ====================

# ==================== ОСНОВНОЙ БЛОК ====================

# ==================== ОСНОВНОЙ БЛОК ====================

if __name__ == "__main__":
    path = "./foots/4/IMG_0302.jpg"
    # path = "./foots/1/IMG_0315.jpg"

    src = cv2.imread(path)
    if src is None:
        print(f"Ошибка загрузки {path}")
        exit()

    # Разделяем стопы
    left_src, right_src, debug_split = split_feet_smart(src, debug=True)

    # === ИНТЕРАКТИВНАЯ НАСТРОЙКА (раскомментируйте для подбора параметров) ===
    # debug_green_range_with_tangent(left_src)

    # === ОСНОВНОЙ АНАЛИЗ С ПОДОБРАННЫМИ ПАРАМЕТРАМИ ===
    print("\n=== АНАЛИЗ ПЛАНТОГРАММЫ (ИНДЕКС ШТРИТЕРА) ===")

    green_mask = extract_green_plantogram(left_src)

    mask_area = np.sum(green_mask) // 255
    print(f"Площадь маски: {mask_area} пикселей")

    if mask_area == 0:
        print("ОШИБКА: Зелёная область не найдена!")
        exit()

    # --- РАСЧЁТ ИНДЕКСА ШТРИТЕРА ---
    print("\n1. Расчёт индекса Штритера...")
    strieter_result = calculate_strieter_index(green_mask)

    if strieter_result[0] is not None:
        strieter_index, points = strieter_result
        print(f"   Индекс Штритера: I = {strieter_index:.2f}")
        print(f"   GD = {points[5]:.1f} px")
        print(f"   VD = {points[6]:.1f} px")

        if strieter_index <= 36:
            print("   ДИАГНОЗ: Высокосводчатая (полая) стопа")
        elif strieter_index <= 43:
            print("   ДИАГНОЗ: Повышенный свод")
        elif strieter_index <= 50:
            print("   ДИАГНОЗ: Нормальная стопа")
        elif strieter_index <= 60:
            print("   ДИАГНОЗ: Уплощенная стопа")
        else:
            print("   ДИАГНОЗ: Плоскостопие")

        # Визуализация индекса Штритера (только она!)
        strieter_viz = visualize_strieter_index(left_src, green_mask, strieter_index, points)
        imshow_fit("Strieter Index", strieter_viz)
    else:
        print("   ОШИБКА: Не удалось рассчитать индекс Штритера")

    print("\nНажмите любую клавишу для выхода...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()