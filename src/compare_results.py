"""
compare_results.py - Сравнение ручных и автоматических расчетов
"""

import cv2
import json
import os
import glob
import numpy as np

from auto_analysis import (
    get_foot_mask,
    get_largest_contour,
    find_AB_points_final,
    calculate_strieter_index_full,
    draw_contours,
    get_foot_type
)

# =============================================================================
# ЗАГРУЗКА ДАННЫХ
# =============================================================================

def load_manual_data():
    """Загружает ручные расчеты из progress_grayscale.json"""
    if not os.path.exists('progress_grayscale.json'):
        print("❌ Файл progress_grayscale.json не найден!")
        return None, None
    
    with open('progress_grayscale.json', 'r') as f:
        data = json.load(f)
    
    return data.get('results', {}), data.get('completed', [])


def load_best_params():
    """Загружает лучшие параметры из best_grayscale.json"""
    if not os.path.exists('best_grayscale.json'):
        print("❌ Файл best_grayscale.json не найден!")
        return None
    
    with open('best_grayscale.json', 'r') as f:
        data = json.load(f)
    
    print(f"📊 Параметры ГА (поколение {data.get('generation', '?')})")
    print(f"  Фитнес: {data.get('fitness', 0):.2f}%")
    print(f"  THRESHOLD: {data['params']['THRESHOLD']}")
    print(f"  MIN_AREA: {data['params']['MIN_AREA']}")
    print(f"  CLOSE_SIZE: {data['params']['CLOSE_SIZE']}")
    print()
    
    return data['params']


# =============================================================================
# ФУНКЦИЯ ДЛЯ GRAYSCALE
# =============================================================================

def get_plantogram_mask_grayscale(image, foot_mask, threshold, min_area, close_size):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    if close_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    mask = cv2.bitwise_and(mask, foot_mask)
    
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
    
    return clean_mask, plant_contour


def calculate_index_auto(image, params):
    """Рассчитывает индекс с параметрами ГА"""
    threshold = params['THRESHOLD']
    min_area = params['MIN_AREA']
    close_size = params['CLOSE_SIZE']
    
    foot_mask = get_foot_mask(image)
    clean_mask, plant_contour = get_plantogram_mask_grayscale(
        image, foot_mask, threshold, min_area, close_size
    )
    
    if plant_contour is None:
        return None, None
    
    A, B = find_AB_points_final(plant_contour)
    index, V, G, D, foot_type = calculate_strieter_index_full(plant_contour, A, B, None)
    
    return index, foot_type


# =============================================================================
# ОСНОВНАЯ ПРОГРАММА
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("📊 СРАВНЕНИЕ РУЧНЫХ И АВТОМАТИЧЕСКИХ РАСЧЕТОВ")
    print("=" * 60)
    
    # 1. Загружаем ручные данные
    manual_results, completed = load_manual_data()
    if manual_results is None:
        exit()
    
    print(f"📂 Ручных расчетов: {len(completed)} стоп")
    
    # 2. Загружаем параметры ГА
    params = load_best_params()
    if params is None:
        exit()
    
    # 3. Сравниваем
    print("\n" + "=" * 60)
    print("🔍 СРАВНЕНИЕ ПО КАЖДОЙ СТОПЕ")
    print("=" * 60)
    print(f"{'Стопа':<8} {'Ручной':<10} {'Авто':<10} {'Ошибка':<10} {'Статус':<10}")
    print("-" * 60)
    
    comparison = []
    success = 0
    failed = 0
    no_manual = 0
    
    # Получаем все стопы из папки
    image_files = glob.glob("./split_feet/*.png")
    image_files = sorted(image_files)
    
    for img_path in image_files:
        name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Ручной индекс
        if name in manual_results:
            manual_idx = manual_results[name]['index']
        else:
            manual_idx = None
            no_manual += 1
            continue
        
        # Автоматический индекс
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        auto_idx, foot_type = calculate_index_auto(image, params)
        
        if auto_idx is not None:
            error = abs(auto_idx - manual_idx)
            error_pct = (error / manual_idx) * 100 if manual_idx > 0 else 0
            
            status = "✅" if error < 5 else "⚠️" if error < 10 else "❌"
            
            print(f"{name:<8} {manual_idx:<10.1f} {auto_idx:<10.1f} {error:<10.1f} {status}")
            
            comparison.append({
                'name': name,
                'manual': manual_idx,
                'auto': auto_idx,
                'error': error,
                'error_pct': error_pct,
                'status': status
            })
            success += 1
        else:
            print(f"{name:<8} {manual_idx:<10.1f} {'—':<10} {'—':<10} ❌ (нет контура)")
            failed += 1
    
    # 4. СТАТИСТИКА
    print("\n" + "=" * 60)
    print("📊 ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 60)
    
    print(f"\n📌 ОБЩАЯ СТАТИСТИКА:")
    print(f"  ✅ Совпадают (ошибка < 5): {len([c for c in comparison if c['error'] < 5])}")
    print(f"  ⚠️ Среднее отклонение (5-10): {len([c for c in comparison if 5 <= c['error'] < 10])}")
    print(f"  ❌ Большое отклонение (> 10): {len([c for c in comparison if c['error'] >= 10])}")
    print(f"  ❌ Нет контура: {failed}")
    
    if comparison:
        errors = [c['error'] for c in comparison]
        errors_pct = [c['error_pct'] for c in comparison]
        manual_indices = [c['manual'] for c in comparison]
        auto_indices = [c['auto'] for c in comparison]
        
        print(f"\n📈 ЧИСЛОВЫЕ ПОКАЗАТЕЛИ:")
        print(f"  Средняя абсолютная ошибка: {np.mean(errors):.2f}")
        print(f"  Средняя относительная ошибка: {np.mean(errors_pct):.2f}%")
        print(f"  Максимальная ошибка: {np.max(errors):.2f}")
        print(f"  Минимальная ошибка: {np.min(errors):.2f}")
        
        print(f"\n📊 СРАВНЕНИЕ ИНДЕКСОВ:")
        print(f"  Ручные: средний = {np.mean(manual_indices):.2f}, min = {np.min(manual_indices):.2f}, max = {np.max(manual_indices):.2f}")
        print(f"  Авто:   средний = {np.mean(auto_indices):.2f}, min = {np.min(auto_indices):.2f}, max = {np.max(auto_indices):.2f}")
    
    print(f"\n📌 ПРОПУЩЕНО:")
    print(f"  Нет ручного расчета: {no_manual}")
    print(f"  Не найден контур: {failed}")
    print(f"  Всего стоп: {len(comparison) + failed + no_manual}")
    
    # 5. Сохраняем результат
    result_data = {
        'comparison': comparison,
        'stats': {
            'total': len(comparison) + failed + no_manual,
            'success': len(comparison),
            'failed': failed,
            'no_manual': no_manual,
            'avg_error': np.mean([c['error'] for c in comparison]) if comparison else 0,
            'avg_error_pct': np.mean([c['error_pct'] for c in comparison]) if comparison else 0,
        }
    }
    
    with open('comparison_results.json', 'w') as f:
        json.dump(result_data, f, indent=2)
    
    print(f"\n✅ Результаты сохранены в comparison_results.json")
    print("=" * 60)
