"""
ga_train_grayscale.py - Обучение ГА для grayscale подхода
Ищет оптимальные параметры: THRESHOLD, MIN_AREA, CLOSE_SIZE
"""

import cv2
import numpy as np
import json
import random
import os
from copy import deepcopy
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

# Импорт из основного кода
from auto_analysis import (
    get_foot_mask,
    get_largest_contour,
    find_AB_points_final,
    calculate_strieter_index_full,
    draw_contours,
    FOOT_TYPE_BOUNDARIES
)

# =============================================================================
# КОНСТАНТЫ
# =============================================================================

# Параметры для grayscale подхода (3 параметра!)
OPTIMIZATION_PARAMS = {
    'THRESHOLD': {'min': 50, 'max': 200, 'type': 'int', 'default': 127},
    'MIN_AREA': {'min': 20000, 'max': 100000, 'type': 'int', 'default': 50000},
    'CLOSE_SIZE': {'min': 0, 'max': 30, 'type': 'int', 'default': 0},
}


# =============================================================================
# ФУНКЦИЯ ДЛЯ GRAYSCALE
# =============================================================================

def get_plantogram_mask_grayscale(image, foot_mask, threshold, min_area, close_size):
    """
    Выделение плантограммы через grayscale + пороговую обработку + замыкание.
    """
    # 1. Переводим в grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. CLAHE для усиления контраста
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 3. Пороговая обработка
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # 4. Морфологическое замыкание (соединяет разрывы)
    if close_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 5. Ограничиваем маской стопы
    mask = cv2.bitwise_and(mask, foot_mask)

    # 6. Оставляем только крупные области
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


# =============================================================================
# ЗАГРУЗКА ДАННЫХ (ИСПРАВЛЕНО!)
# =============================================================================

def load_dataset():
    """
    Загружает размеченный датасет из progress_grayscale.json
    """
    print("\n📂 ЗАГРУЗКА ДАННЫХ")
    print("=" * 50)

    if not os.path.exists('progress_grayscale.json'):
        print("❌ Файл progress_grayscale.json не найден!")
        print("   Сначала запустите interactive_grayscale.py для разметки.")
        return []

    with open('progress_grayscale.json', 'r') as f:
        progress = json.load(f)

    results = progress.get('results', {})
    completed = progress.get('completed', [])
    skipped = progress.get('skipped', [])

    print(f"  ✅ Размечено стоп: {len(completed)}")
    print(f"  ⏭️ Пропущено стоп: {len(skipped)}")

    dataset = []
    for foot_name in completed:
        data = results.get(foot_name)
        if data is None:
            continue

        img_path = f"./split_feet/{foot_name}.png"
        img = cv2.imread(img_path)

        if img is None:
            continue

        dataset.append({
            'name': foot_name,
            'image': img,
            'image_path': img_path,
            'reference_index': data['index'],
            'foot_type': data['foot_type'],
            'threshold': data.get('threshold', 127),
            'min_area': data.get('min_area', 50000),
            'close_size': data.get('close_size', 0),
        })

    print(f"  ✅ Датасет готов: {len(dataset)} стоп")
    print("=" * 50)

    return dataset


# =============================================================================
# ФИТНЕС-ФУНКЦИЯ
# =============================================================================

def calculate_index_grayscale(image, foot_mask, threshold, min_area, close_size):
    """
    Рассчитывает индекс через grayscale подход.
    """
    try:
        clean_mask, plant_contour = get_plantogram_mask_grayscale(
            image, foot_mask, threshold, min_area, close_size
        )

        if plant_contour is None:
            return None

        A, B = find_AB_points_final(plant_contour)
        index, V, G, D, foot_type = calculate_strieter_index_full(
            plant_contour, A, B, None
        )

        return index
    except:
        return None


def fitness_function_single(args):
    """
    Фитнес-функция для одного набора параметров.
    """
    params, dataset = args
    threshold = params['THRESHOLD']
    min_area = params['MIN_AREA']
    close_size = params['CLOSE_SIZE']

    errors = []
    valid_count = 0

    for item in dataset:
        foot_mask = get_foot_mask(item['image'])
        calculated = calculate_index_grayscale(
            item['image'], foot_mask, threshold, min_area, close_size
        )
        expected = item['reference_index']

        if calculated is not None:
            error = abs(calculated - expected)
            errors.append(error)
            valid_count += 1

    if valid_count == 0:
        return 0

    avg_error = np.mean(errors)
    max_possible_error = 100
    fitness = max(0, 100 - (avg_error / max_possible_error * 100))

    return fitness


# =============================================================================
# ГЕНЕТИЧЕСКИЙ АЛГОРИТМ
# =============================================================================

class GeneticAlgorithmGrayscale:
    def __init__(self, dataset, pop_size=20, generations=30, mutation_rate=0.15):
        self.dataset = dataset
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.best = None
        self.best_fitness = -float('inf')
        self.history = []

    def create_individual(self):
        """Создает случайную особь (3 параметра)."""
        individual = {}
        for key, param in OPTIMIZATION_PARAMS.items():
            if param['type'] == 'int':
                individual[key] = random.randint(param['min'], param['max'])
        return individual

    def mutate(self, individual):
        """Мутирует особь."""
        mutated = deepcopy(individual)
        for key, param in OPTIMIZATION_PARAMS.items():
            if random.random() < self.mutation_rate:
                step = max(1, (param['max'] - param['min']) // 10)
                mutated[key] += random.randint(-step, step)
                mutated[key] = max(param['min'], min(param['max'], mutated[key]))
        return mutated

    def crossover(self, p1, p2):
        """Скрещивание."""
        child = {}
        for key in p1:
            child[key] = p1[key] if random.random() < 0.5 else p2[key]
        return child

    def evaluate_population_parallel(self, population):
        """Параллельная оценка популяции."""
        args = [(ind, self.dataset) for ind in population]

        print(f"    Оценка {len(population)} особей на {cpu_count()} ядрах...")

        with Pool(processes=6) as pool:
            scores = pool.map(fitness_function_single, args)

        for i, score in enumerate(scores):
            if score > self.best_fitness:
                self.best_fitness = score
                self.best = deepcopy(population[i])

        return scores

    def evolve(self):
        """Запускает эволюцию."""
        print("\n🧬 ЗАПУСК ГЕНЕТИЧЕСКОГО АЛГОРИТМА (GRAYSCALE)")
        print("=" * 60)
        print(f"  Размер популяции: {self.pop_size}")
        print(f"  Поколений: {self.generations}")
        print(f"  Размер датасета: {len(self.dataset)} стоп")
        print(f"  Параметров: {len(OPTIMIZATION_PARAMS)} (THRESHOLD, MIN_AREA, CLOSE_SIZE)")
        print("=" * 60 + "\n")

        print("📊 Создание начальной популяции...")
        population = [self.create_individual() for _ in range(self.pop_size)]
        print(f"  ✅ Популяция создана: {len(population)} особей")

        for gen in range(self.generations):
            print(f"\n🔄 Поколение {gen + 1}/{self.generations}")

            try:
                scores = self.evaluate_population_parallel(population)
            except:
                scores = self.evaluate_population(population)

            sorted_indices = np.argsort(scores)[::-1]

            self.history.append({
                'generation': gen,
                'best_fitness': self.best_fitness,
                'avg_fitness': np.mean(scores),
            })

            print(f"  Лучший фитнес: {self.best_fitness:.2f}%")
            print(f"  Средний фитнес: {np.mean(scores):.2f}%")

            # АВТОСОХРАНЕНИЕ
            if self.best is not None:
                with open('best_grayscale.json', 'w') as f:
                    json.dump({
                        'generation': gen + 1,
                        'fitness': self.best_fitness,
                        'params': self.best
                    }, f, indent=2)
                print(f"  💾 Автосохранение: поколение {gen + 1}, фитнес {self.best_fitness:.2f}%")

            # Элита (20%)
            elite_size = max(1, int(self.pop_size * 0.2))
            elite = [population[i] for i in sorted_indices[:elite_size]]

            new_population = elite.copy()
            while len(new_population) < self.pop_size:
                p1 = random.choice(elite)
                p2 = random.choice(elite)
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                new_population.append(child)

            population = new_population

        print("\n" + "=" * 60)
        print("✅ ЭВОЛЮЦИЯ ЗАВЕРШЕНА")
        print("=" * 60)
        print(f"  Лучший фитнес: {self.best_fitness:.2f}%")
        print("\n  Лучшие параметры:")
        for key, value in self.best.items():
            print(f"    {key}: {value}")
        print("=" * 60)

        return self.best, self.history


# =============================================================================
# ОСНОВНАЯ ПРОГРАММА
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🧬 GRAYSCALE + ГЕНЕТИЧЕСКИЙ АЛГОРИТМ")
    print("=" * 60)

    # 1. Загрузка данных
    dataset = load_dataset()

    if len(dataset) < 10:
        print("\n❌ Слишком мало данных для обучения!")
        print("   Нужно минимум 10 размеченных стоп.")
        exit()

    # 2. Запуск ГА
    ga = GeneticAlgorithmGrayscale(
        dataset=dataset,
        pop_size=15,
        generations=25,
        mutation_rate=0.15
    )

    best_params, history = ga.evolve()

    # 3. Сохранение результатов
    results = {
        'best_params': best_params,
        'best_fitness': history[-1]['best_fitness'] if history else 0,
        'history': history,
        'dataset_size': len(dataset),
        'generations': len(history),
        'approach': 'grayscale'
    }

    with open('ga_grayscale_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n✅ Результаты сохранены в ga_grayscale_results.json")
    print("   📁 best_grayscale.json - автосохранение")

    # 4. Построение графика
    try:
        generations = [h['generation'] for h in history]
        best_fitness = [h['best_fitness'] for h in history]
        avg_fitness = [h['avg_fitness'] for h in history]
        plt.figure(figsize=(12, 6))
        plt.plot(generations, best_fitness, 'g-', linewidth=2, label='Лучший фитнес')
        plt.plot(generations, avg_fitness, 'b--', linewidth=2, label='Средний фитнес')
        plt.xlabel('Поколение')
        plt.ylabel('Фитнес (%)')
        plt.title('Эволюция ГА (Grayscale подход)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('../report/evolution_grayscale.png', dpi=150)
        plt.show()
        print("✅ График сохранен в evolution_grayscale.png")
    except:
        print("⚠️ График не построен")

    print("\n🏁 ГОТОВО!")
    print("   Лучшие параметры сохранены в best_grayscale.json")