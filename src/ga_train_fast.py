"""
ga_train_fast.py - БЫСТРАЯ версия обучения генетического алгоритма
С оптимизациями: кэширование, параллельная обработка, уменьшенная популяция
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
    get_plantogram_mask,
    get_largest_contour,
    find_AB_points_final,
    calculate_strieter_index_full,
    draw_contours,
    FOOT_TYPE_BOUNDARIES
)

from cached_functions import get_foot_mask_cached as get_foot_mask

# =============================================================================
# ПАРАМЕТРЫ ДЛЯ ОПТИМИЗАЦИИ
# =============================================================================

OPTIMIZATION_PARAMS = {
    'H_LOW': {'min': 60, 'max': 90, 'type': 'int', 'default': 74},
    'H_HIGH': {'min': 75, 'max': 110, 'type': 'int', 'default': 86},
    'S_LOW': {'min': 100, 'max': 180, 'type': 'int', 'default': 149},
    'S_HIGH': {'min': 200, 'max': 255, 'type': 'int', 'default': 255},
    'V_LOW': {'min': 100, 'max': 180, 'type': 'int', 'default': 141},
    'V_HIGH': {'min': 180, 'max': 255, 'type': 'int', 'default': 194},
    'MIN_AREA': {'min': 20000, 'max': 100000, 'type': 'int', 'default': 50000},
    'TOE_RATIO': {'min': 0.15, 'max': 0.45, 'type': 'float', 'default': 0.3},
    'HEEL_RATIO': {'min': 0.50, 'max': 0.80, 'type': 'float', 'default': 0.65},
    'PERP_TOLERANCE': {'min': 5, 'max': 30, 'type': 'int', 'default': 10},
    'PERP_TOLERANCE_FALLBACK': {'min': 10, 'max': 50, 'type': 'int', 'default': 20},
}


# =============================================================================
# ЗАГРУЗКА ДАННЫХ
# =============================================================================

def load_dataset():
    """Загружает размеченный датасет из progress.json"""
    print("\n📂 ЗАГРУЗКА ДАННЫХ")
    print("=" * 50)

    if not os.path.exists('progress.json'):
        print("❌ Файл progress.json не найден!")
        return []

    with open('progress.json', 'r') as f:
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
            'params_used': data.get('params', {})
        })

    print(f"  ✅ Датасет готов: {len(dataset)} стоп")
    print("=" * 50)

    return dataset


# =============================================================================
# ФИТНЕС-ФУНКЦИЯ
# =============================================================================

def calculate_index_with_params(image, params):
    """Рассчитывает индекс Штриттера с заданными параметрами."""
    try:
        foot_mask = get_foot_mask(image)
        plant_mask = get_plantogram_mask(image, foot_mask, params)
        plant_contour = get_largest_contour(plant_mask)

        if plant_contour is None:
            return None

        A, B = find_AB_points_final(plant_contour, params)
        index, V, G, D, foot_type = calculate_strieter_index_full(
            plant_contour, A, B, None, params
        )

        return index
    except:
        return None


def fitness_function_single(args):
    """Фитнес-функция для одного набора параметров (для параллельной обработки)"""
    params, dataset = args
    errors = []
    valid_count = 0

    for item in dataset:
        calculated = calculate_index_with_params(item['image'], params)
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
# ГЕНЕТИЧЕСКИЙ АЛГОРИТМ (ОПТИМИЗИРОВАННЫЙ)
# =============================================================================

class GeneticAlgorithmFast:
    def __init__(self, dataset, pop_size=30, generations=30, mutation_rate=0.2):
        self.dataset = dataset
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.best = None
        self.best_fitness = -float('inf')
        self.history = []
        self.current_generation = 0

    def create_individual(self):
        individual = {}
        for key, param in OPTIMIZATION_PARAMS.items():
            if param['type'] == 'int':
                individual[key] = random.randint(param['min'], param['max'])
            else:
                individual[key] = random.uniform(param['min'], param['max'])
        return individual

    def mutate(self, individual):
        mutated = deepcopy(individual)
        for key, param in OPTIMIZATION_PARAMS.items():
            if random.random() < self.mutation_rate:
                if param['type'] == 'int':
                    step = max(1, (param['max'] - param['min']) // 10)
                    mutated[key] += random.randint(-step, step)
                    mutated[key] = max(param['min'], min(param['max'], mutated[key]))
                else:
                    step = (param['max'] - param['min']) * 0.1
                    mutated[key] += random.uniform(-step, step)
                    mutated[key] = max(param['min'], min(param['max'], mutated[key]))

        if mutated['H_LOW'] > mutated['H_HIGH']:
            mutated['H_LOW'], mutated['H_HIGH'] = mutated['H_HIGH'], mutated['H_LOW']
        if mutated['S_LOW'] > mutated['S_HIGH']:
            mutated['S_LOW'], mutated['S_HIGH'] = mutated['S_HIGH'], mutated['S_LOW']
        if mutated['V_LOW'] > mutated['V_HIGH']:
            mutated['V_LOW'], mutated['V_HIGH'] = mutated['V_HIGH'], mutated['V_LOW']

        return mutated

    def crossover(self, p1, p2):
        child = {}
        for key in p1:
            child[key] = p1[key] if random.random() < 0.5 else p2[key]
        return child

    def evaluate_population_parallel(self, population):
        """Параллельная оценка популяции"""
        args = [(ind, self.dataset) for ind in population]

        print(f"    Оценка {len(population)} особей на 6 ядрах...")

        with Pool(processes=6) as pool:
            scores = pool.map(fitness_function_single, args)

        for i, score in enumerate(scores):
            if score > self.best_fitness:
                self.best_fitness = score
                self.best = deepcopy(population[i])

        return scores

    def evolve(self):
        print("\n🧬 ЗАПУСК БЫСТРОГО ГЕНЕТИЧЕСКОГО АЛГОРИТМА")
        print("=" * 60)
        print(f"  Размер популяции: {self.pop_size}")
        print(f"  Поколений: {self.generations}")
        print(f"  Размер датасета: {len(self.dataset)} стоп")
        print(f"  Ядер CPU: {cpu_count()}")
        print("=" * 60 + "\n")

        print("📊 Создание начальной популяции...")
        population = [self.create_individual() for _ in range(self.pop_size)]
        print(f"  ✅ Популяция создана: {len(population)} особей")

        for gen in range(self.generations):
            self.current_generation = gen
            print(f"\n🔄 Поколение {gen + 1}/{self.generations}")

            scores = self.evaluate_population_parallel(population)

            sorted_indices = np.argsort(scores)[::-1]

            self.history.append({
                'generation': gen,
                'best_fitness': self.best_fitness,
                'avg_fitness': np.mean(scores),
                'best_individual': self.best
            })

            print(f"  Лучший фитнес: {self.best_fitness:.2f}%")
            print(f"  Средний фитнес: {np.mean(scores):.2f}%")

            # ============================================================
            # АВТОСОХРАНЕНИЕ
            # ============================================================
            if self.best is not None:
                with open('best_autosave.json', 'w') as f:
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
    print("🧬 БЫСТРЫЙ ГЕНЕТИЧЕСКИЙ АЛГОРИТМ - ОБУЧЕНИЕ")
    print("=" * 60)

    dataset = load_dataset()

    if len(dataset) < 10:
        print("\n❌ Слишком мало данных для обучения!")
        exit()

    # ================================================================
    # ПРАВИЛЬНО: используем значения из класса (30, 30, 0.2)
    # ================================================================
    # ================================================================
    # НОВЫЕ АГРЕССИВНЫЕ НАСТРОЙКИ
    # ================================================================
    ga = GeneticAlgorithmFast(
        dataset=dataset,
        pop_size=50,  # ← увеличили
        generations=50,  # ← увеличили
        mutation_rate=0.3  # ← увеличили
    )

    best_params, history = ga.evolve()

    # Сохранение результатов
    results = {
        'best_params': best_params,
        'best_fitness': history[-1]['best_fitness'] if history else 0,
        'history': history,
        'dataset_size': len(dataset),
        'generations': len(history),
    }

    with open('ga_results_fast.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n✅ Результаты сохранены в ga_results_fast.json")
    print("   📁 best_autosave.json - автосохранение лучшего результата")

    # Построение графика
    try:
        generations = [h['generation'] for h in history]
        best_fitness = [h['best_fitness'] for h in history]
        avg_fitness = [h['avg_fitness'] for h in history]

        plt.figure(figsize=(12, 6))
        plt.plot(generations, best_fitness, 'g-', linewidth=2, label='Лучший фитнес')
        plt.plot(generations, avg_fitness, 'b--', linewidth=2, label='Средний фитнес')
        plt.xlabel('Поколение')
        plt.ylabel('Фитнес (%)')
        plt.title('Эволюция генетического алгоритма (быстрая версия)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('evolution_history_fast.png', dpi=150)
        plt.show()
        print("✅ График сохранен в evolution_history_fast.png")
    except:
        print("⚠️ График не построен")