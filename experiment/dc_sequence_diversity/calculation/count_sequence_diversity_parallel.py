import json
import multiprocessing as mp
import os
from functools import partial

import numpy as np
import pandas as pd

from count_sequence_diversity import general_characterize


def process_file(file, path, save_path):
    try:
        name = file.split('.')[0]
        df = pd.read_csv(os.path.join(path, file))
        report = general_characterize(df, name)
        out_path = os.path.join(save_path, name + '.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=4)

        return {
            'file': file,
            'status': 'success'
        }

    except Exception as e:
        return {
            'file': file,
            'status': 'error',
            'message': str(e),
        }


def process_file_with_subsampling(file, path, save_path, n_runs=100, frac=0.8):
    try:
        name = file.split('.')[0]
        df = pd.read_csv(os.path.join(path, file))

        reports = []

        for i in range(n_runs):
            sample_df = df.sample(frac=frac, replace=False, random_state=i)
            report = general_characterize(sample_df, name)
            reports.append(report)

        # --- агрегирование ---
        def aggregate(values):
            return {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": [float(v) for v in values]
            }

        result = {
            "name": name,
            "n_runs": n_runs,
            "sample_fraction": frac,
            "samples_num": int(df.shape[0])
        }

        # берём все ключи кроме name и samples_num
        keys = [k for k in reports[0].keys() if k not in ['name', 'samples_num']]

        for key in keys:
            values = [r[key] for r in reports]
            result[key] = aggregate(values)

        # сохранить
        out_path = os.path.join(save_path, name + '_subsampled.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

        return {
            'file': file,
            'status': 'success'
        }

    except Exception as e:
        return {
            'file': file,
            'status': 'error',
            'message': str(e),
        }


def main():
    # Настройка путей
    path = "../../data"
    save_path = "../results/raw_subsampling"

    # Создание директории для сохранения, если её нет
    os.makedirs(save_path, exist_ok=True)

    # Получение списка файлов
    files = [f for f in os.listdir(path)
             if os.path.isfile(os.path.join(path, f)) and f.endswith('.csv')]

    if not files:
        print("CSV файлы не найдены в директории:", path)
        return

    print(f"Найдено {len(files)} файлов для обработки")
    print(f"Сохранение результатов в: {save_path}")

    # Настройка многопроцессорности
    num_processes = mp.cpu_count()  # Используем все доступные ядра
    if num_processes > 20:
        num_processes = 20
    print(f"Используется процессов: {num_processes}")

    # Создание пула процессов
    with mp.Pool(processes=num_processes) as pool:
        # Частичное применение функции с фиксированными аргументами
        process_func = partial(process_file_with_subsampling, path=path, save_path=save_path)

        # Асинхронный запуск обработки
        results = []
        for i, file in enumerate(files):
            result = pool.apply_async(process_func, (file,))
            results.append(result)

        # Сбор всех результатов
        print("\nОжидание завершения всех процессов...")
        final_results = [r.get() for r in results]

    # Статистика по ошибкам
    errors = [r for r in final_results if r['status'] == 'error']

    if errors:
        print(f"\nОшибки при обработке ({len(errors)}):")
        for error in errors:
            print(f"  - {error['file']}: {error['message']}")
        errors_file = os.path.join(save_path, f'errors.json')
        with open(errors_file, 'w', encoding='utf-8') as f:
            json.dump({
                'total_errors': len(errors),
                'errors': errors
            }, f, ensure_ascii=False, indent=4)
        print(f"\nОшибки сохранены в файл: {errors_file}")
        for error in errors:
            print(f"  - {error['file']}: {error['message']}")
    else:
        print("\nОшибок не обнаружено!")


if __name__ == "__main__":
    main()
    # file = 'amp_gonzales.csv'
    # path = "../../data"
    # save_path = "../results/raw_subsampling"
    #
    # process_file_with_subsampling(file, path, save_path, n_runs=2)