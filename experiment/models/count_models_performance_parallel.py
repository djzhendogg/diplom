import json
import os
import multiprocessing as mp
from functools import partial

from count_models_performance import process_file


def main():
    # Настройка путей
    path = "../data"
    save_path = "./results/"

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
    if num_processes > 10:
        num_processes = 10
    print(f"Используется процессов: {num_processes}")

    # Создание пула процессов
    with mp.Pool(processes=num_processes) as pool:
        # Частичное применение функции с фиксированными аргументами
        process_func = partial(process_file, path=path, save_path=save_path)

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