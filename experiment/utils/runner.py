import os
import json
import multiprocessing as mp
from functools import partial


def run_processing(
    path,
    save_path,
    process_func,
    max_processes=10,
    process_kwargs=None
):
    """
    Универсальный раннер для обработки CSV файлов.

    :param path: путь к папке с CSV
    :param save_path: куда сохранять результаты
    :param process_func: функция обработки одного файла
    :param max_processes: максимум процессов
    :param process_kwargs: дополнительные аргументы для process_func
    """

    if process_kwargs is None:
        process_kwargs = {}

    os.makedirs(save_path, exist_ok=True)

    files = [
        f for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f)) and f.endswith('.csv')
    ]

    if not files:
        print("CSV файлы не найдены в директории:", path)
        return

    print(f"Найдено {len(files)} файлов для обработки")
    print(f"Сохранение результатов в: {save_path}")

    num_processes = min(mp.cpu_count(), max_processes)
    print(f"Используется процессов: {num_processes}")

    with mp.Pool(processes=num_processes) as pool:
        process_func_wrapped = partial(
            process_func,
            path=path,
            save_path=save_path,
            **process_kwargs
        )

        results = [
            pool.apply_async(process_func_wrapped, (file,))
            for file in files
        ]

        print("\nОжидание завершения всех процессов...")
        final_results = [r.get() for r in results]

    _handle_errors(final_results, save_path)


def _handle_errors(results, save_path):
    errors = [r for r in results if r.get('status') == 'error']

    if errors:
        print(f"\nОшибки при обработке ({len(errors)}):")
        for error in errors:
            print(f"  - {error['file']}: {error['message']}")

        errors_file = os.path.join(save_path, 'errors.json')
        with open(errors_file, 'w', encoding='utf-8') as f:
            json.dump({
                'total_errors': len(errors),
                'errors': errors
            }, f, ensure_ascii=False, indent=4)

        print(f"\nОшибки сохранены в файл: {errors_file}")
    else:
        print("\nОшибок не обнаружено!")
