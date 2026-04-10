import json
import multiprocessing as mp
import os
from functools import partial

from count_models_performance_subsampling import process_file_with_subsampling


def main():
    path = "../../data"
    save_path = "../results/raw_subsampling"

    os.makedirs(save_path, exist_ok=True)

    files = [f for f in os.listdir(path)
             if os.path.isfile(os.path.join(path, f)) and f.endswith('.csv')]

    if not files:
        print("CSV файлы не найдены в директории:", path)
        return

    print(f"Найдено {len(files)} файлов для обработки")
    print(f"Сохранение результатов в: {save_path}")

    num_processes = mp.cpu_count()
    if num_processes > 10:
        num_processes = 10
    print(f"Используется процессов: {num_processes}")

    with mp.Pool(processes=num_processes) as pool:
        process_func = partial(process_file_with_subsampling, path=path, save_path=save_path)

        results = []
        for i, file in enumerate(files):
            result = pool.apply_async(process_func, (file,))
            results.append(result)

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
