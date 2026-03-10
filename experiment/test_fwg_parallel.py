import json
import os
import pandas as pd
import multiprocessing as mp
import time
from functools import partial

from fgw import FusedGromovWassersteinComputer
from preset_tools_difflen import encode_seqs


def process_file(file, path, save_path):
    """Обработка одного файла"""
    try:
        start_time = time.time()

        # Чтение и обработка файла
        df = pd.read_csv(os.path.join(path, file))
        df_0 = df[df['label'] == 0][:10]
        df_1 = df[df['label'] == 1][:10]

        X0 = encode_seqs(df_0['sequence'].to_list())
        X1 = encode_seqs(df_1['sequence'].to_list())

        # Создаем экземпляр компьютера для каждого процесса
        fgw_computer = FusedGromovWassersteinComputer()
        dist_res = fgw_computer.compute_fgw_with_search(X0, X1)
        dist_res['len_0'] = len(X0)
        dist_res['len_1'] = len(X1)

        # Сохранение результата
        name = file.split('.')[0]
        output_path = os.path.join(save_path, f'{name}.json')

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dist_res, f, ensure_ascii=False, indent=4)

        elapsed_time = time.time() - start_time

        return {
            'file': file,
            'status': 'success',
            'time': elapsed_time,
            'message': f'Обработан за {elapsed_time:.2f} сек'
        }

    except Exception as e:
        return {
            'file': file,
            'status': 'error',
            'message': str(e),
            'time': time.time() - start_time
        }


def main():
    # Настройка путей
    path = "./data"
    save_path = "./ot_result/fgw/"

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
    if num_processes > 50:
        num_processes = 50
    print(f"Используется процессов: {num_processes}")

    # Глобальный таймер
    start_global_time = time.time()

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

    # Финальный отчет
    elapsed_global = time.time() - start_global_time

    print("\n" + "=" * 60)
    print("ОБРАБОТКА ЗАВЕРШЕНА")
    print("=" * 60)
    print(f"Всего файлов: {len(files)}")
    print(f"Общее время: {elapsed_global:.2f} сек ({elapsed_global / 60:.2f} мин)")

    # Статистика по ошибкам
    errors = [r for r in final_results if r['status'] == 'error']
    if errors:
        print(f"\nОшибки при обработке ({len(errors)}):")
        for error in errors:
            print(f"  - {error['file']}: {error['message']}")

    # Статистика по времени
    successful = [r for r in final_results if r['status'] == 'success']
    if successful:
        times = [r['time'] for r in successful]
        print(f"\nСтатистика по времени (успешные файлы):")
        print(f"  Минимальное: {min(times):.2f} сек")
        print(f"  Максимальное: {max(times):.2f} сек")
        print(f"  Среднее: {sum(times) / len(times):.2f} сек")
        print(f"  Медианное: {sorted(times)[len(times) // 2]:.2f} сек")


if __name__ == "__main__":
    main()