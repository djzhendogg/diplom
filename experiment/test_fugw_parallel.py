import json
import os
import pandas as pd
import multiprocessing as mp
from functools import partial

from fgw import FusedGromovWassersteinComputer
from preset_tools_difflen import encode_seqs


def process_file(file, path, save_path):
    """Обработка одного файла"""
    try:
        # Чтение и обработка файла
        df = pd.read_csv(os.path.join(path, file))
        df_0 = df[df['label'] == 0]
        df_1 = df[df['label'] == 1]

        X0 = encode_seqs(df_0['sequence'].to_list())
        X1 = encode_seqs(df_1['sequence'].to_list())

        # Создаем экземпляр компьютера для каждого процесса
        fgw_computer = FusedGromovWassersteinComputer()
        dist_res = fgw_computer.compute_fugw_with_search(X0, X1, verbose=True,
                                                         reg_marginals=[
                                                             10, 100,
                                                             (500, 3000),
                                                         ], alphas = [0.5, 0.7]
                                                         )
        dist_res['len_0'] = len(X0)
        dist_res['len_1'] = len(X1)

        # Сохранение результата
        name = file.split('.')[0]
        output_path = os.path.join(save_path, f'{name}.json')

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dist_res, f, ensure_ascii=False, indent=4)


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
    path = "./data"
    save_path = "./ot_result/fugw_par/"

    # Создание директории для сохранения, если её нет
    os.makedirs(save_path, exist_ok=True)

    # Получение списка файлов
    files = [f for f in os.listdir(path)
             if os.path.isfile(os.path.join(path, f)) and f.endswith('.csv')]
    files = ['amp_csamp.csv',
'amp_fernandes.csv',]
    if not files:
        print("CSV файлы не найдены в директории:", path)
        return

    print(f"Найдено {len(files)} файлов для обработки")
    print(f"Сохранение результатов в: {save_path}")

    # Настройка многопроцессорности
    num_processes = mp.cpu_count()  # Используем все доступные ядра
    if num_processes > 2:
        num_processes = 2
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
