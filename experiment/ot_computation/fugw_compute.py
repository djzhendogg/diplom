import json
import multiprocessing as mp
import os
from functools import partial

import pandas as pd

from core import OTGridSearch, encode_sequences, get_config, FUGWSearchConfig


def process_file(file, path, save_path, descriptors, config_path):
    """Обработка одного файла"""
    try:
        # Чтение и обработка файла
        df = pd.read_csv(os.path.join(path, file))
        df_0 = df[df['label'] == 0]
        df_1 = df[df['label'] == 1]

        X0 = encode_sequences(df_0['sequence'].to_list(), descriptors)
        X1 = encode_sequences(df_1['sequence'].to_list(), descriptors)

        searcher = OTGridSearch()
        config = get_config(config_path, FUGWSearchConfig)
        dist_res = searcher.fugw_search(X0, X1, config)
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
    path = "../data"
    save_path = "./results/fugw"
    config_path = "./configs/search/fugw.yaml"

    descriptors_path = "./descriptors/aa_descriptors_scaled.csv"
    descriptors = pd.read_csv(descriptors_path, index_col=0)

    files = ['antioxidant.csv', 'hiv_bevirimat.csv', 'cpp_mlcpp.csv', 'cpp_cellppdmod.csv',
             'HemoPI3.csv', 'bce_ibce.csv', 'amp_antibp2.csv', 'avp_amppred.csv', 'antioxidant_UniDL4BioPep.csv',
             'anitinflamatory.csv', 'antifungal.csv', 'amp_modlamp.csv', 'toxic.csv', 'afp_antifp.csv',
             'amp_iamp2l.csv', 'ace_inhibitory_activity.csv', 'antiviral.csv', 'nep_neuropipred.csv', 'toxic_PLPTP.csv',
             'Hemo_HAPPENN.csv', 'afp_amppred.csv', 'cpp_cellppd.csv',
             'anticancer_alternative.csv', 'neuropeptide.csv', 'aip_antiinflam.csv', 'anticancer_main.csv',
             'pip_pipel.csv', 'hem_hemopi.csv', 'antimicrobial.csv', 'blood_brain_barrier.csv', 'quorum_sensing.csv',
             'antimicrobial_UniDL4BioPep.csv', 'antibacterial.csv']

    # Создание директории для сохранения, если её нет
    os.makedirs(save_path, exist_ok=True)

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
        process_func = partial(process_file, path=path, save_path=save_path, descriptors=descriptors,
                               config_path=config_path)

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
