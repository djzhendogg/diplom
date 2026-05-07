import json
import multiprocessing as mp
import os
from functools import partial

import numpy as np
import pandas as pd

from count_sequence_diversity import general_characterize


def process_file_with_subsampling(file, path, save_path, n_runs=100, frac=0.8):
    try:
        name = file.split('.')[0]
        df = pd.read_csv(os.path.join(path, file))

        reports = []

        for i in range(n_runs):
            sample_df = df.sample(frac=frac, replace=False, random_state=i)
            report = general_characterize(sample_df, name)
            reports.append(report)

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

        keys = [k for k in reports[0].keys() if k not in ['name', 'samples_num']]

        for key in keys:
            values = [r[key] for r in reports]
            result[key] = aggregate(values)

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
    path = "../../../data"
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
    if num_processes > 20:
        num_processes = 20
    print(f"Используется процессов: {num_processes}")

    with mp.Pool(processes=num_processes) as pool:
        process_func = partial(process_file_with_subsampling, path=path, save_path=save_path)

        results = []
        for i, file in enumerate(files):
            result = pool.apply_async(process_func, (file,))
            results.append(result)

        print("\nОжидание завершения всех процессов...")
        final_results = [r.get() for r in results]

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
