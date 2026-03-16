import pandas as pd
import json
import glob
import os
from pathlib import Path


def json_to_flat_df_auto(folder_path):
    """
    Использует pandas.json_normalize для автоматического разворачивания вложенных структур
    """
    all_data = []

    for json_file in Path(folder_path).glob('*.json'):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Добавляем имя файла как отдельную колонку
        data['name'] = json_file.name.split('.')[0]

        # Нормализуем JSON (разворачивает вложенные структуры)
        flat_data = pd.json_normalize(data)
        all_data.append(flat_data)

    # Объединяем все DataFrame
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()


# Использование
# df_auto = json_to_flat_df_auto('dc_problexity/results')
# df_auto.set_index(['name'], inplace=True)
# df_auto.to_csv('dc_problexity/dc_problexity.csv')
# print(df_auto.head())

# df_auto = json_to_flat_df_auto('dc_sequence_diversity/results')
# df_auto.set_index(['name'], inplace=True)
# df_auto.to_csv('dc_sequence_diversity/dc_sequence_diversity.csv')
# print(df_auto.head())


# df_auto = json_to_flat_df_auto('ot_result/fgw_par')
# df_auto.set_index(['name'], inplace=True)
# df_auto.to_csv('ot_result/dc_sequence_diversity.csv')
# print(df_auto.head())


def flatten_csv_files(folder_path):
    """
    Читает все CSV файлы из папки и создает плоский DataFrame
    с колонками вида "ModelName_MetricName"
    """
    all_data = []

    # Получаем все CSV файлы в папке
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    for file_path in csv_files:
        # Читаем CSV файл
        df = pd.read_csv(file_path)

        # Получаем имя файла без расширения для идентификации (опционально)
        file_name = Path(file_path).stem

        # Создаем словарь для текущего файла
        row_dict = {'name': file_name}

        # Для каждой строки (модели) создаем колонки с префиксом
        for _, row in df.iterrows():
            model_name = row['Model']
            # Очищаем имя модели для использования в названии колонки
            model_name_clean = model_name.replace(' ', '_').replace('-', '_')

            # Для каждой метрики создаем колонку
            for col in df.columns:
                if col != 'Model':  # пропускаем колонку с названием модели
                    new_col_name = f"{model_name_clean}_{col}"
                    row_dict[new_col_name] = row[col]

        all_data.append(row_dict)

    # Создаем DataFrame
    result_df = pd.DataFrame(all_data)

    return result_df


# Использование
df_flattened = flatten_csv_files('models/results')
df_flattened.to_csv('models/models_results.csv', index=False)
print(df_flattened.head())

