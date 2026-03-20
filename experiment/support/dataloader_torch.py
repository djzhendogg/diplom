import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from experiment.archive.preset_tools_sequant import data_processing


class PeptideDataset(Dataset):
    def __init__(self, batches, monomer_dict, max_len):
        """
        Dataset для пептидов

        Args:
            batches: список батчей, каждый батч - список кортежей (sequence, label)
            monomer_dict: словарь {аминокислота: SMILES}
            max_len: максимальная длина последовательности
        """
        self.batches = batches
        self.monomer_dict = monomer_dict
        self.max_len = max_len

        # Предварительно обрабатываем все данные
        self.processed_data = []
        self.targets = []  # переименовано в targets

        for batch in batches:
            processed_batch, labels = data_processing(batch, monomer_dict, max_len)
            # processed_batch shape: (batch_size, rows, max_len)
            self.processed_data.append(processed_batch)
            self.targets.extend(labels)

        # Объединяем все обработанные последовательности
        self.processed_data = np.vstack(self.processed_data) if len(self.processed_data) > 1 else self.processed_data[0]

        # Преобразуем targets в тензор
        self.targets = torch.LongTensor(self.targets)

        # Уникальные классы
        self.classes = torch.unique(self.targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # Преобразуем numpy массивы в тензоры PyTorch
        sequence = torch.FloatTensor(self.processed_data[idx])
        target = self.targets[idx]  # теперь это тензор

        # Добавляем канальное измерение (batch_size, channels, rows, max_len)
        sequence = sequence.unsqueeze(0)  # shape: (1, rows, max_len)

        return sequence, target

def create_dataloader_from_batches(batches, monomer_dict, max_len,
                                   batch_size=32, shuffle=True, num_workers=0):
    """
    Создает DataLoader из батчей для PyTorch

    Args:
        batches: список батчей, каждый батч - список кортежей (sequence, label)
        monomer_dict: словарь {аминокислота: SMILES}
        max_len: максимальная длина последовательности
        batch_size: размер батча для DataLoader
        shuffle: перемешивать ли данные
        num_workers: количество воркеров для загрузки данных

    Returns:
        torch.utils.data.dataloader.DataLoader
    """
    dataset = PeptideDataset(batches, monomer_dict, max_len)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False  # Не отбрасывать последний неполный батч
    )

    return dataloader


# Альтернативная версия с генератором для экономии памяти
class PeptideDatasetLazy(Dataset):
    def __init__(self, batches, monomer_dict, max_len):
        """
        Ленивая версия Dataset - обрабатывает данные на лету при обращении
        """
        self.batches = batches
        self.monomer_dict = monomer_dict
        self.max_len = max_len

        # Создаем плоский список всех samples с указанием батча и позиции в батче
        self.samples = []
        for batch_idx, batch in enumerate(batches):
            for sample_idx, (seq, label) in enumerate(batch):
                self.samples.append((batch_idx, sample_idx, seq, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        batch_idx, sample_idx, seq, label = self.samples[idx]
        batch = self.batches[batch_idx]

        # Обрабатываем конкретную последовательность
        processed_batch, _ = data_processing([batch[sample_idx]], self.monomer_dict, self.max_len)

        sequence = torch.FloatTensor(processed_batch[0])
        sequence = sequence.unsqueeze(0)  # Добавляем канальное измерение
        label = torch.LongTensor([label])[0]

        return sequence, label


def create_dataloader_lazy(batches, monomer_dict, max_len,
                           batch_size=32, shuffle=True, num_workers=0):
    """
    Создает ленивый DataLoader для больших наборов данных
    """
    dataset = PeptideDatasetLazy(batches, monomer_dict, max_len)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False
    )

    return dataloader


# Функция для обратной совместимости с оригинальным именем
def create_dataset_from_batches(batches, monomer_dict, max_len,
                                batch_size=32, shuffle=True, lazy=False):
    """
    Основная функция для создания DataLoader (аналог create_dataset_from_batches)

    Args:
        batches: список батчей
        monomer_dict: словарь мономеров
        max_len: максимальная длина
        batch_size: размер батча
        shuffle: перемешивать ли данные
        lazy: использовать ли ленивую загрузку (для больших данных)

    Returns:
        torch.utils.data.dataloader.DataLoader
    """
    if lazy:
        return create_dataloader_lazy(batches, monomer_dict, max_len, batch_size, shuffle)
    else:
        return create_dataloader_from_batches(batches, monomer_dict, max_len, batch_size, shuffle)