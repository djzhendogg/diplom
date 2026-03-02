import torch
import numpy as np
from geomloss import SamplesLoss
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class OTDDComputer:
    """
    Класс для вычисления OT-расстояния между двумя классами данных
    с использованием энтропийной регуляризации (GeomLoss)
    """

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Используется устройство: {device}")

    def prepare_data(self, X_class0, X_class1, use_pca=True, n_components=50,
                     scale_data=True, batch_size=None):
        """
        Подготовка данных для OT-расстояния

        Args:
            X_class0, X_class1: numpy массивы формы (n_samples, 46, 100)
            use_pca: применять PCA для снижения размерности
            n_components: количество компонент PCA
            scale_data: стандартизировать данные
            batch_size: если указан, возвращает генератор для батчей
        """
        # Выпрямляем данные из 46x100 в векторы длины 4600
        X0 = X_class0.reshape(X_class0.shape[0], -1)
        X1 = X_class1.reshape(X_class1.shape[0], -1)

        print(f"Исходная размерность: {X0.shape[1]}")

        if scale_data:
            # Стандартизация важна для OT, чтобы все признаки были в одном масштабе
            scaler = StandardScaler()
            X0 = scaler.fit_transform(X0)
            X1 = scaler.transform(X1)
            print("Данные стандартизированы")

        if use_pca and n_components < X0.shape[1]:
            # PCA для снижения размерности
            pca = PCA(n_components=n_components)
            X0 = pca.fit_transform(X0)
            X1 = pca.transform(X1)
            print(f"Размерность после PCA: {X0.shape[1]}")
            print(f"Объясненная дисперсия: {pca.explained_variance_ratio_.sum():.3f}")

        # Конвертируем в тензоры и переносим на устройство
        X0_tensor = torch.from_numpy(X0).float().to(self.device)
        X1_tensor = torch.from_numpy(X1).float().to(self.device)

        # Равномерные веса для всех семплов
        weights0 = torch.ones(len(X0_tensor), device=self.device) / len(X0_tensor)
        weights1 = torch.ones(len(X1_tensor), device=self.device) / len(X1_tensor)

        if batch_size:
            # Возвращаем генераторы для батчевой обработки
            return (self._batch_generator(X0_tensor, weights0, batch_size),
                    self._batch_generator(X1_tensor, weights1, batch_size))
        else:
            return (weights0, X0_tensor), (weights1, X1_tensor)

    def _batch_generator(self, X, weights, batch_size):
        """Генератор для батчевой подачи данных"""
        n_samples = len(X)
        for i in range(0, n_samples, batch_size):
            batch_X = X[i:i + batch_size]
            batch_weights = weights[i:i + batch_size]
            # Нормируем веса в батче
            batch_weights = batch_weights / batch_weights.sum()
            yield batch_weights, batch_X

    def compute_distance(self, data0, data1, blur=0.05, scaling=0.8,
                         debias=True, potentials=False, **kwargs):
        """
        Вычисляет OT-расстояние с энтропийной регуляризацией

        Args:
            data0, data1: кортежи (веса, признаки) или генераторы
            blur: сила энтропийной регуляризации (чем больше, тем сильнее регуляризация)
                  Обычно от 0.001 до 0.1. Для зашумленных данных можно увеличить
            scaling: скорость "охлаждения" при мультимасштабном вычислении (0.5-0.9)
            debias: использовать дебайased Sinkhorn (убирает смещение)
            potentials: вернуть также потенциалы (для градиентов)
        """
        # Создаем функцию потерь с энтропийной регуляризацией
        loss = SamplesLoss(
            loss="sinkhorn",  # Используем Sinkhorn с энтропийной регуляризацией
            blur=blur,  # Параметр регуляризации (аналог epsilon)
            scaling=scaling,  # Мультимасштабный параметр
            debias=debias,  # Убираем смещение
            potentials=potentials,
            **kwargs
        )

        # Вычисляем расстояние
        if isinstance(data0, tuple):
            # Прямое вычисление на полных данных
            weights0, X0 = data0
            weights1, X1 = data1
            distance = loss(weights0, X0, weights1, X1)
        else:
            # Батчевое вычисление
            distance = loss(data0, data1)

        return distance.item()

    def compute_with_parameter_sweep(self, X_class0, X_class1,
                                     blur_values=[0.001, 0.01, 0.05, 0.1, 0.5],
                                     **prepare_kwargs):
        """
        Вычисляет расстояние при разных значениях регуляризации
        Полезно для выбора оптимального blur
        """
        # Подготавливаем данные один раз
        data0, data1 = self.prepare_data(X_class0, X_class1, **prepare_kwargs)

        results = {}
        for blur in blur_values:
            dist = self.compute_distance(data0, data1, blur=blur)
            results[blur] = dist
            print(f"blur={blur:.3f}: расстояние={dist:.4f}")

        # Визуализация
        plt.figure(figsize=(8, 5))
        plt.plot(list(results.keys()), list(results.values()), 'bo-')
        plt.xscale('log')
        plt.xlabel('Параметр регуляризации blur')
        plt.ylabel('OT-расстояние')
        plt.title('Зависимость OT-расстояния от регуляризации')
        plt.grid(True)
        plt.show()

        return results


# ============================================================
# Пример использования
# ============================================================

def create_sample_data(n_samples0=1000, n_samples1=800):
    """Создает синтетические данные для примера"""
    np.random.seed(42)

    # Класс 0: матрицы 46x100 с нормальным распределением
    X0 = np.random.randn(n_samples0, 46, 100) * 0.5

    # Класс 1: немного другое распределение
    X1 = np.random.randn(n_samples1, 46, 100) * 0.7 + 0.2

    # Добавляем некоторую структуру (корреляции между признаками)
    for i in range(min(n_samples0, n_samples1)):
        X0[i] += 0.3 * np.sin(np.linspace(0, 4 * np.pi, 100)).reshape(1, -1)
        X1[i] += 0.5 * np.cos(np.linspace(0, 4 * np.pi, 100)).reshape(1, -1)

    return X0, X1


# Создаем тестовые данные
print("Создание тестовых данных...")
X_class0, X_class1 = create_sample_data(n_samples0=500, n_samples1=600)

# Инициализируем вычислитель
ot_computer = OTDDComputer()

# ============================================================
# Вариант 1: Быстрое вычисление с PCA и регуляризацией
# ============================================================
print("\n" + "=" * 50)
print("Вариант 1: PCA + энтропийная регуляризация")
print("=" * 50)

data0, data1 = ot_computer.prepare_data(
    X_class0, X_class1,
    use_pca=True,  # Используем PCA
    n_components=50,  # Снижаем до 50 компонент
    scale_data=True,  # Стандартизируем
    batch_size=None  # Обрабатываем всё сразу
)

# Пробуем разные значения регуляризации
for blur in [0.001, 0.01, 0.05, 0.1]:
    dist = ot_computer.compute_distance(
        data0, data1,
        blur=blur,  # Параметр энтропийной регуляризации
        scaling=0.8,  # Мультимасштабный параметр
        debias=True  # Убираем смещение
    )
    print(f"blur={blur:.3f}: OTDD = {dist:.4f}")

# ============================================================
# Вариант 2: Точное вычисление без PCA (может быть медленно)
# ============================================================
print("\n" + "=" * 50)
print("Вариант 2: Без PCA, сильная регуляризация")
print("=" * 50)

data0_full, data1_full = ot_computer.prepare_data(
    X_class0, X_class1,
    use_pca=False,  # Без PCA - полная размерность 4600
    scale_data=True
)

dist_full = ot_computer.compute_distance(
    data0_full, data1_full,
    blur=0.1,  # Сильная регуляризация для стабильности
    scaling=0.9
)
print(f"OTDD (полная размерность) = {dist_full:.4f}")

# ============================================================
# Вариант 3: Для очень больших данных (батчевая обработка)
# ============================================================
print("\n" + "=" * 50)
print("Вариант 3: Батчевая обработка для больших данных")
print("=" * 50)

gen0, gen1 = ot_computer.prepare_data(
    X_class0, X_class1,
    use_pca=True,
    n_components=30,
    batch_size=100  # Размер батча
)

dist_batch = ot_computer.compute_distance(
    gen0, gen1,
    blur=0.05,
    scaling=0.7
)
print(f"OTDD (батчево) = {dist_batch:.4f}")

# ============================================================
# Вариант 4: Подбор оптимального blur
# ============================================================
print("\n" + "=" * 50)
print("Вариант 4: Подбор параметра регуляризации")
print("=" * 50)

results = ot_computer.compute_with_parameter_sweep(
    X_class0, X_class1,
    blur_values=[0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5],
    use_pca=True,
    n_components=40
)

# ============================================================
# Дополнительно: Визуализация распределений (первые 2 PCA компоненты)
# ============================================================
print("\n" + "=" * 50)
print("Визуализация данных в пространстве первых двух PCA компонент")
print("=" * 50)

# Получаем данные для визуализации
X0_vis = X_class0.reshape(X_class0.shape[0], -1)
X1_vis = X_class1.reshape(X_class1.shape[0], -1)

# Стандартизация
scaler = StandardScaler()
X0_scaled = scaler.fit_transform(X0_vis)
X1_scaled = scaler.transform(X1_vis)

# PCA для визуализации
pca = PCA(n_components=2)
X0_pca = pca.fit_transform(X0_scaled)
X1_pca = pca.transform(X1_scaled)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(X0_pca[:, 0], X0_pca[:, 1], alpha=0.5, label='Класс 0', s=10)
plt.scatter(X1_pca[:, 0], X1_pca[:, 1], alpha=0.5, label='Класс 1', s=10)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Распределение классов в PCA пространстве')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# Показываем разницу в распределениях
plt.hist(X0_pca[:, 0], bins=30, alpha=0.5, label='Класс 0', density=True)
plt.hist(X1_pca[:, 0], bins=30, alpha=0.5, label='Класс 1', density=True)
plt.xlabel('PC1')
plt.ylabel('Плотность')
plt.title('Распределение по первой компоненте')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()