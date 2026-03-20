import ot
import torch
import numpy as np
from sklearn.decomposition import PCA


def compute_ot_distance_pot(X_class0, X_class1, use_pca=True, n_components=50):
    """
    Вычисляет OT-расстояние между двумя классами

    Args:
        X_class0, X_class1: данные формы (n_samples, 46, 100)
        use_pca: применять ли PCA для снижения размерности
        n_components: количество компонент PCA
    """
    # Выпрямляем данные
    X0 = X_class0.reshape(X_class0.shape[0], -1)
    X1 = X_class1.reshape(X_class1.shape[0], -1)

    if use_pca:
        # Опционально: снижаем размерность для скорости и стабильности
        pca = PCA(n_components=min(n_components, min(X0.shape[1], X0.shape[0], X1.shape[0])))
        X0_pca = pca.fit_transform(X0)
        X1_pca = pca.transform(X1)
        print(f"Снизили размерность с {X0.shape[1]} до {X0_pca.shape[1]}")
        X0, X1 = X0_pca, X1_pca

    # Вычисляем матрицу попарных расстояний
    # Можно использовать euclidean, sqeuclidean, cosine и т.д.
    M = ot.dist(X0, X1, metric='sqeuclidean')
    M /= M.max()  # Нормализация для стабильности

    # Равномерные веса для всех семплов
    a = np.ones(len(X0)) / len(X0)
    b = np.ones(len(X1)) / len(X1)

    # Точный OT (можно заменить на entropic regularization для скорости)
    distance = ot.emd2(a, b, M)

    return distance


# Использование
dist_exact = compute_ot_distance_pot(X_class0, X_class1, use_pca=True)
print(f"Точное OT-расстояние: {dist_exact:.4f}")


# Для больших данных - энтропийный OT (быстрее)
def compute_sinkhorn_distance_pot(X0, X1, reg=0.01):
    """Энтропийный OT (Sinkhorn) - быстрее на больших данных"""
    M = ot.dist(X0, X1, metric='sqeuclidean')
    M /= M.max()
    a = np.ones(len(X0)) / len(X0)
    b = np.ones(len(X1)) / len(X1)

    distance = ot.sinkhorn2(a, b, M, reg)
    return distance


dist_sinkhorn = compute_sinkhorn_distance_pot(X0_flat.numpy(), X1_flat.numpy())
print(f"Sinkhorn расстояние: {dist_sinkhorn:.4f}")