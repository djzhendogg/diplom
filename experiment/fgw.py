import ot
import numpy as np
import torch
import matplotlib.pyplot as plt

class FusedGromovWassersteinComputer:
    """
    Класс для вычисления Fused Gromov-Wasserstein расстояния между наборами матриц
    """

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def prepare_structures(self, X, structure_type='euclidean', **kwargs):
        """
        Создает структурные матрицы (C) для Gromov-Wasserstein

        Args:
            X: входные данные (n_samples, m)
            structure_type: тип структуры ('braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 
            'euclidean', 'hamming', 'jaccard', 'kulczynski1', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 
            'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule')

        Returns:
            C: структурная матрица (n_samples, n_samples) или список матриц
        """
        # Вычисляем попарные расстояния между образцами
        # Используем субдискретизацию для уменьшения размера
        n_samples = X.shape[0]
        if n_samples > 5000:
            # Для больших данных используем batch processing
            C = np.zeros((n_samples, n_samples))
            batch_size = 1000
            for i in range(0, n_samples, batch_size):
                end_i = min(i + batch_size, n_samples)
                for j in range(0, n_samples, batch_size):
                    end_j = min(j + batch_size, n_samples)
                    # Вычисляем расстояния для батча
                    batch_dist = ot.dist(
                        X[i:end_i],
                        X[j:end_j],
                        metric=structure_type
                    )
                    C[i:end_i, j:end_j] = batch_dist
        else:
            C = ot.dist(X, X, metric=structure_type)

        C /= C.max()
        return C

    def compute_fgw_distance(self, X0, X1, alpha=0.5, reg=0.01,
                             structure_type='euclidean', loss_fun='square_loss',
                             max_iter=100, tol=1e-6, verbose=True):
        """
        Вычисляет Fused Gromov-Wasserstein расстояние между двумя наборами матриц

        Args:
            X0, X1: данные формы (n_samples, m)
            alpha: баланс между признаками (1-alpha для структуры, alpha для признаков)
                   alpha=0: только структура (GW)
                   alpha=1: только признаки (OT)
            reg: энтропийная регуляризация
            structure_type: тип структуры ('braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 
            'euclidean', 'hamming', 'jaccard', 'kulczynski1', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 
            'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule')
            loss_fun: функция потерь для GW ('square_loss', 'kl_loss')
            structure0, structure1: предвычисленные структуры (если есть)
        """
        # Матрица признаковых расстояний (для Fused term)
        M = ot.dist(X0, X1, metric='sqeuclidean')
        M /= M.max()

        # Веса объектов (равномерные)
        n0, n1 = len(X0), len(X1)
        p = np.ones(n0) / n0
        q = np.ones(n1) / n1

        # Получаем структурные матрицы
        C0 = self.prepare_structures(X0, structure_type)
        C1 = self.prepare_structures(X1, structure_type)

        # Вычисляем Fused Gromov-Wasserstein
        try:
            # Пробуем сначала с энтропийной регуляризацией (быстрее)
            T, log = ot.fused_gromov_wasserstein(
                M, C0, C1, p, q,
                loss_fun=loss_fun,
                alpha=alpha,
                log=True,
                verbose=verbose
            )

            # Вычисляем расстояние
            fgw_dist = ot.fused_gromov_wasserstein2(
                M, C0, C1, p, q,
                loss_fun=loss_fun,
                alpha=alpha,
                log=False
            )

        except:
            # Если не получилось, используем регуляризованную версию
            T, log = ot.fused_gromov_wasserstein(
                M, C0, C1, p, q,
                loss_fun=loss_fun,
                alpha=alpha,
                log=True,
                verbose=verbose,
                epsilon=reg,
                max_iter=max_iter,
                tol=tol
            )

            fgw_dist = log['fgw_dist']

        if verbose:
            print(f"FGW расстояние (alpha={alpha}): {fgw_dist:.4f}")

        return fgw_dist, T, log

    def compute_fgw_with_structure_search(self, X0, X1, structure_types=['euclidean', 'sqeuclidean'],
                                          alphas=[0.0, 0.3, 0.5, 0.7, 1.0]):
        """
        Сравнивает FGW расстояние для разных типов структуры и alpha
        """
        results = {}

        for struct_type in structure_types:
            results[struct_type] = {}
            print(f"\n=== Структура: {struct_type} ===")

            # Предвычисляем структуры для этого типа
            C0 = self.prepare_structures(X0, struct_type)
            C1 = self.prepare_structures(X1, struct_type)

            for alpha in alphas:
                dist, _, _ = self.compute_fgw_distance(
                    X0, X1,
                    alpha=alpha,
                    structure_type=struct_type,
                    structure0=C0,
                    structure1=C1,
                    verbose=False
                )
                results[struct_type][alpha] = dist
                print(f"  alpha={alpha:.1f}: {dist:.4f}")

        # Визуализация
        self._plot_fgw_results(results, structure_types, alphas)

        return results

    def _plot_fgw_results(self, results, structure_types, alphas):
        """Визуализация результатов FGW"""
        plt.figure(figsize=(12, 5))

        # График зависимости от alpha
        plt.subplot(1, 2, 1)
        for struct_type in structure_types:
            dists = [results[struct_type][alpha] for alpha in alphas]
            plt.plot(alphas, dists, 'o-', label=struct_type, linewidth=2, markersize=8)

        plt.xlabel('alpha (1 - только признаки, 0 - только структура)')
        plt.ylabel('FGW расстояние')
        plt.title('Зависимость FGW от alpha и типа структуры')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Тепловая карта
        plt.subplot(1, 2, 2)
        dist_matrix = np.array([[results[st][alpha] for alpha in alphas]
                                for st in structure_types])

        im = plt.imshow(dist_matrix, aspect='auto', cmap='viridis')
        plt.colorbar(im, label='FGW расстояние')
        plt.xticks(range(len(alphas)), [f'{a:.1f}' for a in alphas])
        plt.yticks(range(len(structure_types)), structure_types)
        plt.xlabel('alpha')
        plt.ylabel('Тип структуры')
        plt.title('Тепловая карта FGW расстояний')

        plt.tight_layout()
        plt.show()
