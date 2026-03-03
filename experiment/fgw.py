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

    def prepare_structures(self, X, structure_type='grid', **kwargs):
        """
        Создает структурные матрицы (C) для Gromov-Wasserstein

        Args:
            X: входные данные (n_samples, 46, 100)
            structure_type: тип структуры ('grid', 'euclidean', 'correlation')

        Returns:
            C: структурная матрица (n_samples, n_samples) или список матриц
        """
        n_samples = X.shape[0]
        if structure_type == 'grid':

            # Вычисляем попарные расстояния между образцами
            # Используем субдискретизацию для уменьшения размера
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
                            metric='euclidean'
                        )
                        C[i:end_i, j:end_j] = batch_dist
            else:
                C = ot.dist(X, X, metric='euclidean')

            C /= C.max()
            return C

        elif structure_type == 'euclidean':
            # Структура на основе евклидовых расстояний между выпрямленными векторами
            X_flat = X.reshape(n_samples, -1)
            C = ot.dist(X_flat, X_flat, metric='euclidean')
            C /= C.max()
            return C

        elif structure_type == 'correlation':
            # Структура на основе корреляций между признаками
            X_flat = X.reshape(n_samples, -1)
            corr = np.corrcoef(X_flat)
            # Преобразуем корреляцию в расстояние
            C = 1 - np.abs(corr)
            np.fill_diagonal(C, 0)
            return C

        elif structure_type == 'individual':
            # Индивидуальная структура для каждого объекта
            # (например, попарные расстояния между точками внутри каждой матрицы)
            C_list = []
            X_flat = X.reshape(n_samples, 46 * 100)

            for i in range(n_samples):
                # Восстанавливаем матрицу 46x100
                matrix = X_flat[i].reshape(46, 100)
                # Создаем структуру на основе градиентов или локальных патчей
                grad_x = np.gradient(matrix, axis=0)
                grad_y = np.gradient(matrix, axis=1)
                structure = np.sqrt(grad_x ** 2 + grad_y ** 2).ravel()
                # Попарные расстояния между "структурными признаками"
                C_i = ot.dist(structure.reshape(-1, 1), structure.reshape(-1, 1))
                C_list.append(C_i / C_i.max())

            return C_list
        return None

    def compute_fgw_distance(self, X0, X1, alpha=0.5, reg=0.01,
                             structure_type='grid', loss_fun='square_loss',
                             structure0=None, structure1=None,
                             max_iter=100, tol=1e-6, verbose=True):
        """
        Вычисляет Fused Gromov-Wasserstein расстояние между двумя наборами матриц

        Args:
            X0, X1: данные формы (n_samples, 46, 100)
            alpha: баланс между признаками (1-alpha для структуры, alpha для признаков)
                   alpha=0: только структура (GW)
                   alpha=1: только признаки (OT)
            reg: энтропийная регуляризация
            structure_type: тип структуры ('grid', 'euclidean', 'correlation', 'individual')
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
        if structure0 is None:
            if structure_type == 'individual':
                C0 = self.prepare_structures(X0, structure_type)
                C1 = self.prepare_structures(X1, structure_type)
                # Для individual нужно усреднить или использовать список
                C0 = np.mean(C0, axis=0) if isinstance(C0, list) else C0
                C1 = np.mean(C1, axis=0) if isinstance(C1, list) else C1
            else:
                C0 = self.prepare_structures(X0, structure_type)
                C1 = self.prepare_structures(X1, structure_type)
        else:
            C0, C1 = structure0, structure1

        # Убеждаемся, что C0 и C1 - матрицы правильной формы
        if len(C0.shape) == 2 and C0.shape[0] == n0:
            pass  # уже правильно
        else:
            # Если C0 - общая структура для всех объектов
            C0 = np.tile(C0, (n0, 1))[:n0, :n0] if C0.shape[0] != n0 else C0
            C1 = np.tile(C1, (n1, 1))[:n1, :n1] if C1.shape[0] != n1 else C1

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

    def compute_fgw_with_structure_search(self, X0, X1, structure_types=['grid', 'euclidean', 'correlation'],
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
