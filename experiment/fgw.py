import ot
import numpy as np
import torch
import matplotlib.pyplot as plt
from pyexpat import features

from experiment.distances import dist_pairwise_matrix, fastdtw_dist, masked_length_awarded, dist_matrix


class FusedGromovWassersteinComputer:
    """
    Класс для вычисления Fused Gromov-Wasserstein расстояния между наборами матриц
    """
    @staticmethod
    def prepare_structures(x, structure_metric='dtw'):
        """
        Создает структурные матрицы (C) для Gromov-Wasserstein

        Args:
            x: входные данные (n_samples, m)
            structure_metric: тип структуры (masked_length_awarded, dtw)

        Returns:
            C: структурная матрица (n_samples, n_samples) или список матриц
        """
        # Вычисляем попарные расстояния между образцами
        # Используем субдискретизацию для уменьшения размера
        c = None
        if structure_metric == 'dtw':
            c = dist_pairwise_matrix(x, fastdtw_dist)
        elif structure_metric == 'masked_length_awarded':
            c = dist_pairwise_matrix(x, masked_length_awarded)
        return c

    @staticmethod
    def prepare_feature_matrix(x, y, feature_metric='masked_length_awarded'):
        """
        Создает фичи матрицы (M) для Gromov-Wasserstein
        """
        # Вычисляем попарные расстояния между образцами
        # Используем субдискретизацию для уменьшения размера
        m = None
        if feature_metric == 'dtw':
            m = dist_matrix(x, y, fastdtw_dist)
        elif feature_metric == 'masked_length_awarded':
            m = dist_matrix(x, y, masked_length_awarded)
        return m

    def compute_fgw_distance(self, x_0, x_1,
                             structure_metric='dtw',
                             feature_metric='masked_length_awarded',
                             loss_fun='square_loss',
                             precomputed_c_0 = None,
                             precomputed_c_1 = None,
                             precomputed_m=None,
                             alpha=0.5, count_plan=False, verbose=False):
        """
        Вычисляет Fused Gromov-Wasserstein расстояние между двумя наборами матриц

        Args:
            x_0:
            x_1:
            structure_metric: тип дистанции структурной матрицы (masked_length_awarded, dtw)
            feature_metric: тип дистанции признаковой матрицы (masked_length_awarded, dtw)
            loss_fun: функция потерь для GW ('square_loss', 'kl_loss')
            alpha: баланс между признаками (1-alpha для структуры, alpha для признаков)
                   alpha=0: только структура (GW)
                   alpha=1: только признаки (OT)
            count_plan:  расчитать ли план
            verbose: print log
            precomputed_c_0: вычисленные с_0
            precomputed_c_1: вычисленные с_1
            precomputed_m: вычисленные m
        """
        # Матрица признаковых расстояний (для Fused term)
        log = {}
        if not precomputed_m:
            m = self.prepare_feature_matrix(x_0, x_1, feature_metric)
        else:
            m = precomputed_m
        log['M'] = m

        # Веса объектов (равномерные)
        n0, n1 = len(x_0), len(x_1)
        p = np.ones(n0) / n0
        q = np.ones(n1) / n1

        # Получаем структурные матрицы
        if not precomputed_c_0:
            c_0 = self.prepare_structures(x_0, structure_metric)
        else:
            c_0 = precomputed_c_0

        if not precomputed_c_1:
            c_1 = self.prepare_structures(x_1, structure_metric)
        else:
            c_1 = precomputed_c_1
        log['C0'] = c_0
        log['C1'] = c_1
        # Вычисляем Fused Gromov-Wasserstein
        if count_plan:
            t_plan, plan_log = ot.fused_gromov_wasserstein(
                m, c_0, c_1, p, q,
                loss_fun=loss_fun,
                alpha=alpha,
                log=True,
                max_iter=100000
            )
            log['T'] = t_plan
            log['plan_log'] = plan_log

        # Вычисляем расстояние
        fgw_dist = ot.fused_gromov_wasserstein2(
            m, c_0, c_1, p, q,
            loss_fun=loss_fun,
            alpha=alpha,
            log=False,
            max_iter=100000
        )

        if verbose:
            print(f"FGW расстояние (alpha={alpha}): {fgw_dist:.4f}")

        return fgw_dist, log

    def compute_fugw_distance(self, x_0, x_1,
                             structure_metric='dtw',
                             feature_metric='masked_length_awarded',
                             loss_fun='square_loss',
                             reg_marginals: int | tuple =10,
                             precomputed_c_0 = None,
                             precomputed_c_1 = None,
                             precomputed_m=None,
                             alpha=0.5, verbose=False):
        """
        Вычисляет Fused Unbalanced Gromov-Wasserstein расстояние между двумя наборами матриц

        Args:
            x_0:
            x_1:
            structure_metric: тип дистанции структурной матрицы (masked_length_awarded, dtw)
            feature_metric: тип дистанции признаковой матрицы (masked_length_awarded, dtw)
            reg_marginals:
            loss_fun: функция потерь для GW ('square_loss', 'kl_loss')
            alpha: баланс между признаками (1-alpha для структуры, alpha для признаков)
                   alpha=0: только структура (GW)
                   alpha=1: только признаки (OT)
            verbose: print log
            precomputed_c_0: вычисленные с_0
            precomputed_c_1: вычисленные с_1
            precomputed_m: вычисленные m
        """
        # Матрица признаковых расстояний (для Fused term)
        log = {}
        if not precomputed_m:
            m = self.prepare_feature_matrix(x_0, x_1, feature_metric)
        else:
            m = precomputed_m
        log['M'] = m

        # Веса объектов (равномерные)
        n0, n1 = len(x_0), len(x_1)
        p = np.ones(n0) / n0
        q = np.ones(n1) / n1

        # Получаем структурные матрицы
        if not precomputed_c_0:
            c_0 = self.prepare_structures(x_0, structure_metric)
        else:
            c_0 = precomputed_c_0

        if not precomputed_c_1:
            c_1 = self.prepare_structures(x_1, structure_metric)
        else:
            c_1 = precomputed_c_1
        log['C0'] = c_0
        log['C1'] = c_1
        fugw_dist, plan_log = ot.gromov.fused_unbalanced_gromov_wasserstein2(
            Cx=c_0, Cy=c_1, wx=p, wy=q,
            reg_marginals=reg_marginals,
            M=m,
            loss_fun=loss_fun,
            alpha=alpha,
            log=True,
            max_iter=100000
        )
        log['distance'] = fugw_dist
        log['plan_log'] = plan_log

        if verbose:
            print(f"FGW расстояние (alpha={alpha}): {fugw_dist:.4f}")

        return fugw_dist, log

    def compute_fgw_with_structure_search(self, x_0, x_1, structure_metrics=None,
                                          alphas=None, make_plot=False):
        """
        Сравнивает FGW расстояние для разных типов структуры и alpha
        """
        if alphas is None:
            alphas = [0.0, 0.3, 0.5, 0.7, 1.0]
        if structure_metrics is None:
            structure_metrics = ['masked_length_awarded', 'dtw']
        results = {}
        pre_m = self.prepare_feature_matrix(x_0, x_1, 'masked_length_awarded')

        for struct_type in structure_metrics:
            results[struct_type] = {}
            print(f"\n=== Структура: {struct_type} ===")

            # Предвычисляем структуры для этого типа
            pre_c_0 = self.prepare_structures(x_0, struct_type)
            pre_c_1 = self.prepare_structures(x_1, struct_type)

            for alpha in alphas:
                dist = self.compute_fgw_distance(
                    x_0, x_1,
                    alpha=alpha,
                    structure_metric=struct_type,
                    precomputed_c_0=pre_c_0,
                    precomputed_c_1=pre_c_1,
                    precomputed_m=pre_m
                )
                results[struct_type][str(alpha)] = dist
                print(f"  alpha={alpha:.1f}: {dist:.4f}")

        # Визуализация
        if make_plot:
            self._plot_fgw_results(results, structure_metrics, alphas)

        return results

    def compute_fugw_with_structure_search(self, x_0, x_1,
                                          alphas=None, reg_marginals=None):
        """
        Сравнивает FGW расстояние для разных типов структуры и alpha
        """
        if alphas is None:
            alphas = [0.0, 0.3, 0.5, 0.7, 1.0]
        if reg_marginals is None:
            reg_marginals = [
                10, 100, 1000,
                (3000, 300),
                (1000, 500),
                (500, 1000),
                (300, 3000)
            ]

        structure_metric = 'dtw'
        feature_metric = 'masked_length_awarded'
        results = {}
        pre_m = self.prepare_feature_matrix(x_0, x_1, feature_metric)

        for alpha in alphas:
            results[str(alpha)] = {}
            print(f"\n=== alpha: {alpha} ===")

            # Предвычисляем структуры для этого типа
            pre_c_0 = self.prepare_structures(x_0, structure_metric)
            pre_c_1 = self.prepare_structures(x_1, structure_metric)

            for reg_marginal in reg_marginals:
                dist = self.compute_fugw_distance(
                    x_0, x_1,
                    alpha=alpha,
                    reg_marginals=reg_marginal,
                    precomputed_c_0=pre_c_0,
                    precomputed_c_1=pre_c_1,
                    precomputed_m=pre_m
                )
                results[str(alpha)][str(reg_marginal)] = dist
                print(f"  reg_marginal={reg_marginal:.1f}: {dist:.4f}")

        return results

    @staticmethod
    def _plot_fgw_results(results, structure_types, alphas):
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
        distances_matrix = np.array([[results[st][alpha] for alpha in alphas]
                                for st in structure_types])

        im = plt.imshow(distances_matrix, aspect='auto', cmap='viridis')
        plt.colorbar(im, label='FGW расстояние')
        plt.xticks(range(len(alphas)), [f'{a:.1f}' for a in alphas])
        plt.yticks(range(len(structure_types)), structure_types)
        plt.xlabel('alpha')
        plt.ylabel('Тип структуры')
        plt.title('Тепловая карта FGW расстояний')

        plt.tight_layout()
        plt.show()
