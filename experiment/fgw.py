import matplotlib.pyplot as plt
import numpy as np
import ot
import torch

from distances import dist_pairwise_matrix, fastdtw_dist, masked_length_awarded, dist_matrix
from preset_tools_difflen import pad_encoded_sequences


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
            structure_metric: тип структуры (masked_length_awarded, dtw, ot)

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
        elif structure_metric == 'ot':
            x_padding = pad_encoded_sequences(x)
            n_samples = x_padding.shape[0]

            if n_samples > 5000:
                c = np.zeros((n_samples, n_samples))
                batch_size = 1000
                for i in range(0, n_samples, batch_size):
                    end_i = min(i + batch_size, n_samples)
                    for j in range(0, n_samples, batch_size):
                        end_j = min(j + batch_size, n_samples)
                        # Вычисляем расстояния для батча
                        batch_dist = ot.dist(
                            x_padding[i:end_i],
                            x_padding[j:end_j],
                            metric='euclidean'
                        )
                        c[i:end_i, j:end_j] = batch_dist
            else:
                c = ot.dist(x_padding, x_padding, metric='euclidean')

            c /= c.max()

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
        elif feature_metric == 'ot':
            max_len_x = max(len(seq) for seq in x)
            max_len_y = max(len(seq) for seq in y)
            x_padding = pad_encoded_sequences(x, max_len=max(max_len_x, max_len_y))
            y_padding = pad_encoded_sequences(y, max_len=max(max_len_x, max_len_y))
            n_samples0 = x_padding.shape[0]
            n_samples1 = y_padding.shape[0]
            max_size_threshold = 4000
            # Проверяем, нужно ли использовать батчи
            if max(n_samples0, n_samples1) > max_size_threshold:
                m = np.zeros((n_samples0, n_samples1))
                batch_size = 1000
                # Вычисляем по батчам
                for i in range(0, n_samples0, batch_size):
                    end_i = min(i + batch_size, n_samples0)
                    for j in range(0, n_samples1, batch_size):
                        end_j = min(j + batch_size, n_samples1)

                        # Вычисляем расстояния для батча
                        batch_dist = ot.dist(
                            x_padding[i:end_i],
                            y_padding[j:end_j],
                            metric='sqeuclidean'
                        )
                        m[i:end_i, j:end_j] = batch_dist
            else:
                # Для небольших матриц вычисляем сразу
                m = ot.dist(x_padding, y_padding, metric='sqeuclidean')

            # Нормализуем
            if m.max() > 0:
                m /= m.max()
        return m

    def compute_fgw_distance(self, x_0, x_1,
                             structure_metric='dtw',
                             feature_metric='masked_length_awarded',
                             loss_fun='square_loss',
                             precomputed_c_0=None,
                             precomputed_c_1=None,
                             precomputed_m=None,
                             alpha=0.5, count_plan=False, verbose=False, on_gpu=False, write_log=False):
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
            on_gpu:
            write_log:
        """
        # Матрица признаковых расстояний (для Fused term)
        device = None
        dtype = None
        if on_gpu:
            assert torch.cuda.is_available(), "CUDA не доступна"
            device = torch.device("cuda")
            if verbose:
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            dtype = torch.float32

        log = {}
        if precomputed_m is None:
            m = self.prepare_feature_matrix(x_0, x_1, feature_metric)
            if on_gpu:
                if verbose:
                    print("перенос m на cuda")
                m = torch.tensor(m, dtype=dtype, device=device)
        else:
            m = precomputed_m
            if on_gpu:
                assert isinstance(m, torch.Tensor), f"m должен быть torch.Tensor, а получен {type(m)}"
                if verbose:
                    print("m is torch.Tensor")
        # Получаем структурные матрицы
        if precomputed_c_0 is None:
            c_0 = self.prepare_structures(x_0, structure_metric)
            if on_gpu:
                if verbose:
                    print("перенос c_0 на cuda")
                c_0 = torch.tensor(c_0, dtype=dtype, device=device)
        else:
            c_0 = precomputed_c_0
            if on_gpu:
                assert isinstance(c_0, torch.Tensor), f"c_0 должен быть torch.Tensor, а получен {type(c_0)}"
                if verbose:
                    print("c_0 is torch.Tensor")

        if precomputed_c_1 is None:
            c_1 = self.prepare_structures(x_1, structure_metric)
            if on_gpu:
                if verbose:
                    print("перенос c_1 на cuda")
                c_1 = torch.tensor(c_1, dtype=dtype, device=device)
        else:
            c_1 = precomputed_c_1
            if on_gpu:
                assert isinstance(c_1, torch.Tensor), f"c_1 должен быть torch.Tensor, а получен {type(c_1)}"
                if verbose:
                    print("c_1 is torch.Tensor")

        # Веса объектов (равномерные)
        n0, n1 = len(x_0), len(x_1)
        p = np.ones(n0) / n0
        q = np.ones(n1) / n1
        if on_gpu:
            p = torch.tensor(p, dtype=dtype, device=device)
            q = torch.tensor(q, dtype=dtype, device=device)

        if write_log:
            log['M'] = m
            log['C0'] = c_0
            log['C1'] = c_1

        if verbose:
            print(f"Используемый backend: {ot.backend.get_backend(c_0, c_1, m, p, q)}")

        # Вычисляем Fused Gromov-Wasserstein
        if count_plan:
            t_plan, plan_log = ot.fused_gromov_wasserstein(
                m, c_0, c_1, p, q,
                loss_fun=loss_fun,
                alpha=alpha,
                log=True,
                max_iter=50000
            )
            if write_log:
                log['T'] = t_plan
                log['plan_log'] = plan_log

        # Вычисляем расстояние
        fgw_dist = ot.fused_gromov_wasserstein2(
            m, c_0, c_1, p, q,
            loss_fun=loss_fun,
            alpha=alpha,
            log=False,
            max_iter=50000
        )
        if isinstance(fgw_dist, torch.Tensor):
            fgw_dist = fgw_dist.cpu().numpy().item()
            if verbose: print(f"FGW distance from Tensor: {fgw_dist}")

        return fgw_dist, log

    def compute_fugw_distance(self, x_0, x_1,
                              structure_metric='dtw',
                              feature_metric='masked_length_awarded',
                              loss_fun='square_loss',
                              reg_marginals: int | tuple = 10,
                              precomputed_c_0=None,
                              precomputed_c_1=None,
                              precomputed_m=None,
                              alpha=0.5, verbose=False, on_gpu=False, write_log=False):
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
            on_gpu: use GPU
            write_log: make log
        """
        # Матрица признаковых расстояний (для Fused term)
        log = {}
        device = None
        dtype = None
        if on_gpu:
            assert torch.cuda.is_available(), "CUDA не доступна"
            device = torch.device("cuda")
            if verbose: print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            dtype = torch.float32

        if precomputed_m is None:
            m = self.prepare_feature_matrix(x_0, x_1, feature_metric)
            if on_gpu:
                if verbose: print("перенос m на cuda")
                m = torch.tensor(m, dtype=dtype, device=device)
        else:
            m = precomputed_m
            if on_gpu:
                assert isinstance(m, torch.Tensor), f"m должен быть torch.Tensor, а получен {type(m)}"
                if verbose: print("m is torch.Tensor")
        # Получаем структурные матрицы
        if precomputed_c_0 is None:
            c_0 = self.prepare_structures(x_0, structure_metric)
            if on_gpu:
                if verbose: print("перенос c_0 на cuda")
                c_0 = torch.tensor(c_0, dtype=dtype, device=device)
        else:
            c_0 = precomputed_c_0
            if on_gpu:
                assert isinstance(c_0, torch.Tensor), f"c_0 должен быть torch.Tensor, а получен {type(c_0)}"
                if verbose: print("c_0 is torch.Tensor")

        if precomputed_c_1 is None:
            c_1 = self.prepare_structures(x_1, structure_metric)
            if on_gpu:
                if verbose: print("перенос c_1 на cuda")
                c_1 = torch.tensor(c_1, dtype=dtype, device=device)
        else:
            c_1 = precomputed_c_1
            if on_gpu:
                assert isinstance(c_1, torch.Tensor), f"c_1 должен быть torch.Tensor, а получен {type(c_1)}"
                if verbose: print("c_1 is torch.Tensor")
        # Веса объектов (равномерные)
        n0, n1 = len(x_0), len(x_1)
        p = np.ones(n0) / n0
        q = np.ones(n1) / n1
        if on_gpu:
            p = torch.tensor(p, dtype=dtype, device=device)
            q = torch.tensor(q, dtype=dtype, device=device)

        if write_log:
            log['M'] = m
            log['C0'] = c_0
            log['C1'] = c_1

        if verbose: print(f"Используемый backend: {ot.backend.get_backend(c_0, c_1, m, p, q)}")
        fugw_dist, plan_log = ot.gromov.fused_unbalanced_gromov_wasserstein2(
            Cx=c_0, Cy=c_1, wx=p, wy=q,
            reg_marginals=reg_marginals,
            M=m,
            loss_fun=loss_fun,
            alpha=alpha,
            log=True,
            max_iter=10000,
            max_iter_ot=10000,
            divergence='kl',
            unbalanced_solver='sinkhorn',
            epsilon=0.1
        )
        if isinstance(fugw_dist, torch.Tensor):
            fugw_dist = fugw_dist.cpu().numpy()
            if verbose: print(f"FGW distance from Tensor: {fugw_dist}")
        else:
            if verbose: print(f"FGW distance: {fugw_dist}")
        if write_log:
            log['distance'] = fugw_dist
            log['plan_log'] = plan_log

        if verbose: print(f"FGW расстояние (alpha={alpha}): {fugw_dist:.4f}")

        return fugw_dist, log

    def compute_fgw_with_search(self, x_0, x_1, structure_metrics=None, feature_metrics=None,
                                alphas=None, make_plot=False, verbose=False):
        """
        Сравнивает FGW расстояние для разных типов структуры и alpha
        """
        if alphas is None:
            alphas = [0.0, 0.3, 0.5, 0.7, 1.0]
        if structure_metrics is None:
            structure_metrics = ['ot','masked_length_awarded', 'dtw']
        if feature_metrics is None:
            feature_metrics = ['ot','masked_length_awarded', 'dtw']
        results = {}

        m_by_type = {}
        for feature_metric in feature_metrics:
            m = self.prepare_feature_matrix(x_0, x_1, feature_metric)
            m_by_type[feature_metric] = m

        c_by_type = {}
        for struct_type in structure_metrics:
            c_classes = {}
            pre_c_0 = self.prepare_structures(x_0, struct_type)
            pre_c_1 = self.prepare_structures(x_1, struct_type)
            c_classes['0'] = pre_c_0
            c_classes['1'] = pre_c_1
            c_by_type[struct_type] = c_classes

        for feature_metric in feature_metrics:
            pre_m = m_by_type[feature_metric]
            results[feature_metric + '_M'] = {}
            for struct_type in structure_metrics:
                results[feature_metric + '_M'][struct_type + '_C'] = {}
                if verbose:
                    print(f"\n=== Структура: {struct_type} ===")

                # Предвычисляем структуры для этого типа
                pre_c_0 = c_by_type[struct_type]['0']
                pre_c_1 = c_by_type[struct_type]['1']

                for alpha in alphas:
                    dist, _ = self.compute_fgw_distance(
                        x_0, x_1,
                        alpha=alpha,
                        structure_metric=struct_type,
                        precomputed_c_0=pre_c_0,
                        precomputed_c_1=pre_c_1,
                        precomputed_m=pre_m
                    )
                    results[feature_metric + '_M'][struct_type + '_C'][str(alpha)] = dist
                    if verbose:
                        print(f"  alpha={alpha:.1f}: {dist:.4f}")

        # Визуализация
        if make_plot:
            self._plot_fgw_results(results, structure_metrics, alphas)

        return results

    def compute_fgw_with_search_gpu(self, x_0, x_1, structure_metrics=None, feature_metrics=None,
                                alphas=None, make_plot=False, verbose=False):
        """
        Сравнивает FGW расстояние для разных типов структуры и alpha
        """
        assert torch.cuda.is_available(), "CUDA не доступна"
        device = torch.device("cuda")
        if verbose: print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        dtype = torch.float32

        if alphas is None:
            alphas = [0.0, 0.3, 0.5, 0.7, 1.0]
        if structure_metrics is None:
            structure_metrics = ['ot','masked_length_awarded', 'dtw']
        if feature_metrics is None:
            feature_metrics = ['ot','masked_length_awarded', 'dtw']
        results = {}

        m_by_type = {}
        for feature_metric in feature_metrics:
            m = self.prepare_feature_matrix(x_0, x_1, feature_metric)
            m = torch.tensor(m, dtype=dtype, device=device)
            m_by_type[feature_metric] = m

        c_by_type = {}
        for struct_type in structure_metrics:
            c_classes = {}
            pre_c_0 = self.prepare_structures(x_0, struct_type)
            pre_c_1 = self.prepare_structures(x_1, struct_type)

            pre_c_0 = torch.tensor(pre_c_0, dtype=dtype, device=device)
            pre_c_1 = torch.tensor(pre_c_1, dtype=dtype, device=device)

            c_classes['0'] = pre_c_0
            c_classes['1'] = pre_c_1
            c_by_type[struct_type] = c_classes

        for feature_metric in feature_metrics:
            pre_m = m_by_type[feature_metric]
            results[feature_metric + '_M'] = {}
            for struct_type in structure_metrics:
                results[feature_metric + '_M'][struct_type + '_C'] = {}
                if verbose: print(f"\n=== Структура: {struct_type} ===")

                # Предвычисляем структуры для этого типа
                pre_c_0 = c_by_type[struct_type]['0']
                pre_c_1 = c_by_type[struct_type]['1']

                for alpha in alphas:
                    dist, _ = self.compute_fgw_distance(
                        x_0, x_1,
                        alpha=alpha,
                        structure_metric=struct_type,
                        precomputed_c_0=pre_c_0,
                        precomputed_c_1=pre_c_1,
                        precomputed_m=pre_m,
                        verbose=verbose,
                        on_gpu=True
                    )
                    results[feature_metric + '_M'][struct_type + '_C'][str(alpha)] = dist
                    if verbose: print(f"  alpha={alpha:.1f}: {dist:.4f}")

        # Визуализация
        if make_plot:
            self._plot_fgw_results(results, structure_metrics, alphas)

        return results

    def compute_fugw_with_search(self, x_0, x_1, structure_metrics=None, feature_metrics=None,
                                 alphas=None, reg_marginals=None, verbose=False, on_gpu=False):
        """
        Сравнивает FGW расстояние для разных типов структуры и alpha
        """
        device = None
        dtype = None
        if on_gpu:
            assert torch.cuda.is_available(), "CUDA не доступна"
            device = torch.device("cuda")
            if verbose: print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            dtype = torch.float32

        if alphas is None:
            alphas = [0.3, 0.5, 0.7]
        if reg_marginals is None:
            reg_marginals = [
                10, 100,
                1000,
                3000,
                (3000, 500),
                (500, 3000),
            ]
        if structure_metrics is None:
            structure_metrics = ['ot', 'masked_length_awarded', 'dtw']
        if feature_metrics is None:
            feature_metrics = ['ot', 'masked_length_awarded', 'dtw']
        results = {}

        m_by_type = {}
        for feature_metric in feature_metrics:
            m = self.prepare_feature_matrix(x_0, x_1, feature_metric)

            if on_gpu:
                m = torch.tensor(m, dtype=dtype, device=device)

            m_by_type[feature_metric] = m

        c_by_type = {}
        for struct_type in structure_metrics:
            c_classes = {}
            pre_c_0 = self.prepare_structures(x_0, struct_type)
            pre_c_1 = self.prepare_structures(x_1, struct_type)

            if on_gpu:
                pre_c_0 = torch.tensor(pre_c_0, dtype=dtype, device=device)
                pre_c_1 = torch.tensor(pre_c_1, dtype=dtype, device=device)

            c_classes['0'] = pre_c_0
            c_classes['1'] = pre_c_1
            c_by_type[struct_type] = c_classes


        for feature_metric in feature_metrics:
            pre_m = m_by_type[feature_metric]
            results[feature_metric + '_M'] = {}
            if verbose: print(f"\n=== Структура M: {feature_metric} ===")
            for struct_type in structure_metrics:
                results[feature_metric + '_M'][struct_type + '_C'] = {}
                pre_c_0 = c_by_type[struct_type]['0']
                pre_c_1 = c_by_type[struct_type]['1']
                if verbose: print(f"\n=== Структура C: {struct_type} ===")
                for alpha in alphas:
                    results[feature_metric + '_M'][struct_type + '_C'][str(alpha)] = {}
                    if verbose: print(f"\n=== alpha: {alpha} ===")
                    for reg_marginal in reg_marginals:
                        dist, _ = self.compute_fugw_distance(
                            x_0, x_1,
                            alpha=alpha,
                            reg_marginals=reg_marginal,
                            precomputed_c_0=pre_c_0,
                            precomputed_c_1=pre_c_1,
                            precomputed_m=pre_m,
                            on_gpu=on_gpu,
                            verbose=verbose
                        )
                        results[feature_metric + '_M'][struct_type + '_C'][str(alpha)][str(reg_marginal)] = dist
                        if verbose: print(f"  reg_marginal={reg_marginal}: {dist:.4f}")
        return results

    def compute_fgw_fugw_with_search(self, x_0, x_1):
        alphas = [0.0, 0.5, 1.0]
        reg_marginals = [
            1, (1, 0), (0, 1),
            10, 100, 1000,
            (1000, 500),
            (500, 1000),
            (3000, 300),
            (300, 3000),
            3000
        ]

        structure_metric = 'dtw'
        feature_metric = 'masked_length_awarded'
        results_fugw = {}
        pre_m = self.prepare_feature_matrix(x_0, x_1, feature_metric)
        pre_c_0 = self.prepare_structures(x_0, structure_metric)
        pre_c_1 = self.prepare_structures(x_1, structure_metric)
        for alpha in alphas:
            results_fugw[str(alpha)] = {}
            # print(f"\n=== alpha: {alpha} ===")

            for reg_marginal in reg_marginals:
                dist, _ = self.compute_fugw_distance(
                    x_0, x_1,
                    alpha=alpha,
                    reg_marginals=reg_marginal,
                    precomputed_c_0=pre_c_0,
                    precomputed_c_1=pre_c_1,
                    precomputed_m=pre_m
                )
                results_fugw[str(alpha)][str(reg_marginal)] = dist
                # print(f"  reg_marginal={reg_marginal}: {dist:.4f}")

        results_fgw = {}
        alphas = [0.0,
                  0.3,
                  0.5,
                  0.7,
                  1.0]
        structure_metrics = ['masked_length_awarded', 'dtw', 'ot']
        for struct_type in structure_metrics:
            results_fgw[struct_type] = {}
            # print(f"\n=== Структура: {struct_type} ===")

            # Предвычисляем структуры для этого типа
            pre_c_0 = self.prepare_structures(x_0, struct_type)
            pre_c_1 = self.prepare_structures(x_1, struct_type)

            for alpha in alphas:
                dist, _ = self.compute_fgw_distance(
                    x_0, x_1,
                    alpha=alpha,
                    structure_metric=struct_type,
                    precomputed_c_0=pre_c_0,
                    precomputed_c_1=pre_c_1,
                    precomputed_m=pre_m
                )
                results_fgw[struct_type][str(alpha)] = dist
                # print(f"  alpha={alpha:.1f}: {dist:.4f}")

        result = {'fgw': results_fgw, 'fugw': results_fugw}

        return result

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
