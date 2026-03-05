import matplotlib.pyplot as pl
import numpy as np

from experiment.distances import dist_pairwise_matrix, dist_matrix


def show_dist(seqs_0, seqs_1, func_C, func_M = None):
    if not func_M:
          func_M = func_C
    C0 = dist_pairwise_matrix(seqs_0, func_C)
    C1 = dist_pairwise_matrix(seqs_1, func_C)
    M = dist_matrix(seqs_0, seqs_1, func_M)
    print(f"mean C0: {np.mean(C0)}")
    print(f"std C0: {np.std(C0)}")
    print(f"mean C1: {np.mean(C1)}")
    print(f"std C1: {np.std(C1)}")
    print(f"mean M: {np.mean(M)}")
    print(f"std M: {np.std(M)}")
    print(f"min M: {np.min(M)}")
    cmap = "Reds"

    pl.figure(2, (5, 5))
    fs = 15
    l_x = [0, 5, 10, 15]
    l_y = [0, 5, 10, 15, 20, 25]
    gs = pl.GridSpec(5, 5)

    ax1 = pl.subplot(gs[3:, :2])

    pl.imshow(C0, cmap=cmap, interpolation="nearest")
    pl.title("$C_1$", fontsize=fs)
    pl.xlabel("$k$", fontsize=fs)
    pl.ylabel("$i$", fontsize=fs)
    pl.xticks(l_x)
    pl.yticks(l_x)

    ax2 = pl.subplot(gs[:3, 2:])

    pl.imshow(C1, cmap=cmap, interpolation="nearest")
    pl.title("$C_2$", fontsize=fs)
    pl.ylabel("$l$", fontsize=fs)
    pl.xticks(())
    pl.yticks(l_y)
    ax2.set_aspect("auto")

    ax3 = pl.subplot(gs[3:, 2:], sharex=ax2, sharey=ax1)
    pl.imshow(M, cmap=cmap, interpolation="nearest")
    pl.yticks(l_x)
    pl.xticks(l_y)
    pl.ylabel("$i$", fontsize=fs)
    pl.title("$M_{AB}$", fontsize=fs)
    pl.xlabel("$j$", fontsize=fs)
    pl.tight_layout()
    ax3.set_aspect("auto")
    pl.show()
