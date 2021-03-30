import numpy as np


def dist(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def rotate_figure(figure, angle):
    fig = []
    for i in range(len(figure)):
        point = figure[i]
        fig_extra = [0, 0]
        for j in (0, 1):
            fig_extra[j] = np.cos(angle) * point[j] \
                              - ((-1) ** j) * np.sin(angle) * point[1 - j]
        fig.append((fig_extra[0], fig_extra[1]))
    return fig


def scale_figure(f, scale):
    res = []
    for i in range(len(f)):
        res.append((f[i][0] * scale, f[i][1] * scale))
    return res


def shift_figure(f, shift):
    res = []
    for i in range(len(f)):
        res.append((f[i][0] + shift[0], f[i][1] + shift[1]))
    return res


def compare_figures(fig1, fig2):
    f1 = fig1.copy()
    f2 = fig2.copy()

    s = 0
    idx = 0
    for i in range(len(f1)):
        min_d = np.Inf
        for j in range(len(f2)):
            d = dist(f1[i], f2[j])
            if d < min_d:
                min_d = d
                idx = j
        f2.pop(idx)
        s += min_d
    return s