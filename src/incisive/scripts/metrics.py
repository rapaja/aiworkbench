import numpy as np


def evaluate(target: np.array, prediction: np.array) -> tuple[int, int, int, int]:
    agreed = target == prediction
    tp = prediction & agreed
    tn = ~prediction & agreed
    fp = prediction & ~agreed
    fn = ~prediction & ~agreed
    return int(np.sum(tp)), int(np.sum(tn)), int(np.sum(fp)), int(np.sum(fn))


def p_r_f1(tp: int, tn: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = 2*tp / (2*tp + fp + fn)
    return p, r, f1
