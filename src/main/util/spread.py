"""Spread Size"""

import numpy as np


def spread_size(seeds: list, gin_mat: np.ndarray, gout_sz: float) -> list:
    sub_mat = gin_mat[np.array(seeds)]
    virus_mat = np.cumsum(sub_mat, axis=0) > 0
    virus = virus_mat.sum(axis=1)
    sp = virus / gin_mat.shape[1] * gout_sz
    return list(sp)