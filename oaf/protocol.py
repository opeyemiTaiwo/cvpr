"""Consistent-identity evaluation used in the paper.

Fingerprints -> identities by average-linkage clustering on cosine distance.
The cut is set on one half of the people (largest cut with pair precision
>= target) and applied unchanged to the other half; 25 random person splits
x 2 directions = 50 tests. Identical to the notebooks' evaluation cells.
"""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform

CUTS = np.geomspace(0.001, 1.5, 120)


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def score_identities(pred, p, c):
    """Clip-level IDF1, IDs per person, pair precision, cross-camera recall."""
    _, pi = np.unique(p, return_inverse=True)
    _, ci = np.unique(pred, return_inverse=True)
    M = np.zeros((pi.max() + 1, ci.max() + 1)); np.add.at(M, (pi, ci), 1)
    r, cc = linear_sum_assignment(-M)
    nc = M.sum(0); tp, same = (M * (M - 1)).sum(), (nc * (nc - 1)).sum()
    _, ki = np.unique(c, return_inverse=True)
    Kc = np.zeros((M.shape[0], ki.max() + 1, M.shape[1])); np.add.at(Kc, (pi, ki, ci), 1)
    nq = M.sum(1)
    x_tot = (nq ** 2 - (Kc.sum(2) ** 2).sum(1)).sum()
    x_same = ((M ** 2).sum(1) - (Kc ** 2).sum((1, 2))).sum()
    return {'IDF1 %': 100 * M[r, cc].sum() / len(p),
            'IDs per person': (M > 0).sum(1).mean(),
            'pair precision %': 100 * tp / same if same else 100.0,
            'cross-camera recall %': 100 * x_same / max(1, x_tot)}


def tree(X, idx):
    D = np.clip(1 - X[idx] @ X[idx].T, 0, None); D = (D + D.T) / 2
    np.fill_diagonal(D, 0)
    return linkage(squareform(D, checks=False), 'average')


def calibrate(L, idx, pid, cam, target=0.90):
    best = CUTS[0]
    for t in CUTS:
        if score_identities(fcluster(L, t, 'distance'), pid[idx], cam[idx]
                            )['pair precision %'] >= 100 * target:
            best = t
    return best


def consistent_id_50tests(X, pid, cam, n_splits=25, seed=2027, target=0.90):
    """Returns a list of 50 result dicts (one per test)."""
    X = unit(np.asarray(X, np.float32))
    people = np.unique(pid)
    rng = np.random.default_rng(seed)
    out = []
    for s in range(n_splits):
        perm = rng.permutation(people)
        a = np.isin(pid, perm[:len(people) // 2])
        G = [np.flatnonzero(a), np.flatnonzero(~a)]
        Ls = [tree(X, g) for g in G]
        for c, t in [(0, 1), (1, 0)]:
            r = score_identities(fcluster(Ls[t], calibrate(Ls[c], G[c], pid, cam, target),
                                          'distance'), pid[G[t]], cam[G[t]])
            r.update(split=s, direction=c)
            out.append(r)
    return out
