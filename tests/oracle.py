from decimal import Context, Decimal, localcontext
from operator import mul

import numpy as np

# The reference of design section 11. Every operation rounds to 60 significant digits,
# so the float64 results are correctly rounded for the condition numbers of the tests,
# and they are the same on every platform.
CONTEXT = Context(prec=60)


def cholesky(A):
    L = []
    for i, row in enumerate(A):
        Li = []
        L.append(Li)
        for j in range(i + 1):
            s = row[j] - sum(map(mul, Li, L[j]))
            Li.append(s.sqrt() if i == j else s / L[j][j])
    return L


def forward(L, b):
    y = []
    for row, value in zip(L, b, strict=True):
        y.append((value - sum(map(mul, row, y))) / row[len(y)])
    return y


def conditional(C, G, X, T=()):
    # Mean (nspec, |G|) and covariance (|G|, |G|) of x_G given x_O, O the other pixels,
    # from a Cholesky solve of C_OO; and chi2 (nspec,) = r_T^T C_TT^-1 r_T over T, a
    # subset of G, with r_T = x_T - mean_T and C_TT the conditional covariance at T.
    G = list(G)
    O = [i for i in range(len(C)) if i not in G]
    t = [G.index(i) for i in T]
    with localcontext(CONTEXT):
        c = [[Decimal(v) for v in row] for row in C.tolist()]
        L = cholesky([[c[i][j] for j in O] for i in O])
        Z = [forward(L, [c[i][g] for i in O]) for g in G]
        cov = [
            [c[g][h] - sum(map(mul, Z[a], Z[b])) for b, h in enumerate(G)]
            for a, g in enumerate(G)
        ]
        LT = cholesky([[cov[a][b] for b in t] for a in t])
        mean, chi2 = [], []
        for x in X.tolist():
            y = forward(L, [Decimal(x[i]) for i in O])
            mean.append([sum(map(mul, z, y)) for z in Z])
            r = forward(LT, [Decimal(x[G[a]]) - mean[-1][a] for a in t])
            chi2.append(sum(map(mul, r, r)))
    return np.array(mean, float), np.array(cov, float), np.array(chi2, float)
