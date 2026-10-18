from decimal import Context, Decimal, localcontext
from operator import mul

import numpy as np
from scipy import stats

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
    # C is an array, or the rows of Decimal of low_rank.
    G = list(G)
    O = [i for i in range(len(C)) if i not in G]
    t = [G.index(i) for i in T]
    with localcontext(CONTEXT):
        c = (
            C
            if isinstance(C, list)
            else [[Decimal(v) for v in row] for row in C.tolist()]
        )
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
    mean = np.array(mean, float).reshape(len(X), len(G))
    return mean, np.array(cov, float), np.array(chi2, float)


def low_rank(V, values, d):
    # C = V diag(values) V^T + diag(d) as rows of Decimal, from the float64 fields: the
    # products are exact and the sums round to 60 digits, so C is not rounded to
    # float64 as a dense matrix would be.
    with localcontext(CONTEXT):
        rows = [[Decimal(v) for v in row] for row in V.tolist()]
        scaled = [
            [v * Decimal(s) for v, s in zip(row, values.tolist())] for row in rows
        ]
        return [
            [
                sum(map(mul, scaled[i], rows[j])) + (Decimal(d[i]) if i == j else 0)
                for j in range(len(rows))
            ]
            for i in range(len(rows))
        ]


def inverse(c):
    # The inverse (float64) of the rows of Decimal c, by a 60-digit Cholesky
    # factorization c = L L^T: P_ij = sum_k (L^-1)_ki (L^-1)_kj.
    n = len(c)
    with localcontext(CONTEXT):
        L = cholesky(c)
        Z = [forward(L, [Decimal(int(i == j)) for i in range(n)]) for j in range(n)]
        return np.array(
            [[sum(map(mul, Z[i], Z[j])) for j in range(n)] for i in range(n)], float
        )


def kappa_w(V, values, d):
    # kappa_W of design section 4.2 (decision A28).
    c = d + np.square(V) @ values
    B = V * np.sqrt(values) / np.sqrt(c)[:, np.newaxis]
    return (np.max(d / c) + np.linalg.eigvalsh(B.T @ B)[-1]) / np.min(d / c)


def window(i, nguard, npix, width=1):
    # The guard window of pixels i to i + width - 1: the pixels within nguard of them.
    return range(max(0, i - nguard), min(npix, i + width + nguard))


def guard_conditional(C, nguard, X, pixels):
    # Mean (nspec, len(pixels)) and variance (len(pixels),) of each pixel given the
    # pixels farther than nguard from it, as pixelwise predicts it.
    means, variances = [], []
    for i in pixels:
        g = window(i, nguard, len(C))
        mean, cov, _ = conditional(C, g, X)
        means.append(mean[:, i - g.start])
        variances.append(cov[i - g.start, i - g.start])
    return np.array(means).T, np.array(variances)


def random_spd(npix, kappa, seed):
    # A test matrix of design section 11: Q diag(lambda) Q^T with log-spaced eigenvalues
    # from 1 to 1/kappa. The mean of C and C^T is exactly symmetric.
    Q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((npix, npix)))
    C = (Q * np.logspace(0, -np.log10(kappa), npix)) @ Q.T
    return (C + C.T) / 2


def random_low_rank(npix, rank, kappa, seed):
    # The fields (V, values, d) of a low-rank test matrix of design section 11, with
    # orthonormal V, values log-spaced from 1 to 4 / kappa and d uniform on
    # [2 / kappa, 8 / kappa]: C scaled to unit diagonal then has a condition number
    # within a factor 4 of kappa.
    rng = np.random.default_rng(seed)
    V, _ = np.linalg.qr(rng.standard_normal((npix, rank)))
    return V, np.geomspace(1, 4 / kappa, rank), rng.uniform(2, 8, npix) / kappa


def dense(C):
    # C as a float64 matrix, also for a LowRank.
    if isinstance(C, np.ndarray):
        return C
    return (C.vectors * C.values) @ C.vectors.T + np.diag(C.diag)


def condition(C):
    # Of C scaled to unit diagonal: the kappa of the tolerances of design section 11.
    d = 1 / np.sqrt(np.diagonal(C))
    return np.linalg.cond(d[:, np.newaxis] * C * d)


def assert_gaussian(samples, mean, cov):
    # If the n rows of samples are iid N(mean, cov), with cov = L L^T, then
    # y = L^-1 (sample - mean) is iid N(0, I). Each entry of sqrt(n) mean(y) is then
    # N(0, 1); and (n - 1) cov(y) is Wishart, so the diagonal of cov(y) has the standard
    # deviation sqrt(2 / (n - 1)) about 1 and the other entries sqrt(1 / (n - 1)) about
    # 0, close to normal for large n. Over the m = k + k (k + 1) / 2 entries tested,
    # z = Phi^-1(1 - 0.001 / (2 m)) standard deviations each bound the probability of
    # a failure by 0.001 (Bonferroni).
    n, k = samples.shape
    y = np.linalg.solve(np.linalg.cholesky(cov), (samples - mean).T).T
    upper = np.triu_indices(k)
    z = stats.norm.isf(0.001 / (2 * (k + upper[0].size)))
    assert np.all(np.abs(y.mean(axis=0)) <= z / np.sqrt(n))
    deviation = (np.cov(y, rowvar=False) - np.eye(k))[upper]
    sd = np.where(upper[0] == upper[1], np.sqrt(2 / (n - 1)), np.sqrt(1 / (n - 1)))
    assert np.all(np.abs(deviation) <= z * sd)
