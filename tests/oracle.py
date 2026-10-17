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
    mean = np.array(mean, float).reshape(len(X), len(G))
    return mean, np.array(cov, float), np.array(chi2, float)


def window(i, nguard, npix):
    # The guard window of pixel i: the pixels within nguard of it.
    return range(max(0, i - nguard), min(npix, i + nguard + 1))


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
