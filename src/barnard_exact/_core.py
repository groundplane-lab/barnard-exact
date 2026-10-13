"""Core machinery for unconditional exact tests of 2x2 tables (free core).

Implements the classical mathematics behind the tests:

* Barnard's exact test: order all tables with the observed row totals by
  the pooled Wald statistic

      T = (x1/n1 - x2/n2) / sqrt(p_pool (1-p_pool) (1/n1 + 1/n2)),

  with p_pool = (x1 + x2) / (n1 + n2).  The two-sided p-value is

      max_{pi in [0,1]} P(|T| >= |t_obs| | pi),

  where under the null each row is Binomial(n_i, pi) independently.

* Boschloo's exact test: use Fisher's exact two-sided p-value of each
  table as the test statistic B.  The p-value is

      max_{pi in [0,1]} P(B <= b_obs | pi)

  with the same binomial-product probabilities.

Because the rejection region is fixed (the test statistic does not depend
on the nuisance parameter pi), the tail probability being maximized is a
*polynomial* in pi (degree n1+n2).  It is maximized by a fine grid plus
local Brent refinement of the best candidates.
"""

import math

import numpy as np
from scipy.optimize import minimize_scalar

__all__ = [
    "_validate_table",
    "_barnard_region_twosided_exact",
    "_fisher_twosided_grid",
    "_exact_fisher_le",
    "_TailProb",
    "_maximize_over_pi",
]


def _validate_table(table):
    """Validate a 2x2 table of counts; return (a, b, c, d, n1, n2) as ints.

    Anything that is not a 2x2 array of nonnegative integers (fractional
    entries, negatives, NaN/inf, wrong shape, non-numeric input) raises
    ValueError loudly.  Nothing is silently coerced or truncated.
    """
    try:
        T = np.asarray(table)
    except Exception as exc:
        raise ValueError(f"table could not be interpreted as an array: {exc}")
    if T.dtype == object:
        # e.g. ragged input like [[1, 2], [3]]; asarray gives object dtype
        raise ValueError("table must be a 2x2 array of counts")
    if T.dtype.kind == "b":
        raise ValueError("table entries must be integer counts, not booleans")
    if T.dtype.kind == "c":
        raise ValueError("table entries must be real counts, not complex")
    if T.dtype.kind not in "iuf":
        raise ValueError("table entries must be numeric counts")
    if T.ndim != 2 or T.shape != (2, 2):
        raise ValueError(f"table must be 2x2, got shape {T.shape}")
    if not np.all(np.isfinite(T)):
        raise ValueError("table entries must be finite (no NaN or inf)")
    if np.any(T < 0):
        raise ValueError("table entries must be nonnegative")
    if not np.all(np.equal(np.mod(T, 1), 0)):
        raise ValueError("table entries must be integers (no fractional counts)")
    a, b, c, d = (int(v) for v in T.flat)
    return a, b, c, d, a + b, c + d


def _log_binom_coefs(n):
    """log C(n, k) for k = 0..n (float array)."""
    ks = np.arange(n + 1, dtype=float)
    return (
        math.lgamma(n + 1)
        - np.array([math.lgamma(k + 1) for k in ks])
        - np.array([math.lgamma(n - k + 1) for k in ks])
    )


def _fisher_twosided_grid(n1, n2):
    """Two-sided Fisher exact p-value for every table with row totals n1, n2.

    Each table (x1, x2) is scored under its own column margin s = x1 + x2:
    with the hypergeometric law Hypergeometric(N, n1, s), the p-value is the
    total probability of tables whose probability is <= the observed table's
    (Fisher's "minimum likelihood" ordering).  Tie classification uses exact
    integer arithmetic on the weights C(n1,x1)*C(n2,s-x1), so no
    floating-point tolerance decides extremeness.
    """
    N = n1 + n2
    lg1 = _log_binom_coefs(n1)
    lg2 = _log_binom_coefs(n2)
    lgN = math.lgamma(N + 1)
    F = np.empty((n1 + 1, n2 + 1))
    for s in range(N + 1):
        lo = max(0, s - n2)
        hi = min(n1, s)
        xs = np.arange(lo, hi + 1)
        x2s = s - xs
        lp = lg1[xs] + lg2[x2s] - (lgN - math.lgamma(s + 1) - math.lgamma(N - s + 1))
        # Exact tie weights (integers): P(x) <= P(y)  <=>  w(x) >= w(y).
        w = [math.comb(n1, int(x)) * math.comb(n2, int(s - x)) for x in xs]
        order = sorted(range(len(xs)), key=lambda i: w[i])
        pmf = np.exp(lp - lp.max())
        pmf /= pmf.sum()
        spo = pmf[order]
        cso = np.cumsum(spo)
        fp = np.empty(len(xs))
        j = 0
        while j < len(xs):
            k = j
            while k + 1 < len(xs) and w[order[k + 1]] == w[order[j]]:
                k += 1
            # tied tables share the largest cumulative probability
            fp[[order[t] for t in range(j, k + 1)]] = cso[k]
            j = k + 1
        F[xs, x2s] = fp
    return F


def _exact_fisher_margin(n1, n2, s, direction):
    """Exact rational Fisher p-values for every x1 on column margin s.

    Returns ``(A, D)`` with ``p(x1) == A[x1] / D`` exactly, as Python
    integers.  ``direction`` is ``"minlike"`` (two-sided
    minimum-likelihood ordering).  Used to resolve cross-margin ties that
    floating point cannot decide.
    """
    N = n1 + n2
    lo, hi = max(0, s - n2), min(n1, s)
    ys = list(range(lo, hi + 1))
    ws = [math.comb(n1, y) * math.comb(n2, s - y) for y in ys]
    D = math.comb(N, s)
    A = {}
    if direction == "minlike":
        # group tied weights: A(w) = sum of weights <= w (minimum-
        # likelihood ordering sums the probabilities of tables no more
        # likely than the observed one)
        pairs = sorted(zip(ws, ys), key=lambda t: t[0])
        run = 0
        i = 0
        while i < len(pairs):
            j = i
            while j < len(pairs) and pairs[j][0] == pairs[i][0]:
                j += 1
            run += sum(w for w, _ in pairs[i:j])
            for _, y in pairs[i:j]:
                A[y] = run
            i = j
    else:
        raise ValueError("direction must be 'minlike'")
    return A, D


def _exact_fisher_le(n1, n2, mask, a_obs, s_obs, direction):
    """Decide ``F(t) <= F(obs)`` exactly for cells where ``mask`` is True.

    Compares exact rational Fisher p-values by integer
    cross-multiplication (``A_t * D_obs <= A_obs * D_s``), so
    cross-margin ties are never misclassified by floating-point noise.
    Cells where ``mask`` is False are False in the output.
    """
    A_obs_d, D_obs = _exact_fisher_margin(n1, n2, s_obs, direction)
    A_obs = A_obs_d[a_obs]
    out = np.zeros_like(mask)
    cache = {}
    for x1, x2 in np.argwhere(mask):
        x1, x2 = int(x1), int(x2)
        s = x1 + x2
        if s not in cache:
            cache[s] = _exact_fisher_margin(n1, n2, s, direction)
        A_d, D_s = cache[s]
        if A_d[x1] * D_obs <= A_obs * D_s:
            out[x1, x2] = True
    return out


def _wald_t2_num_den(x1, x2, n1, n2):
    """Exact integer numerator/denominator of the squared pooled Wald statistic.

    T(x1, x2)^2 = num / den, where with s = x1 + x2 and N = n1 + n2::

        num = (x1*n2 - x2*n1)^2
        den = s * (N - s)

    (the factors N, n1, n2 cancel in any ratio comparison).  A zero
    denominator means T = 0 (the numerator is zero too in that case).
    All values are Python integers: comparisons of squared statistics are
    exact, so tied tables are never misclassified by floating-point noise.
    """
    s = x1 + x2
    return (x1 * n2 - x2 * n1) ** 2, s * (n1 + n2 - s)


def _barnard_region_twosided_exact(a, b, c, d, n1, n2):
    """|T| >= |t_obs| region via exact integer arithmetic (no float ties)."""
    num0, den0 = _wald_t2_num_den(a, c, n1, n2)
    x1 = np.arange(n1 + 1, dtype=object)[:, None]
    x2 = np.arange(n2 + 1, dtype=object)[None, :]
    num1, den1 = _wald_t2_num_den(x1, x2, n1, n2)
    if num0 == 0 or den0 == 0:
        # t_obs == 0: every table is at least as extreme
        return np.ones((n1 + 1, n2 + 1), dtype=bool)
    with np.errstate(all="ignore"):
        # den1 == 0  ->  T = 0 < |t_obs|  ->  excluded
        region = np.logical_and(den1 != 0, num1 * den0 >= num0 * den1)
    return np.asarray(region, dtype=bool)


class _TailProb:
    """Tail probability P(region | pi) as a callable of pi in [0, 1].

    Precomputes the log binomial coefficients and the sufficient statistic
    S = x1 + x2 over the rejection region, so each evaluation is a single
    vectorized log-sum-exp: O(|region|) work.  pi = 0 and pi = 1 are handled
    exactly (only the all-failure / all-success table has mass there).
    """

    def __init__(self, n1, n2, region):
        self._n = n1 + n2
        x1 = np.arange(n1 + 1)[:, None]
        x2 = np.arange(n2 + 1)[None, :]
        lg1 = _log_binom_coefs(n1)[:, None]
        lg2 = _log_binom_coefs(n2)[None, :]
        self._K = (lg1 + lg2)[region].astype(float)
        self._S = (x1 + x2)[region].astype(float)
        self._p0 = 1.0 if bool((self._S == 0).any()) else 0.0
        self._p1 = 1.0 if bool((self._S == self._n).any()) else 0.0

    def __call__(self, pi):
        if self._K.size == 0:
            return 0.0
        if pi <= 0.0:
            return self._p0
        if pi >= 1.0:
            return self._p1
        lp = self._K + self._S * math.log(pi) + (self._n - self._S) * math.log1p(-pi)
        m = float(lp.max())
        return float(np.exp(lp - m).sum() * math.exp(m))


def _maximize_over_pi(tail, n_grid=2001, n_refine=32):
    """max_{pi in [0,1]} tail(pi): dense grid plus Brent local refinement.

    The tail probability is a polynomial in pi (the rejection region is
    fixed), so it is smooth; still, narrow peaks are possible, hence the
    fine grid plus bounded Brent refinement around the best candidates.
    Deterministic: same input always gives the same output.
    """
    pis = np.linspace(0.0, 1.0, n_grid)
    vals = np.empty(n_grid)
    for i, pi in enumerate(pis):
        vals[i] = tail(pi)
    best = float(vals.max())
    h = 1.0 / (n_grid - 1)
    order = np.argsort(vals, kind="stable")
    blocked = np.zeros(n_grid, dtype=bool)
    done = 0
    for i in order[::-1]:
        if done >= n_refine:
            break
        if blocked[i]:
            continue
        blocked[max(0, i - 2) : i + 3] = True
        done += 1
        lo = max(0.0, pis[i] - 3 * h)
        hi = min(1.0, pis[i] + 3 * h)
        if hi <= lo:
            continue
        res = minimize_scalar(
            lambda p: -tail(p),
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": 1e-13, "maxiter": 300},
        )
        if math.isfinite(res.fun) and -res.fun > best:
            best = -res.fun
    return min(1.0, max(0.0, best))
