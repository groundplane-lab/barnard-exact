"""Barnard's and Boschloo's exact unconditional tests for 2x2 tables (free core).

Both tests condition only on the row totals (the two sample sizes) and
treat the column totals as random.  Under the null hypothesis of equal
success probabilities, each row is Binomial(n_i, pi) independently, with
the common success probability pi an unknown *nuisance parameter*.  The
p-value is the worst case over pi in [0, 1] of the tail probability.

:func:`barnard_test` orders tables by the pooled Wald statistic and is
two-sided by test-statistic ordering: the p-value is
max_pi P(|T| >= |t_obs| | pi).

:func:`boschloo_test` uses Fisher's exact two-sided p-value of each table
as the test statistic B; the p-value is max_pi P(B <= b_obs | pi).
Boschloo's test is uniformly at least as powerful as Fisher's exact test:
on every table its p-value is <= Fisher's.
"""

import numpy as np

from ._core import (
    _barnard_region_twosided_exact,
    _exact_fisher_le,
    _fisher_twosided_grid,
    _maximize_over_pi,
    _TailProb,
    _validate_table,
)

__all__ = ["barnard_test", "boschloo_test", "FREE_MAX_N"]

#: Hard cap on the total sample size (n1 + n2) in the free version.
#: Beyond this the functions raise ValueError immediately instead of
#: attempting the calculation.
FREE_MAX_N = 200

_PAID_URL = "https://groundplane0.gumroad.com/l/conxfk"


def _check_free_cap(n1, n2):
    if n1 + n2 > FREE_MAX_N:
        raise ValueError(
            f"Free version is capped at total N <= {FREE_MAX_N}. "
            f"The full version ($30, one-time) is uncapped: {_PAID_URL}"
        )


def _unconditional_pvalue(table, region_fn):
    """Shared engine: build the rejection region, maximize over pi."""
    a, b, c, d, n1, n2 = _validate_table(table)
    _check_free_cap(n1, n2)
    if n1 == 0 or n2 == 0:
        # A group with no observations carries no information about its
        # success probability; there is nothing to test.
        return 1.0
    region = region_fn(a, b, c, d, n1, n2)
    tail = _TailProb(n1, n2, region)
    return _maximize_over_pi(tail)


def _barnard_region_twosided(a, b, c, d, n1, n2):
    # Exact integer tie classification: tables tied with the observed
    # table are always counted as "as extreme".
    return _barnard_region_twosided_exact(a, b, c, d, n1, n2)


def _band_region(F, f_obs, a, b, c, n1, n2, direction):
    # Float comparison is reliable outside a 1e-9 relative band (the float
    # error in these normalized sums is ~1e-14).  Cells inside the band,
    # true cross-margin ties plus float dust, are decided by exact
    # integer arithmetic.
    lo = f_obs * (1.0 - 1e-9)
    hi = f_obs * (1.0 + 1e-9)
    region = F <= lo
    band = (F > lo) & (F <= hi)
    if band.any():
        region = region | _exact_fisher_le(n1, n2, band, a, a + c, direction)
    return region


def _boschloo_region_twosided(a, b, c, d, n1, n2):
    F = _fisher_twosided_grid(n1, n2)
    return _band_region(F, F[a, c], a, b, c, n1, n2, "minlike")


def barnard_test(table):
    """Barnard's exact unconditional test for a 2x2 table (two-sided).

    Parameters
    ----------
    table : array_like, shape (2, 2)
        Observed counts ``[[a, b], [c, d]]`` (nonnegative integers).
        Rows are the two groups; only the row totals are conditioned on.
        The free version requires total N = n1 + n2 <= 200.

    Returns
    -------
    float
        Two-sided exact p-value: max over the nuisance parameter pi in
        [0, 1] of the null probability (independent Binomial(n1, pi) x
        Binomial(n2, pi)) of tables whose absolute pooled Wald statistic
        is at least as large as the observed table's.

    Notes
    -----
    Two-sidedness is by test-statistic ordering (|T| >= |t_obs|), not by
    doubling a one-sided p-value.  Tables with a zero row total return 1.0
    (no information).
    """
    return _unconditional_pvalue(table, _barnard_region_twosided)


def boschloo_test(table):
    """Boschloo's exact unconditional test for a 2x2 table (two-sided).

    Parameters
    ----------
    table : array_like, shape (2, 2)
        Observed counts ``[[a, b], [c, d]]`` (nonnegative integers).
        Rows are the two groups; only the row totals are conditioned on.
        The free version requires total N = n1 + n2 <= 200.

    Returns
    -------
    float
        Two-sided exact p-value: max over pi in [0, 1] of the null
        probability of tables whose Fisher exact two-sided p-value is no
        larger than the observed table's.  Always <= the Fisher exact
        two-sided p-value of the same table.

    Notes
    -----
    Fisher's two-sided p-value here uses the minimum-likelihood ordering
    (sum of hypergeometric probabilities <= the observed table's), with
    ties classified in exact integer arithmetic.  Tables with a zero row
    total return 1.0.
    """
    return _unconditional_pvalue(table, _boschloo_region_twosided)
