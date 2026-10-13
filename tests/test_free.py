"""Tests for the free barnard_exact core.

Correctness is checked against the paid package (imported from its own
source tree, never copied).  Cap behavior and input validation are
checked directly.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

import barnard_exact

HERE = Path(__file__).resolve().parent
PAID_SRC = HERE.parent.parent.parent / "barnard" / "src" / "barnard"


def _load_paid():
    spec = importlib.util.spec_from_file_location(
        "paid_barnard",
        PAID_SRC / "__init__.py",
        submodule_search_locations=[str(PAID_SRC)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["paid_barnard"] = mod
    spec.loader.exec_module(mod)
    return mod


paid = _load_paid()

PAID_URL = "https://groundplane0.gumroad.com/l/conxfk"

TABLES = [
    [[2, 3], [3, 2]],
    [[7, 12], [3, 18]],
    [[1, 9], [11, 3]],
    [[8, 2], [1, 5]],
    [[0, 5], [5, 0]],
    [[10, 0], [0, 10]],
    [[40, 10], [30, 20]],
    [[0, 0], [5, 5]],      # zero row total -> 1.0
    [[25, 25], [25, 25]],
]


def test_matches_paid_exactly():
    for t in TABLES:
        for free_fn, paid_fn in (
            (barnard_exact.barnard_test, paid.barnard_test),
            (barnard_exact.boschloo_test, paid.boschloo_test),
        ):
            assert free_fn(t) == pytest.approx(paid_fn(t), rel=1e-9), t


def test_matches_paid_on_random_tables():
    rng = np.random.default_rng(3)
    for _ in range(15):
        t = rng.integers(0, 40, size=(2, 2)).tolist()
        assert barnard_exact.barnard_test(t) == pytest.approx(
            paid.barnard_test(t), rel=1e-9
        )
        assert barnard_exact.boschloo_test(t) == pytest.approx(
            paid.boschloo_test(t), rel=1e-9
        )


def test_over_cap_refuses_loudly():
    big = [[61, 40], [55, 45]]  # N = 201
    for fn in (barnard_exact.barnard_test, barnard_exact.boschloo_test):
        with pytest.raises(ValueError, match="total N <= 200"):
            fn(big)
        try:
            fn(big)
        except ValueError as e:
            assert PAID_URL in str(e)
            assert "30" in str(e)


def test_at_cap_still_works():
    t = [[60, 40], [55, 45]]  # N = 200
    assert barnard_exact.barnard_test(t) == pytest.approx(
        paid.barnard_test(t), rel=1e-9
    )
    assert barnard_exact.boschloo_test(t) == pytest.approx(
        paid.boschloo_test(t), rel=1e-9
    )


@pytest.mark.parametrize(
    "bad",
    [
        [[1.5, 2], [3, 4]],      # fractional
        [[-1, 2], [3, 4]],       # negative
        [[1, 2, 3], [4, 5, 6]],  # wrong shape
        [[1, 2]],                # not 2x2
        [[float("nan"), 2], [3, 4]],
        [[float("inf"), 2], [3, 4]],
        [[True, False], [True, True]],  # booleans
        "not a table",
    ],
)
def test_garbage_raises(bad):
    for fn in (barnard_exact.barnard_test, barnard_exact.boschloo_test):
        with pytest.raises(ValueError):
            fn(bad)


def test_paid_only_symbols_absent():
    assert set(barnard_exact.__all__) == {
        "barnard_test",
        "boschloo_test",
        "FREE_MAX_N",
        "__version__",
    }
    for name in ("_barnard_onesided", "_boschloo_onesided", "_fisher_twosided"):
        assert not hasattr(barnard_exact, name), name


def test_free_cap_constant():
    assert barnard_exact.FREE_MAX_N == 200


def test_pvalues_in_unit_interval():
    rng = np.random.default_rng(9)
    for _ in range(15):
        t = rng.integers(0, 30, size=(2, 2)).tolist()
        for fn in (barnard_exact.barnard_test, barnard_exact.boschloo_test):
            p = fn(t)
            assert 0.0 <= p <= 1.0
