"""barnard_exact: Barnard's and Boschloo's exact tests for 2x2 tables (free core)."""

from ._exact import FREE_MAX_N, barnard_test, boschloo_test

__version__ = "0.1.0"

__all__ = ["barnard_test", "boschloo_test", "FREE_MAX_N", "__version__"]
