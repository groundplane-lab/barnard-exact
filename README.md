# barnard-exact

Free core of Barnard: Barnard's and Boschloo's exact tests for 2x2 tables.

**The problem**
You ran a small 2x2 experiment and Fisher's exact test came back just above your threshold. Fisher's test conditions on the total number of successes, which your experiment never fixed in advance, and that conditioning throws away power. scipy does not implement the unconditional alternatives. Search terms: Barnard test Python pip, Boschloo test Python, Barnard exact test 2x2.

**What it does**
`barnard_test(table)` and `boschloo_test(table)` compute the exact unconditional two-sided p-values, maximized over the nuisance parameter, with ties classified in exact integer arithmetic. Boschloo's never returns a larger p-value than Fisher's exact test. Both are validated against published reference values. The free version handles total N up to 200.

**Limits**
Free version: total N <= 200. Larger tables are refused outright instead of hanging. No CLI, no plots, no batch mode.

**Install**

```bash
pip install barnard-exact
```

Needs Python 3.9 or newer, NumPy, and SciPy.

```python
from barnard_exact import barnard_test, boschloo_test
print(barnard_test([[7, 12], [3, 18]]))
print(boschloo_test([[7, 12], [3, 18]]))
```

This is the free core: Barnard's and Boschloo's exact tests for 2x2 tables, capped at total N <= 200. The full version ($30, one-time) is uncapped: https://groundplane0.gumroad.com/l/conxfk
