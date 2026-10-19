Pre-specified primary comparisons: complete, mode `primary` ({'type': 'token_budget', 'budget': 2048}), paper-level.
Diff = mean over papers of (A − B). CI: paired percentile bootstrap over papers. p: two-sided paired sign-flip permutation test. Holm across all 8 tests.

| A | B | Mean A | Mean B | Diff (95% CI) | p (raw) | p (Holm) | Papers A>B / A<B / = | n papers | Test |
|---|---|---:|---:|---|---:|---:|---|---:|---|
| s128_o10 | s128_o0 | 0.350 | 0.323 | +0.026 (-0.042 to +0.097) | 0.4697 | 0.4697 | 22 / 19 / 59 | 100 | Monte Carlo (100000 sign vectors) |
| s128_o20 | s128_o0 | 0.411 | 0.323 | +0.087 (+0.018 to +0.156) | 0.0156 | 0.1095 | 30 / 15 / 55 | 100 | Monte Carlo (100000 sign vectors) |
| s256_o10 | s256_o0 | 0.552 | 0.482 | +0.070 (-0.003 to +0.145) | 0.0666 | 0.2604 | 29 / 20 / 51 | 100 | Monte Carlo (100000 sign vectors) |
| s256_o20 | s256_o0 | 0.564 | 0.482 | +0.083 (+0.014 to +0.151) | 0.0212 | 0.1270 | 28 / 19 / 53 | 100 | Monte Carlo (100000 sign vectors) |
| s512_o10 | s512_o0 | 0.592 | 0.543 | +0.049 (-0.022 to +0.119) | 0.1863 | 0.3725 | 30 / 24 / 46 | 100 | Monte Carlo (100000 sign vectors) |
| s512_o20 | s512_o0 | 0.614 | 0.543 | +0.071 (+0.003 to +0.137) | 0.0455 | 0.2276 | 31 / 15 / 54 | 100 | Monte Carlo (100000 sign vectors) |
| s128_o0 | s256_o0 | 0.323 | 0.482 | -0.159 (-0.229 to -0.088) | 0.0000 | 0.0002 | 8 / 43 / 49 | 100 | Monte Carlo (100000 sign vectors) |
| s256_o0 | s512_o0 | 0.482 | 0.543 | -0.061 (-0.126 to +0.002) | 0.0651 | 0.2604 | 18 / 27 / 55 | 100 | Monte Carlo (100000 sign vectors) |

A p-value above 0.05 is not evidence that two configurations are equivalent.
