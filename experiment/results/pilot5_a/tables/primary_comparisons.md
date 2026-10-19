Pre-specified primary comparisons: complete, mode `primary` ({'type': 'token_budget', 'budget': 2048}), paper-level.
Diff = mean over papers of (A − B). CI: paired percentile bootstrap over papers. p: two-sided paired sign-flip permutation test. Holm across all 8 tests.

| A | B | Mean A | Mean B | Diff (95% CI) | p (raw) | p (Holm) | Papers A>B / A<B / = | n papers | Test |
|---|---|---:|---:|---|---:|---:|---|---:|---|
| s128_o10 | s128_o0 | 0.317 | 0.267 | +0.050 (+0.000 to +0.150) | 1.0000 | 1.0000 | 1 / 0 / 4 | 5 | exact (32 sign vectors) |
| s128_o20 | s128_o0 | 0.383 | 0.267 | +0.117 (+0.000 to +0.250) | 0.5000 | 1.0000 | 2 / 0 / 3 | 5 | exact (32 sign vectors) |
| s256_o10 | s256_o0 | 0.583 | 0.450 | +0.133 (-0.200 to +0.600) | 1.0000 | 1.0000 | 1 / 1 / 3 | 5 | exact (32 sign vectors) |
| s256_o20 | s256_o0 | 0.617 | 0.450 | +0.167 (+0.000 to +0.367) | 0.5000 | 1.0000 | 2 / 0 / 3 | 5 | exact (32 sign vectors) |
| s512_o10 | s512_o0 | 0.767 | 0.567 | +0.200 (+0.000 to +0.600) | 1.0000 | 1.0000 | 1 / 0 / 4 | 5 | exact (32 sign vectors) |
| s512_o20 | s512_o0 | 0.483 | 0.567 | -0.083 (-0.683 to +0.517) | 1.0000 | 1.0000 | 2 / 2 / 1 | 5 | exact (32 sign vectors) |
| s128_o0 | s256_o0 | 0.267 | 0.450 | -0.183 (-0.317 to -0.050) | 0.2500 | 1.0000 | 0 / 3 / 2 | 5 | exact (32 sign vectors) |
| s256_o0 | s512_o0 | 0.450 | 0.567 | -0.117 (-0.250 to +0.000) | 0.5000 | 1.0000 | 0 / 2 / 3 | 5 | exact (32 sign vectors) |

A p-value above 0.05 is not evidence that two configurations are equivalent.
