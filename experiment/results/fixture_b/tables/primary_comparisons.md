Pre-specified primary comparisons: Complete@K, nominal 2,048-token budget, paper-level.
Diff = mean over papers of (A − B). CI: paired percentile bootstrap over papers. p: two-sided paired sign-flip permutation test. Holm across all 8 tests.

| A | B | Mean A | Mean B | Diff (95% CI) | p (raw) | p (Holm) | Papers A>B / A<B / = | n papers | Test |
|---|---|---:|---:|---|---:|---:|---|---:|---|
| s8_o10 | s8_o0 | 0.000 | 0.000 | +0.000 (+0.000 to +0.000) | 1.0000 | 1.0000 | 0 / 0 / 6 | 6 | exact (64 sign vectors) |
| s8_o20 | s8_o0 | 0.000 | 0.000 | +0.000 (+0.000 to +0.000) | 1.0000 | 1.0000 | 0 / 0 / 6 | 6 | exact (64 sign vectors) |
| s16_o10 | s16_o0 | 0.667 | 0.000 | +0.667 (+0.667 to +0.667) | 0.0312 | 0.2500 | 6 / 0 / 0 | 6 | exact (64 sign vectors) |
| s16_o20 | s16_o0 | 0.222 | 0.000 | +0.222 (+0.111 to +0.333) | 0.1250 | 0.7500 | 4 / 0 / 2 | 6 | exact (64 sign vectors) |
| s32_o10 | s32_o0 | 0.333 | 0.278 | +0.056 (-0.111 to +0.222) | 1.0000 | 1.0000 | 2 / 1 / 3 | 6 | exact (64 sign vectors) |
| s32_o20 | s32_o0 | 0.556 | 0.278 | +0.278 (+0.111 to +0.444) | 0.1250 | 0.7500 | 4 / 0 / 2 | 6 | exact (64 sign vectors) |
| s8_o0 | s16_o0 | 0.000 | 0.000 | +0.000 (+0.000 to +0.000) | 1.0000 | 1.0000 | 0 / 0 / 6 | 6 | exact (64 sign vectors) |
| s16_o0 | s32_o0 | 0.000 | 0.278 | -0.278 (-0.333 to -0.167) | 0.0625 | 0.4375 | 0 / 5 / 1 | 6 | exact (64 sign vectors) |

A p-value above 0.05 is not evidence that two configurations are equivalent.
