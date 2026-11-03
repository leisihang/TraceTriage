# Reported results

All entries use canonical split files and validation-only threshold selection.

| Model / protocol | Net seconds (mean ± model-seed SD) |
| --- | ---: |
| Qwen single-domain | 68.75 ± 0.21 |
| Qwen LODO | 68.82 ± 0.11 |
| Qwen cross-node LODO | 67.12 ± 3.04 |
| HGB cross-node | 61.22 ± 13.26 |
| XGBoost cross-node | 24.24 ± 0.20 |
| Logistic regression cross-node | 20.40 ± 0.01 |

Hosted charged macro Net: Claude weak 2.07, strong −8.41; GPT weak −11.81, strong −2.43; DeepSeek weak −7.03, strong −8.11.

The Qwen cross-node run-cluster bootstrap 95% interval is [45.81, 102.53]. The model-seed SD and this conditional test-run interval describe different sources of variability.

See `REPRODUCING.md` for executable scoring and table-generation commands.
