# Correction Note

A localized row-alignment error affected the assembly of the hosted-LLM result table (Table 3 in the submitted version) and the corresponding hosted panel of Figure 3. The error affected the aggregation and presentation of hosted-model outputs; it did not affect the dataset, labels, canonical run-level splits, predictions, or evaluation protocol.

At the default operating point, the corrected charged Net values are:

| Configuration | Charged Net (s) |
| --- | ---: |
| Claude weak | 2.07 |
| Claude strong | -8.41 |
| GPT weak | -11.81 |
| GPT strong | -2.43 |
| DeepSeek weak | -7.03 |
| DeepSeek strong | -8.11 |

These values replace the corresponding hosted entries in the main and appendix tables and the plotted hosted points in Figure 3. Related supplementary per-dataset and transfer-control cells were updated to match the frozen evaluation summaries.

The released dataset, success labels, canonical run-level splits, supervised headline point estimates, Qwen headline point estimates, and label-budget analysis are unchanged. At the default operating point, Claude weak is the only hosted configuration with positive charged utility, while the strongest supervised references remain substantially higher: 61.22 seconds for HGB and 67.12 seconds for Qwen. The corrected results preserve the conclusion that hosted models retain predictive signal but do not reliably convert it into cost-effective intervention under validation-only false-stop control and request-latency accounting.

The calibration controls fit Platt and isotonic calibrators on validation probabilities, select the stop threshold on validation under the stated false-stop cap, and evaluate the frozen policy on test. Platt leaves four hosted configurations' test decisions unchanged and changes both GPT configurations. Isotonic changes charged Net by less than 5 seconds across all six configurations. The corrected hosted tables and Figure 3 can be regenerated from the released hosted outputs and canonical split files under the evaluation protocol specified in the paper.

Bootstrap intervals are also revised to use common test-run resampling across the three Qwen model seeds, preserving dependence induced by their shared test set. The cross-node 95% interval is [45.81, 102.53] seconds, and the S130-IBEX interval is [108.99, 613.02]; their point estimates remain 67.12 and 286.03. Raw-direct hosted intervals use the same run-cluster convention without request-latency subtraction. The paired DeepSeek-weak minus Claude-weak interval on S130-IBEX includes zero. This statistical revision is separate from the table-assembly correction and does not change the reported point-estimate rankings.
