# TraceTriage Responsible AI Metadata

This document describes intended uses, limitations, provenance and privacy measures for the TraceTriage dataset.

## Data Limitations

TraceTriage is designed for EDA flow triage and stop/continue prediction in
OpenROAD-based digital design runs. The dataset should not be treated as a
general-purpose representation of all EDA tools, all process technologies,
all design styles, or all commercial physical design flows.

Known limitations:

```text
The data focuses on two technology/platform families represented in the package: asap7 and sky130hd.
The design set is limited to five open benchmark designs per platform.
The labels are derived from the packaged flow outcomes and strict success criteria.
The package is intended for research on flow triage, early stopping, and dataset/benchmark methodology.
The dataset is not recommended for signoff-quality EDA decisions, commercial tapeout decisions, or safety-critical automation without additional validation.
```

## Data Biases

Potential biases include:

```text
Design selection bias from the included benchmark circuits.
Technology selection bias from the included platform families.
Toolchain bias from OpenROAD-generated logs and reports.
Configuration bias from the sweep parameters represented in the dataset.
Outcome-label bias from the chosen strict success and actionable labels.
```

Models trained on TraceTriage may overfit to OpenROAD log/report conventions,
the included designs, or the included platforms. Cross-design and cross-node
evaluation should be used when assessing generalization.

## Personal Or Sensitive Information

The dataset is not intended to contain personal, demographic, health,
political, religious, or socioeconomic information. It contains EDA logs,
reports, configuration values, derived metrics, and labels from design-flow
runs.

Privacy sanitization applied in the package:

```text
Private absolute paths are removed from public manifests.
Manifest log paths are package-relative under logs/openroad/.
history_csv and task_dir are blanked.
API endpoints, API base URLs, and response identifiers are redacted in package-local API output metadata.
No API keys are included.
```

## Data Use Cases

Intended use cases:

```text
Benchmarking early stop/continue decisions in EDA flows.
Training and evaluating supervised baselines for flow triage.
Evaluating local language models on EDA log/report triage tasks.
Evaluating API-based LLM triage methods when users supply their own provider credentials.
Studying cross-design and cross-platform generalization in EDA prediction tasks.
```

Use cases not established by this dataset alone:

```text
Generalizing to all EDA tools or all process nodes.
Replacing signoff verification.
Making commercial tapeout decisions.
Assessing human subjects or demographic fairness.
Inferring proprietary design quality or commercial PDK behavior.
```

## Social Impact

Potential positive impacts:

```text
Reducing wasted compute in EDA exploration.
Improving reproducible evaluation of EDA triage methods.
Lowering the barrier to studying ML-assisted physical design workflows.
Encouraging transparent benchmark construction for EDA prediction tasks.
```

Potential negative impacts and mitigations:

```text
Models may learn brittle heuristics tied to the included tools/designs; mitigation: report cross-design and cross-platform results.
Incorrect stop/continue predictions can waste engineering effort or hide failing runs; mitigation: use as decision support, not signoff authority.
Large reports may impose storage and compute costs; mitigation: include smoke workflows and clear manifests.
```

## Synthetic Data

The dataset consists of generated EDA flow artifacts, logs, reports, metrics,
and labels from tool runs. It does not contain synthetic people or human
subjects data.

Recommended Croissant value:

```text
rai:hasSyntheticData = false
```

The workflow configurations are generated experimental conditions; the observations are tool execution outputs.

## Source Datasets

TraceTriage is derived from OpenROAD-flow-scripts executions using ASAP7 and SkyWater 130nm platforms and the AES, GCD, IBEX, JPEG and PicoRV32 design families. Upstream resources include [OpenROAD](https://github.com/The-OpenROAD-Project/OpenROAD) and [OpenROAD-flow-scripts](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts).

```text
prov:wasDerivedFrom = OpenROAD and OpenROAD-flow-scripts executions; see the dataset documentation for upstream design/platform licensing.
```

## Provenance Activities

Collection and preprocessing summary:

```text
EDA runs were collected from OpenROAD flow executions over the included design/platform pairs.
Prefix-level examples were generated from run logs and flow checkpoints.
Labels were derived from final flow status and strict success/actionability criteria.
Train, validation and test membership is defined by the packaged split CSV files, grouped by completed run.
Private paths were removed or blanked.
Referenced logs and reports were materialized inside the package.
Verification checks confirmed dataset names, row counts, split consistency, package-relative paths, and materialized log presence.
```

Annotation summary:

```text
Labels are rule-derived from run outcomes and packaged metrics rather than human demographic annotations.
No annotator demographics or human-subject labels are involved.
```
