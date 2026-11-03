# Reproducing the reported evaluations

Run these commands from the repository root using Python 3.11. Install the optional offline dependencies with `python -m pip install -r requirements-replay.txt`. The experiment runners have additional model dependencies listed in `requirements.txt`.

## Hosted table, direct actions, and calibration

```sh
python scripts/reproduction/recompute_hosted.py --out reproduction_outputs/hosted
```

Fits Platt (`LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000, tol=1e-10)`) or isotonic (`out_of_bounds="clip"`) on validation only, selects a threshold on the exact i/100 grid for i=0..95 under the validation FSR cap, then freezes it on test. One-class validation data use constant probabilities. Fixed latency charges come from the measured successful-request latency table. The script asserts the default hosted cells and the calibration reference results. Direct rows use dataset-macro aggregation and include latency; `net_noapi` exposes the uncharged diagnostic.

## Run-cluster bootstrap

```sh
python scripts/reproduction/recompute_bootstrap.py --root . --out reproduction_outputs/bootstrap
```

Uses 10,000 percentile replicates, primary bootstrap seed 20261029, shared run multiplicities across three fixed model seeds and paired methods. Draws retain all prefixes and compute utility sum divided by prefix count, followed by equal dataset weights. This is conditional on frozen fitted models, thresholds and costs; it excludes refitting uncertainty. Alternative sampling modes and seeds in the CSV are diagnostic controls; the paper uses `shared_run`, seed 20261029. Hosted raw-direct intervals omit latency.

## Two-sided continue cost

```sh
python scripts/reproduction/recompute_continue_cost.py --out reproduction_outputs/continue_cost
```

Keeps the default lambda=1, mu=0 validation-selected decisions frozen and scores mu=0, .25, .5, 1. Tests the saved Qwen mean results and reports best hosted utility at each mu. A negative two-sided value does not by itself establish inferiority to never-stop.

## Sensitivity grid

```sh
python scripts/reproduction/aggregate_grid.py --out reproduction_outputs/grid
```

Aggregates saved tabular per-fold metrics and frozen Qwen grid cells, verifies all 64 cells and writes the table sources. To train the tabular models with the released data, set `PACKAGE_ROOT` then run the driver, one model followed by its 16 cells:

```sh
python scripts/reproduction/run_supervised_lambda_cap_sensitivity.py --model hgb
python scripts/reproduction/run_supervised_lambda_cap_sensitivity.py --model-lambda hgb --lambda 1 --cap 0.10
```

Repeat the second command for lambda=.5,1,2,5 and cap=.01,.05,.10,.20, and repeat for models xgboost and logreg. Then run `--finalize`. The harness requires NumPy, scikit-learn and XGBoost and exact canonical source/target splits. 

## Figure and saved transfer results

Figure 3 sources are in `figures/`. Run `python figures/generate_capability_boundary.py`, then compile `figures/fig_capability_boundary.tex` with pdflatex. The generator renders the included input CSV.

`reference_results/lodfo_per_fold.csv` and `lodfo_summary.csv` provide saved transfer results. These files provide the transfer-table inputs. LODFO model training and fine-tuned checkpoints are outside this package.

## Dataset metadata

The release manifest records the SHA-256 of the uploaded inner `TraceTriage.zip`. Kaggle may wrap this file in a separate download archive.
