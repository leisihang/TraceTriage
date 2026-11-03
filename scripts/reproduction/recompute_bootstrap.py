"""Run-cluster bootstrap for frozen evaluation predictions.

Usage: python scripts/reproduction/recompute_bootstrap.py --root . --out reproduction_outputs/bootstrap
Requires NumPy. Reads flat, sanitized frozen prediction files.
"""
import argparse
import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np


def rows(stream):
    return list(csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8")))


def rng_for(seed, dataset, mode):
    key = f"{seed}|{dataset}|{mode}".encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "little"))


def boot(sums, counts, rng, n):
    # sums: independent runs x model seeds. Resampling keeps whole runs,
    # including their prefix multiplicities. The ratio preserves prefix-mean Net.
    out = np.empty((n, sums.shape[1]))
    for start in range(0, n, 256):
        size = min(256, n-start)
        idx = rng.integers(0, len(counts), size=(size, len(counts)))
        out[start:start+size] = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)[:, None]
    return out


def cluster(keys, values):
    ids, inverse = np.unique(keys, return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.column_stack([
        np.bincount(inverse, weights=values[:, j]) for j in range(values.shape[1])
    ])
    return ids, sums, counts


def interval(target, mode, seed, point, samples):
    lo, hi = np.quantile(samples, [.025, .975])
    return dict(target=target, method=mode, bootstrap_seed=seed,
                point_net=float(point), ci95_low=float(lo), ci95_high=float(hi),
                bootstrap_sd=float(np.std(samples, ddof=1)), replicates=len(samples))


def load(root):
    inv = rows((root / "reference_results/qwen_fold_inventory.csv").open("rb"))
    inv = {(x["dataset"], int(x["seed"])): x for x in inv if x["regime"] == "crossnode"}
    data, gate, input_hashes = {}, [], {}
    for folder in sorted((root / "reproduction_inputs/truth").iterdir()):
        dataset = folder.name
        path = folder / "test.csv"
        raw = path.read_bytes()
        input_hashes[str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
        truth = sorted(csv.DictReader(io.StringIO(raw.decode())), key=lambda x:x["prefix_id"])
        prefix_ids = [x["prefix_id"] for x in truth]
        assert len(set(prefix_ids)) == len(prefix_ids)
        labels = np.array([int(x["strict_success"]) for x in truth])
        remaining = np.array([float(x["remaining_seconds"]) for x in truth])
        runkeys = np.array([x["campaign_id"]+":"+x["run_id"] for x in truth])
        assert all(x["record_key"] == k for x,k in zip(truth,runkeys))
        assert np.allclose(remaining,[max(0,float(x["final_wallclock_seconds"])-float(x["prefix_elapsed_seconds"])) for x in truth])
        values,costs = [],[]
        for model_seed in (42,123,3407):
            base = root / "reproduction_inputs/qwen" / f"seed{model_seed}" / dataset
            summary = json.loads((base/"metrics_summary.json").read_text())
            setting = next(x for x in summary["lambda_sweep"] if x["lambda_false_stop"] == 1)
            tau,cost = setting["stop_threshold"],setting["inference_cost_test"]
            pred = sorted(rows((base/"predictions.csv").open("rb")),key=lambda x:x["prefix_id"])
            assert [x["prefix_id"] for x in pred] == prefix_ids
            assert np.array_equal([int(x["strict_success"]) for x in pred],labels)
            assert np.allclose([float(x["remaining_seconds"]) for x in pred],remaining)
            stop = np.array([float(x["success_prob"]) < tau for x in pred])
            u = stop*(1-2*labels)*remaining
            net = float(u.mean()-cost)
            expected = float(inv[(dataset,model_seed)]["net"])
            assert abs(net-setting["net_s"]) < 1e-7
            assert abs(net-expected) < 1e-7
            assert abs(tau-float(inv[(dataset,model_seed)]["frozen_threshold"])) < 1e-12
            gate.append(dict(dataset=dataset,seed=model_seed,prefixes=len(truth),runs=len(set(runkeys)),threshold=tau,cost=cost,point=net,reference=expected,difference=net-expected))
            values.append(u);costs.append(cost)
        u = np.column_stack(values)
        ids,sums,counts = cluster(runkeys,u)
        data[dataset] = dict(values=u,sums=sums,counts=counts,costs=np.array(costs),truth=truth,ids=ids,runkeys=runkeys,point=float((u.mean(axis=0)-costs).mean()))
    historical = []
    return data, gate, historical, input_hashes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--replicates", type=int, default=10000)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data, gate, historical, hashes = load(args.root)
    result = []
    macro_point = np.mean([d["point"] for d in data.values()])
    for mcseed in (20261029, 42, 123):
        for mode in ("shared_run", "independent_seed_run", "shared_prefix"):
            macro = np.zeros(args.replicates)
            for dataset, d in data.items():
                rng = rng_for(mcseed, dataset, mode)
                if mode == "independent_seed_run":
                    b = np.column_stack([boot(d["sums"][:, j:j+1], d["counts"], rng,
                                              args.replicates)[:, 0] for j in range(3)])
                elif mode == "shared_prefix":
                    b = boot(d["values"], np.ones(len(d["values"])), rng, args.replicates)
                else:
                    b = boot(d["sums"], d["counts"], rng, args.replicates)
                values = (b-d["costs"]).mean(axis=1)
                macro += values/len(data)
                if dataset == "sky130hd_ibex_core":
                    result.append(interval("Qwen S130-IBEX", mode, mcseed, d["point"], values))
            result.append(interval("Qwen cross-node macro", mode, mcseed, macro_point, macro))
    dataset = "sky130hd_ibex_core"
    d = data[dataset]
    ids = [x["prefix_id"] for x in d["truth"]]
    y = np.array([int(x["strict_success"]) for x in d["truth"]])
    rem = np.array([float(x["remaining_seconds"]) for x in d["truth"]])
    hosted = []
    for name in ("claude_opus47_weak", "deepseek_v4_flash_weak"):
        path = args.root / "reproduction_inputs/hosted" / name / dataset / "test.csv"
        raw = path.read_bytes()
        hashes[str(path.relative_to(args.root))] = hashlib.sha256(raw).hexdigest()
        pred = sorted(csv.DictReader(io.StringIO(raw.decode())), key=lambda x: x["prefix_id"])
        assert [x["prefix_id"] for x in pred] == ids
        assert np.array_equal([int(x["strict_success"]) for x in pred], y)
        stop = np.array([int(x["pred_stop"]) for x in pred])
        assert np.array_equal(stop, [x["decision"] == "STOP" for x in pred])
        hosted.append(stop*(1-2*y)*rem)
    _, sums, counts = cluster(d["runkeys"], np.column_stack(hosted))
    for mcseed in (20261029, 42, 123):
        b = boot(sums, counts, rng_for(mcseed, dataset, "hosted_shared_run"), args.replicates)
        for j, target in enumerate(("Claude-w raw direct", "DeepSeek-w raw direct")):
            result.append(interval(target, "shared_run_no_latency", mcseed, np.mean(hosted[j]), b[:, j]))
        result.append(interval("DeepSeek-w minus Claude-w raw direct", "paired_run_no_latency",
                               mcseed, np.mean(hosted[1]-hosted[0]), b[:, 1]-b[:, 0]))
    for name, records in (("intervals.csv", result), ("point_gate.csv", gate)):
        with (args.out/name).open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    metadata = dict(numpy_version=np.__version__, datasets=len(data),
                    prefixes=sum(len(d["truth"]) for d in data.values()),
                    unique_runs=sum(len(d["counts"]) for d in data.values()),
                    model_seeds=[42,123,3407], bootstrap_seeds=[20261029,42,123],
                    replicates=args.replicates, macro_point=float(macro_point),
                    historical=historical, sha256=hashes,
                    estimand="Equal-dataset macro of three fixed policies' prefix-mean Net; conditional on fitted policies.",
                    algorithm="PCG64; SHA256-derived deterministic streams; dataset and prefix IDs sorted; "
                              "whole campaign:run clusters; ratio of summed utilities to summed prefix counts; "
                              "default linear percentile interpolation; lambda=1, mu=0; fixed local inference cost.",
                    shared_run="Same sampled run multiplicities across all model seeds; datasets sampled independently.",
                    independent_seed_run="Diagnostic: independently sample test runs for each model seed.",
                    shared_prefix="Diagnostic: independently sample prefixes, shared across model seeds.")
    (args.out/"metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps({k:metadata[k] for k in ("macro_point","prefixes","unique_runs")}))
    for r in result:
        print(r["bootstrap_seed"], r["method"], r["target"],
              f'{r["point_net"]:.5f} [{r["ci95_low"]:.5f}, {r["ci95_high"]:.5f}]')


if __name__ == "__main__":
    main()
