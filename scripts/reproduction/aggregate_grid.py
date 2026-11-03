#!/usr/bin/env python3
"""Regenerate the grid from saved per-fold tabular metrics and frozen Qwen cells.

This entry point aggregates saved scores; it does not refit tabular models or
repeat Qwen threshold selection. Use the supervised runner for tabular refits.
"""
import argparse,csv,statistics
from pathlib import Path
import run_supervised_lambda_cap_sensitivity as grid

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 root=Path(__file__).resolve().parents[2]
 with (root/'reference_results/tabular_lambda_cap_per_fold.csv').open() as f:folds=list(csv.DictReader(f))
 combined=grid.aggregate(folds)+grid.qwen_rows()
 expected={(r['model'],float(r['lambda']),float(r['cap'])):r for r in grid.read_csv(root/'reference_results/supervised_lambda_cap_grid.csv')}
 assert len(combined)==64
 for r in combined:
  e=expected[r['model'],r['lambda'],r['cap']]
  for k in ['net_mean','net_std','frs_mean','frs_std','fsr_mean','fsr_std']:assert abs(r[k]-float(e[k]))<1e-7,(r,k,e)
 grid.write_csv(a.out/'supervised_lambda_cap_grid.csv',combined)
 grid.write_csv(a.out/'gate_check.csv',grid.gate_rows(combined,[]))
 grid.write_csv(a.out/'inversion_check.csv',grid.inversion_rows(combined))
 print('64 cells match saved grid; tabular aggregation and frozen Qwen cells checked.')
if __name__=='__main__':main()
