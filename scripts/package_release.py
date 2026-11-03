#!/usr/bin/env python3
"""Create a portable ZIP without .git, credentials, caches or generated outputs."""
from pathlib import Path
import argparse,hashlib,zipfile
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path('TraceTriage-camera-ready.zip'));a=ap.parse_args()
root=Path(__file__).resolve().parents[1]
with zipfile.ZipFile(a.out,'w',compression=zipfile.ZIP_DEFLATED) as z:
 for p in sorted(root.rglob('*')):
  if not p.is_file() or p.resolve()==a.out.resolve():continue
  rel=p.relative_to(root)
  if any(x in {'.git','__pycache__','reproduction_outputs','.env'} for x in rel.parts):continue
  if p.name in {'kaggle.json'} or p.suffix in {'.pyc','.pem','.zip'}:continue
  z.write(p,Path('TraceTriage')/rel)
print(a.out,hashlib.sha256(a.out.read_bytes()).hexdigest())
