"""Export numerical table bodies from released evaluation CSVs.

Prediction replay and training commands are documented in REPRODUCING.md.
This exporter formats saved results; it does not fit models or select policies.
"""
import argparse
import csv
from pathlib import Path


def escape(value):
    value = str(value)
    for char, repl in [('\\', r'\textbackslash{}'), ('&', r'\&'), ('%', r'\%'), ('_', r'\_'), ('#', r'\#')]:
        value = value.replace(char, repl)
    return value


def render(path, out):
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    if not fields:
        return
    lines = [r'\begin{tabular}{' + 'l' * len(fields) + '}', r'\toprule',
             ' & '.join(escape(k) for k in fields) + r' \\', r'\midrule']
    lines += [' & '.join(escape(row[k]) for k in fields) + r' \\' for row in rows]
    lines += [r'\bottomrule', r'\end{tabular}']
    out.write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sources = sorted((args.root / 'reference_results').glob('*.csv'))
    sources += sorted((args.root / 'verified_results').glob('*/*.csv'))
    for path in sources:
        prefix = path.parent.name + '_' if path.parent.name != 'reference_results' else ''
        render(path, args.out / (prefix + path.stem + '.tex'))
    print(f'Exported {len(sources)} numerical table sources to {args.out}')


if __name__ == '__main__':
    main()
