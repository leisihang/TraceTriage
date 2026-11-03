#!/usr/bin/env python3
"""Generate Figure 3 from fig_capability_boundary_data.csv."""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "fig_capability_boundary_data.csv"
OUT = ROOT / "fig_capability_boundary.tex"


def read_rows() -> dict[str, dict[str, str]]:
    with DATA.open(newline="", encoding="utf-8") as f:
        return {row["label"]: row for row in csv.DictReader(f)}


def coord(row: dict[str, str]) -> str:
    return f"({float(row['frs']):.4f},{float(row['net_charged_s']):.2f})"


def line(points: list[str], color: str) -> str:
    return f"\\addplot+[no marks, color={color}!58, line width=0.65pt] coordinates {{{' '.join(points)}}};"


def marker(row: dict[str, str], color: str, shape: str, fill: str, size: str) -> str:
    draw = "black!65" if fill == color else color
    return (
        f"\\addplot+[only marks, mark={shape}, mark size={size}, color={color}, "
        f"fill={fill}, mark options={{line width=0.45pt, draw={draw}}}] coordinates {{{coord(row)}}};"
    )


def label(text: str, x: float, y: float, anchor: str = "west", color: str = "textC") -> str:
    return (
        f"\\node[font=\\scriptsize, anchor={anchor}, text={color}] "
        f"at (axis cs:{x:.3f},{y:.2f}) {{{text}}};"
    )


def generate(rows: dict[str, dict[str, str]]) -> str:
    colors = {"Qwen": "qwenC", "LogReg": "logregC", "XGBoost": "xgbC", "HGB": "hgbC"}
    family_order = ["Qwen", "LogReg", "XGBoost", "HGB"]
    regimes = ["single-domain", "LODO", "cross-node LODO"]
    regime_markers = {"single-domain": ("*", "3.5pt", "filled"), "LODO": ("triangle*", "4.0pt", "open"), "cross-node LODO": ("square*", "3.4pt", "filled")}
    regime_rows: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows.values():
        model = row["label"].split()[0]
        if model in colors and row["protocol"] in regimes:
            regime_rows[(model, row["protocol"])] = row

    out: list[str] = [
        r"\documentclass[tikz,border=2pt]{standalone}",
        r"\usepackage{pgfplots}",
        r"\usetikzlibrary{shapes.geometric}",
        r"\usepgfplotslibrary{groupplots}",
        r"\pgfplotsset{compat=1.18}",
        r"\definecolor{qwenC}{HTML}{009E73}",
        r"\definecolor{logregC}{HTML}{0072B2}",
        r"\definecolor{xgbC}{HTML}{D55E00}",
        r"\definecolor{hgbC}{HTML}{CC79A7}",
        r"\definecolor{claudeC}{HTML}{CC79A7}",
        r"\definecolor{gptC}{HTML}{D55E00}",
        r"\definecolor{deepseekC}{HTML}{0072B2}",
        r"\definecolor{zeroC}{HTML}{111111}",
        r"\definecolor{gridC}{HTML}{D9D9D9}",
        r"\definecolor{textC}{HTML}{333333}",
        r"\begin{document}",
        r"\begin{tikzpicture}",
        r"\begin{groupplot}[",
        r"  group style={group name=capbd, group size=2 by 1, horizontal sep=0.92cm},",
        r"  width=7.05cm,",
        r"  height=5.35cm,",
        r"  xlabel={Stop-worthy failure recall, FRS},",
        r"  ylabel={Macro net utility (s)},",
        r"  xlabel style={font=\small},",
        r"  ylabel style={font=\small},",
        r"  tick label style={font=\scriptsize},",
        r"  title style={font=\small\bfseries, yshift=-0.5ex},",
        r"  scaled ticks=false,",
        r"  axis line style={black!70, line width=0.35pt},",
        r"  tick style={black!65, line width=0.35pt},",
        r"  ymajorgrids,",
        r"  xmajorgrids,",
        r"  grid style={gridC, line width=0.25pt},",
        r"  clip marker paths=true,",
        r"]",
        r"\nextgroupplot[",
        r"  title={A. Supervised transfer boundary},",
        r"  xmin=0.32,",
        r"  xmax=1.01,",
        r"  ymin=10,",
        r"  ymax=122,",
        r"  xtick={0.4,0.6,0.8,1.0},",
        r"  ytick={20,50,80,110},",
        r"]",
    ]

    for family in family_order:
        color = colors[family]
        out.append(line([coord(regime_rows[(family, regime)]) for regime in regimes], color))
    for family in family_order:
        color = colors[family]
        for regime in regimes:
            shape, size, _ = regime_markers[regime]
            fill = color if regime != "LODO" else "white"
            draw = color if regime == "LODO" else "black!65"
            row = regime_rows[(family, regime)]
            out.append(
                f"\\addplot+[only marks, mark={shape}, mark size={size}, color={color}, fill={fill}, "
                f"mark options={{line width={0.85 if regime == 'LODO' else 0.45:.2f}pt, draw={draw}}}] coordinates {{{coord(row)}}};"
            )

    out += [
        r"\draw[qwenC, line width=0.9pt] (axis cs:0.35,116) -- (axis cs:0.405,116);",
        r"\node[font=\scriptsize, anchor=west, text=textC] at (axis cs:0.420,116) {Qwen};",
        r"\draw[logregC, line width=0.9pt] (axis cs:0.68,116) -- (axis cs:0.735,116);",
        r"\node[font=\scriptsize, anchor=west, text=textC] at (axis cs:0.750,116) {LogReg};",
        r"\draw[xgbC, line width=0.9pt] (axis cs:0.35,108) -- (axis cs:0.405,108);",
        r"\node[font=\scriptsize, anchor=west, text=textC] at (axis cs:0.420,108) {XGBoost};",
        r"\draw[hgbC, line width=0.9pt] (axis cs:0.68,108) -- (axis cs:0.735,108);",
        r"\node[font=\scriptsize, anchor=west, text=textC] at (axis cs:0.750,108) {HGB};",
        r"\addplot+[only marks, mark=*, mark size=2.2pt, color=textC, fill=textC] coordinates {(0.35,96)};",
        r"\addplot+[only marks, mark=triangle*, mark size=2.7pt, color=textC, fill=white, mark options={line width=0.70pt, draw=textC}] coordinates {(0.35,88)};",
        r"\addplot+[only marks, mark=square*, mark size=2.2pt, color=textC, fill=textC] coordinates {(0.35,80)};",
        r"\node[font=\scriptsize, anchor=west, text=textC] at (axis cs:0.365,96) {single};",
        r"\node[font=\scriptsize, anchor=west, text=textC] at (axis cs:0.365,88) {LODO};",
        r"\node[font=\scriptsize, anchor=west, text=textC] at (axis cs:0.365,80) {cross-node};",
        r"\nextgroupplot[",
        r"  title={B. Cold-start hosted rows},",
        r"  xmin=-0.010,",
        r"  xmax=0.92,",
        r"  ymin=-14,",
        r"  ymax=4.0,",
        r"  ylabel={},",
        r"  xtick={0,0.20,0.40,0.60,0.80},",
        r"  xticklabels={0,.20,.40,.60,.80},",
        r"  ytick={-14,-10,-6,-2,2},",
        r"]",
        r"\draw[dashed, black!50, line width=0.45pt] (axis cs:-0.010,0) -- (axis cs:0.92,0);",
    ]
    zero = rows["Qwen zero-shot"]
    out += [marker(zero, "zeroC", "*", "zeroC", "3.6pt"), label("zero-shot Qwen", 0.056, 0.36, color="zeroC")]
    hosted = {
        "Claude Opus 4.7 weak": ("claudeC", "diamond*", "claudeC", 0.300, 1.38, "west", "Claude weak"),
        "Claude Opus 4.7 strong": ("claudeC", "diamond*", "white", 0.455, -8.15, "west", "Claude strong"),
        "GPT-5.5-high weak": ("gptC", "diamond*", "gptC", 0.805, -11.42, "east", "GPT weak"),
        "GPT-5.5-high strong": ("gptC", "diamond*", "white", 0.625, -2.05, "west", "GPT strong"),
        "DeepSeek-v4-flash weak": ("deepseekC", "diamond*", "deepseekC", 0.235, -6.55, "west", "DeepSeek weak"),
        "DeepSeek-v4-flash strong": ("deepseekC", "diamond*", "white", 0.315, -9.75, "west", "DeepSeek strong"),
    }
    for key, (color, shape, fill, x, y, anchor, text) in hosted.items():
        out.append(marker(rows[key], color, shape, fill, "3.6pt" if fill == color else "4.0pt"))
        out.append(label(text, x, y, anchor=anchor, color=color))
    out += [
        r"\node[font=\scriptsize, anchor=east, text=textC, fill=white, fill opacity=0.86, text opacity=1, inner sep=1pt] at (axis cs:0.90,3.45) {filled=weak, open=strong};",
        r"\end{groupplot}",
        r"\end{tikzpicture}",
        r"\end{document}",
    ]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    rows = read_rows()
    OUT.write_text(generate(rows), encoding="utf-8")
    print(f"generated {OUT} from {DATA}")
