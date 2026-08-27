#!/usr/bin/env python3
"""Compute CUE-Bench evaluation metrics from JSONL prediction files."""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from collections import OrderedDict

try:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        recall_score,
        precision_recall_fscore_support,
    )
    import numpy as np
except ImportError:
    print("pip install scikit-learn numpy")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))
import config

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
RESULTS_DIR = ROOT_DIR / "results"

# Models and methods for table generation
MODELS = ["deepseek-v4-flash", "gpt-4o-mini", "llama-4-maverick", "llama-3.1-8b-instruct", "qwen3-8b", "glm-5.1"]
DISPLAY_METHODS = OrderedDict([
    ("zeroshot",     ("direct",      "zeroshot")),
    ("fewshot",      ("direct",      "fewshot")),
    ("cot",          ("cot",         "fewshot")),
    ("chain",        ("chain",       "fewshot")),
    ("structure_only", ("structure_only", "fewshot")),
    ("gold_p",       ("gold_p",      "fewshot")),
    ("gold_pi",      ("gold_pi",     "fewshot")),
])
TASKS = ["stance", "intent", "emotion"]
METRIC_NAMES = ["accuracy", "macro_recall", "macro_f1", "micro_f1", "weighted_f1"]
METRIC_DISPLAY = ["Acc", "Recall", "Ma-F1", "Mi-F1", "W-F1"]


def load_predictions(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    return records


def parse_filename(fname):
    """Parse model, method, task, shot from filename like 'gpt-4o-mini_direct_stance_zeroshot.jsonl'"""
    base = fname.replace(".jsonl", "")
    for task in TASKS:
        for shot in ["zeroshot", "fewshot"]:
            suffix = f"_{task}_{shot}"
            if base.endswith(suffix):
                prefix = base[:-len(suffix)]
                for method in ["structure_only", "gold_pi", "gold_p", "direct", "chain", "cot"]:
                    method_suffix = f"_{method}"
                    if prefix.endswith(method_suffix):
                        model = prefix[:-len(method_suffix)]
                        return model, method, task, shot
    return None, None, None, None


def compute_metrics(records, task):
    labels = config.TASKS[task]["labels"]

    golds = []
    preds = []
    parse_failures = 0

    for r in records:
        g = r.get("gold")
        p = r.get("predicted")
        if p is None:
            parse_failures += 1
            p = "PARSE_FAIL"
        golds.append(g)
        preds.append(p)

    valid_mask = [(g in labels) for g in golds]
    golds_f = [g for g, v in zip(golds, valid_mask) if v]
    preds_f = [p for p, v in zip(preds, valid_mask) if v]

    if not golds_f:
        return None

    acc = accuracy_score(golds_f, preds_f)
    macro_f1 = f1_score(golds_f, preds_f, labels=labels, average="macro", zero_division=0)
    micro_f1 = f1_score(golds_f, preds_f, labels=labels, average="micro", zero_division=0)
    weighted_f1 = f1_score(golds_f, preds_f, labels=labels, average="weighted", zero_division=0)
    macro_recall = float(np.mean(
        recall_score(golds_f, preds_f, labels=labels, average=None, zero_division=0)
    ))

    return {
        "total": len(records),
        "valid": len(golds_f),
        "parse_failures": parse_failures,
        "accuracy": round(acc, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
    }


def build_index():
    """Scan all result files and compute metrics. Returns dict[(model, method, task, shot)] = metrics."""
    index = {}
    for fpath in sorted(RESULTS_DIR.glob("*.jsonl")):
        model, method, task, shot = parse_filename(fpath.name)
        if not model:
            continue
        records = load_predictions(fpath)
        if not records:
            continue
        metrics = compute_metrics(records, task)
        if metrics:
            metrics["file"] = fpath.name
            metrics["model"] = model
            metrics["method"] = method
            metrics["task"] = task
            metrics["shot"] = shot
            index[(model, method, task, shot)] = metrics
    return index


def generate_table(index):
    """
    Generate the main comparison table:
    Rows: Model × Method (zero-shot / few-shot / cot)
    Columns: Task (stance / intent / emotion) × Metric (Acc, Recall, Ma-F1, Mi-F1, W-F1)
    """
    lines = []

    # Header row 1: task groups
    h1 = f"{'Model':<20} {'Method':<12}"
    for task in TASKS:
        h1 += f" | {task:^{len(METRIC_DISPLAY)*7-1}}"
    lines.append(h1)

    # Header row 2: metric names
    h2 = f"{'':<20} {'':<12}"
    for task in TASKS:
        for md in METRIC_DISPLAY:
            h2 += f"  {md:>5}"
        h2 += " "
    lines.append(h2)

    # Separator
    lines.append("-" * len(h2))

    # Data rows
    for model in MODELS:
        first_row = True
        for method_label, (method, shot) in DISPLAY_METHODS.items():
            model_col = model if first_row else ""
            row = f"{model_col:<20} {method_label:<12}"

            for task in TASKS:
                m = index.get((model, method, task, shot))
                if m:
                    for mn in METRIC_NAMES:
                        val = m.get(mn, 0)
                        row += f"  {val:5.3f}"
                else:
                    for _ in METRIC_NAMES:
                        row += f"  {'—':>5}"
                row += " "

            lines.append(row)
            first_row = False
        lines.append("")  # blank line between models

    return "\n".join(lines)


def generate_markdown_table(index):
    """Generate Markdown table for paper."""
    n_metrics = len(METRIC_DISPLAY)

    # Header
    header = "| Model | Method |"
    for task in TASKS:
        for md in METRIC_DISPLAY:
            header += f" {md} |"
    lines = [header]

    # Separator
    sep = "|---|---|"
    for task in TASKS:
        for _ in METRIC_DISPLAY:
            sep += "---:|"
    lines.append(sep)

    # Task header row (merged cells shown as comment)
    task_row = "| | |"
    for task in TASKS:
        task_row += f" **{task}** |" + " |" * (n_metrics - 1)
    lines.insert(1, task_row)

    # Data
    for model in MODELS:
        for i, (method_label, (method, shot)) in enumerate(DISPLAY_METHODS.items()):
            model_col = model if i == 0 else ""
            row = f"| {model_col} | {method_label} |"

            for task in TASKS:
                m = index.get((model, method, task, shot))
                if m:
                    for mn in METRIC_NAMES:
                        val = m.get(mn, 0)
                        row += f" {val:.3f} |"
                else:
                    for _ in METRIC_NAMES:
                        row += " — |"

            lines.append(row)

    return "\n".join(lines)


def generate_latex_table(index):
    """Generate LaTeX table."""
    n_metrics = len(METRIC_DISPLAY)
    col_spec = "ll|" + "|".join(["r" * n_metrics] * len(TASKS))

    lines = [
        "\\begin{table*}[h]",
        "\\centering",
        "\\caption{LLM Performance Comparison}",
        "\\small",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
    ]

    # Task header
    task_header = "Model & Method"
    for task in TASKS:
        task_header += f" & \\multicolumn{{{n_metrics}}}{{c|}}{{{task.capitalize()}}}"
    task_header = task_header.rstrip("|") + " \\\\"
    lines.append(task_header)

    # Metric header
    metric_header = " & "
    for task in TASKS:
        for md in METRIC_DISPLAY:
            metric_header += f" & {md}"
    metric_header += " \\\\"
    lines.append(metric_header)
    lines.append("\\midrule")

    # Data
    for model in MODELS:
        for i, (method_label, (method, shot)) in enumerate(DISPLAY_METHODS.items()):
            model_col = model.replace("_", "\\_") if i == 0 else ""
            row = f"{model_col} & {method_label}"

            for task in TASKS:
                m = index.get((model, method, task, shot))
                if m:
                    for mn in METRIC_NAMES:
                        val = m.get(mn, 0)
                        row += f" & {val:.3f}"
                else:
                    for _ in METRIC_NAMES:
                        row += " & —"

            row += " \\\\"
            lines.append(row)
        lines.append("\\midrule")

    lines[-1] = "\\bottomrule"
    lines.extend([
        "\\end{tabular}",
        "\\end{table*}",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="指标计算 + 表格生成")
    parser.add_argument("--all", action="store_true", help="计算所有结果并生成表格")
    parser.add_argument("--file", type=str, help="计算单个文件")
    args = parser.parse_args()

    if args.file:
        fpath = Path(args.file)
        if not fpath.exists():
            fpath = RESULTS_DIR / args.file
        records = load_predictions(fpath)
        _, _, task, _ = parse_filename(fpath.name)
        if task and records:
            m = compute_metrics(records, task)
            print(json.dumps(m, ensure_ascii=False, indent=2))
        return

    if not args.all:
        parser.print_help()
        return

    print("Scanning result files...")
    index = build_index()
    print(f"Found {len(index)} result files\n")

    # Save raw metrics
    summary = list(index.values())
    summary_path = RESULTS_DIR / "metrics_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Print text table
    print(generate_table(index))

    # Save Markdown
    md_path = RESULTS_DIR / "results_table.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# LLM Experiment Results\n\n")
        f.write(generate_markdown_table(index))
        f.write("\n")
    print(f"\nMarkdown table saved to {md_path}")

    # Save LaTeX
    tex_path = RESULTS_DIR / "results_table.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(generate_latex_table(index))
    print(f"LaTeX table saved to {tex_path}")

    print(f"Metrics summary saved to {summary_path}")


if __name__ == "__main__":
    main()
