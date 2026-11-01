# dentex_bench/eval/reporter.py
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Any, Dict, Union
from tabulate import tabulate


def format_benchmark_markdown(results: Dict[str, Dict[str, Any]]) -> str:
    """Formats benchmark results dictionary into a GitHub markdown table."""
    headers = [
        "Model",
        "mAP50",
        "mAP50:95",
        "Caries",
        "Deep Caries",
        "Periapical",
        "Impacted",
        "Latency (ms)",
        "Peak VRAM (MB)",
    ]
    rows = []
    for model_name, m in results.items():
        rows.append([
            model_name,
            f"{m.get('mAP50', 0.0):.3f}",
            f"{m.get('mAP50_95', 0.0):.3f}",
            f"{m.get('AP_Caries', 0.0):.3f}",
            f"{m.get('AP_DeepCaries', 0.0):.3f}",
            f"{m.get('AP_PeriapicalLesion', 0.0):.3f}",
            f"{m.get('AP_ImpactedTooth', 0.0):.3f}",
            f"{m.get('latency_ms', 0.0):.1f}",
            f"{m.get('peak_vram_mb', 0.0):.0f}",
        ])
    return tabulate(rows, headers=headers, tablefmt="github")


def format_benchmark_latex(results: Dict[str, Dict[str, Any]]) -> str:
    """Formats benchmark results dictionary into LaTeX table code."""
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{DENTEX 4-Class Abnormal Tooth Detection Benchmark Results}",
        "\\label{tab:dentex_benchmark}",
        "\\begin{tabular}{lcccccccc}",
        "\\hline",
        "Model & $mAP_{50}$ & $mAP_{50:95}$ & Caries & Deep Caries & Periapical & Impacted & Latency (ms) & VRAM (MB) \\\\",
        "\\hline",
    ]
    for model_name, m in results.items():
        lines.append(
            f"{model_name} & {m.get('mAP50', 0.0):.3f} & {m.get('mAP50_95', 0.0):.3f} & "
            f"{m.get('AP_Caries', 0.0):.3f} & {m.get('AP_DeepCaries', 0.0):.3f} & "
            f"{m.get('AP_PeriapicalLesion', 0.0):.3f} & {m.get('AP_ImpactedTooth', 0.0):.3f} & "
            f"{m.get('latency_ms', 0.0):.1f} & {m.get('peak_vram_mb', 0.0):.0f} \\\\"
        )
    lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table}",
    ])
    return "\n".join(lines)


def generate_markdown_report(benchmark_results: Dict[str, Dict[str, Any]], output_file: Union[str, Path]) -> None:
    """Generates and writes a markdown report to the specified file."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    md_content = format_benchmark_markdown(benchmark_results)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# DENTEX 4-Class Benchmark Results\n\n" + md_content + "\n")


def generate_latex_table(benchmark_results: Dict[str, Dict[str, Any]], output_file: Union[str, Path]) -> None:
    """Generates and writes a LaTeX table to the specified file."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tex_content = format_benchmark_latex(benchmark_results)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(tex_content + "\n")


def save_metrics_csv(results: Dict[str, Dict[str, Any]], output_file: Union[str, Path]) -> None:
    """Saves benchmark results to a CSV file."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Model", "mAP50", "mAP50_95", "AP_Caries", "AP_DeepCaries",
        "AP_PeriapicalLesion", "AP_ImpactedTooth", "latency_ms", "peak_vram_mb"
    ]
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, m in results.items():
            row = {"Model": model_name}
            for field in fieldnames[1:]:
                row[field] = m.get(field, 0.0)
            writer.writerow(row)


def save_comparison_plot(results: Dict[str, Dict[str, Any]], output_file: Union[str, Path]) -> None:
    """Saves a comparison bar chart of benchmark results."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    models = list(results.keys())
    if not models:
        return
        
    metrics_to_plot = ["mAP50", "mAP50_95", "AP_Caries", "AP_DeepCaries", "AP_PeriapicalLesion", "AP_ImpactedTooth"]
    metric_labels = ["mAP@50", "mAP@50:95", "Caries", "Deep Caries", "Periapical", "Impacted"]
    
    x = np.arange(len(metric_labels))
    width = 0.8 / max(1, len(models))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77", "#e7298a"]
    for i, model in enumerate(models):
        m = results[model]
        vals = [float(m.get(k, 0.0)) for k in metrics_to_plot]
        ax.bar(x + (i - len(models) / 2 + 0.5) * width, vals, width, label=model, color=colors[i % len(colors)], alpha=0.85)
        
    ax.set_ylabel("Score (0.0 - 1.0)")
    ax.set_title("DENTEX 4-Class Abnormal Tooth Detection Benchmark Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_file, dpi=160)
    plt.close(fig)


def save_reports(results: Dict[str, Dict[str, Any]], output_dir: Union[str, Path]) -> None:
    """Saves benchmark summary in Markdown, LaTeX, CSV, and comparison bar chart."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    md_file = output_dir / "benchmark_summary.md"
    tex_file = output_dir / "benchmark_table.tex"
    csv_file = output_dir / "metrics.csv"
    plot_file = output_dir / "metrics_comparison.png"

    generate_markdown_report(results, md_file)
    generate_latex_table(results, tex_file)
    save_metrics_csv(results, csv_file)
    save_comparison_plot(results, plot_file)

    print(f"[dentex_bench] Reports generated:\n  Markdown: {md_file}\n  LaTeX: {tex_file}\n  CSV: {csv_file}\n  Plot: {plot_file}")
