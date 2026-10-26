# dentex_bench/eval/reporter.py
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


def save_reports(results: Dict[str, Dict[str, Any]], output_dir: Union[str, Path]) -> None:
    """Saves benchmark summary in both Markdown and LaTeX format inside output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    md_file = output_dir / "benchmark_summary.md"
    tex_file = output_dir / "benchmark_table.tex"

    generate_markdown_report(results, md_file)
    generate_latex_table(results, tex_file)

    print(f"[dentex_bench] Reports generated:\n  Markdown: {md_file}\n  LaTeX: {tex_file}")
