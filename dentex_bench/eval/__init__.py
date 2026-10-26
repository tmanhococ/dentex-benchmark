# dentex_bench/eval/__init__.py
from .coco_eval import evaluate_coco_predictions, evaluate_coco
from .reporter import (
    format_benchmark_markdown,
    format_benchmark_latex,
    generate_markdown_report,
    generate_latex_table,
    save_reports,
)

__all__ = [
    "evaluate_coco_predictions",
    "evaluate_coco",
    "format_benchmark_markdown",
    "format_benchmark_latex",
    "generate_markdown_report",
    "generate_latex_table",
    "save_reports",
]
