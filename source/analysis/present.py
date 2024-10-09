from collections import defaultdict

from typing import Optional
from rich import print as RichPrint
from rich.text import Text as RichText
from tabulate import tabulate

from source.analysis.metrics import METRIC_FUNC, Metrics


def generate_metrics_table(metrics: list[Metrics]):
    headers = ["ID"] + list(METRIC_FUNC.keys())
    grouped_metrics = defaultdict(list)
    for metric in metrics:
        grouped_metrics[metric.meta_id].append(
            f"{metric.mean:.2f} (+- {metric.std:.2f})"
        )
    rows = [[meta_id] + values for meta_id, values in grouped_metrics.items()]
    table = tabulate(rows, headers, tablefmt="grid")
    metrics_all = metrics[-len(METRIC_FUNC.keys()) :]
    return table, metrics_all


def display_metrics(
    metrics: dict[str, list[Metrics]],
    model: Optional[str] = None,
    do_optimize: Optional[bool] = None,
    group_id: Optional[int] = None,
    imaging_id: Optional[list[int]] = [1, 2, 3],
    camera_label: Optional[list[str]] = ["vnir", "swir"],
):
    headers = ["ID"] + list(METRIC_FUNC.keys())
    rows = []
    for key, metrics_ in metrics.items():
        row = [key]
        for metric in metrics_:
            row.append(str(metric.mean))
        rows.append(row)

    table = tabulate(rows, headers, tablefmt="grid")
    RichPrint(RichText(table, style="white"))
