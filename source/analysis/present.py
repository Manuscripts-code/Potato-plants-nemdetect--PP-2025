from collections import defaultdict

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


def display_metrics():
    pass
