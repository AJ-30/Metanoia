"""The human-readable report: one text table, built from the summary.

The JSON summary is the artifact; this is the part a person reads at the end of a
run. It is a pure function of the summary dict so it is trivially unit-testable,
and it mirrors the comparison table the YOLO validation stage prints so the two
stages read the same way.
"""

from __future__ import annotations

from typing import Any

RULE_WIDTH: int = 78


def _format_metric(value: float | None) -> str:
    """Format one optional metric for the table.

    Args:
        value (float | None): The metric, or ``None`` when it does not apply.

    Returns:
        str: The value at four decimals, or ``n/a``.
    """
    return "n/a" if value is None else f"{value:.4f}"


def _format_statistics(statistics: dict[str, float]) -> str:
    """Format one distribution as ``mean (median) [p10..p90]``.

    Args:
        statistics (dict[str, float]): The summary statistics of a distribution.

    Returns:
        str: One-line summary; ``n/a`` when the distribution is empty.
    """
    if not statistics:
        return "n/a"
    return (
        f"mean {statistics.get('mean', 0.0):.4f} | "
        f"median {statistics.get('median', 0.0):.4f} | "
        f"p10 {statistics.get('p10', 0.0):.4f}..p90 {statistics.get('p90', 0.0):.4f}"
    )


def _format_result_section(title: str, result: dict[str, Any] | None) -> list[str]:
    """Format one result's detector, box-loss, channel and outcome numbers.

    Args:
        title (str): Section heading.
        result (dict[str, Any] | None): A per-draw or ensemble summary, or
            ``None`` when the section is absent.

    Returns:
        list[str]: Report lines for this result.
    """
    lines: list[str] = [title]
    if not result:
        lines.append("  n/a")
        return lines
    metrics: dict[str, float] | None = result.get("detector_metrics")
    if metrics is None:
        lines.append("  detector metrics   n/a")
    else:
        lines.append(
            f"  mAP@50             {_format_metric(metrics.get('mAP50'))} | "
            f"mAP@50-95 {_format_metric(metrics.get('mAP50_95'))} | "
            f"precision {_format_metric(metrics.get('precision'))} | "
            f"recall {_format_metric(metrics.get('recall'))}"
        )
    box_loss: dict[str, dict[str, float]] = result.get("box_loss", {})
    lines.append(
        f"  box IoU            {_format_statistics(box_loss.get('box_iou', {}))}"
    )
    lines.append(
        f"  L1 on (cx,cy,w,h)  {_format_statistics(box_loss.get('l1_norm_xywh', {}))}"
    )
    lines.append(
        f"  centre dist (px)   {_format_statistics(box_loss.get('center_distance_px', {}))}"
    )
    channel: dict[str, dict[str, float]] = result.get("channel_scores", {})
    lines.append(f"  channel Dice       {_format_statistics(channel.get('dice', {}))}")
    lines.append(f"  channel IoU        {_format_statistics(channel.get('iou', {}))}")
    outcomes: dict[str, int] = result.get("outcomes", {})
    lines.append(
        f"  outcomes           gt={outcomes.get('ground_truth_boxes')} "
        f"matched={outcomes.get('matched')} missed={outcomes.get('missed')} "
        f"false_positives={outcomes.get('false_positives')}"
    )
    return lines


def format_report(summary: dict[str, Any]) -> list[str]:
    """Build the report lines of one evaluation.

    Args:
        summary (dict[str, Any]): The summary document written as JSON.

    Returns:
        list[str]: Report lines, one per row, ready to be joined by a newline.
    """
    component: dict[str, Any] = summary.get("missing_component", {})
    configuration: dict[str, Any] = summary.get("configuration", {})
    lines: list[str] = [
        "",
        "=" * RULE_WIDTH,
        "            MISSING-COMPONENT EVALUATION REPORT",
        "=" * RULE_WIDTH,
        f"Missing component : {component.get('name')} "
        f"(mask channel {component.get('mask_channel')}, "
        f"detector class {component.get('detector_class_id')} "
        f"'{component.get('detector_class_name')}', "
        f"dataset labels {component.get('dataset_label_names')})",
        f"Diffusion ckpt    : {configuration.get('mask_diffusion_checkpoint')} "
        f"(epoch {configuration.get('mask_diffusion_epoch')})",
        f"Detector          : {configuration.get('yolo_weights')}",
        f"Split / samples   : {configuration.get('samples_evaluated')} samples from "
        f"{configuration.get('split_dir')}",
        f"Ensemble members  : {configuration.get('draws_per_sample')} "
        f"(seed {configuration.get('seed')})",
        f"Ensemble threshold: "
        f"{configuration.get('ensemble_probability_threshold')} "
        "(softmax probability of the missing component)",
        "-" * RULE_WIDTH,
        "PRIMARY RESULT: ensemble of all draws",
        "-" * RULE_WIDTH,
    ]
    lines += _format_result_section(
        "ensemble", summary.get("ensemble")
    )
    lines += [
        "-" * RULE_WIDTH,
        "PER-DRAW DIAGNOSTICS",
        "-" * RULE_WIDTH,
    ]
    for draw, draw_summary in enumerate(summary.get("per_draw", [])):
        lines += _format_result_section(f"draw {draw}", draw_summary)
    aggregate: dict[str, Any] = summary.get("aggregate", {})
    aggregate_metrics: dict[str, float] | None = aggregate.get("detector_metrics")
    lines += [
        "-" * RULE_WIDTH,
        "Aggregate across draws (diagnostic)",
        "-" * RULE_WIDTH,
        f"  mAP@50             {_format_metric((aggregate_metrics or {}).get('mAP50'))}",
        f"  mAP@50-95          {_format_metric((aggregate_metrics or {}).get('mAP50_95'))}",
        f"  box IoU mean       {_format_metric((aggregate.get('box_loss', {}).get('box_iou', {}) or {}).get('mean'))}",
        f"  channel Dice mean  {_format_metric((aggregate.get('channel_scores', {}).get('dice', {}) or {}).get('mean'))}",
        f"  elapsed            {summary.get('elapsed_seconds', 0.0):.1f}s",
        f"  output root        {summary.get('output_root')}",
        "=" * RULE_WIDTH,
        "",
    ]
    return lines
