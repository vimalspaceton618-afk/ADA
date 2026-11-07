"""Dependency-light metrics for evaluating world-model predictions."""
from __future__ import annotations
import math
from collections.abc import Iterable


def _numbers(values, name: str) -> list[float]:
    if hasattr(values, "detach"):
        values = values.detach().cpu().reshape(-1).tolist()
    elif hasattr(values, "ravel"):
        values = values.ravel().tolist()
    result = [float(x) for x in values]
    if not result or any(not math.isfinite(x) for x in result):
        raise ValueError(f"{name} must contain finite numeric values")
    return result


def mean_squared_error(predictions, targets) -> float:
    pred, truth = _numbers(predictions, "predictions"), _numbers(targets, "targets")
    if len(pred) != len(truth):
        raise ValueError("predictions and targets must have equal flattened length")
    return sum((a-b) ** 2 for a, b in zip(pred, truth)) / len(pred)


def mean_absolute_error(predictions, targets) -> float:
    pred, truth = _numbers(predictions, "predictions"), _numbers(targets, "targets")
    if len(pred) != len(truth):
        raise ValueError("predictions and targets must have equal flattened length")
    return sum(abs(a-b) for a, b in zip(pred, truth)) / len(pred)


def expected_calibration_error(confidences, outcomes, *, bins: int = 10) -> float:
    """Equal-width ECE for scalar binary correctness outcomes in [0,1]."""
    confidence = _numbers(confidences, "confidences")
    outcome = _numbers(outcomes, "outcomes")
    if len(confidence) != len(outcome):
        raise ValueError("confidences and outcomes must have equal length")
    if type(bins) is not int or bins < 1:
        raise ValueError("bins must be a positive integer")
    if any(not 0 <= x <= 1 for x in confidence + outcome):
        raise ValueError("confidences and outcomes must be in [0,1]")
    total = len(confidence)
    error = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        selected = [i for i, value in enumerate(confidence)
                    if lower <= value < upper or (index == bins - 1 and value == 1.0)]
        if selected:
            avg_conf = sum(confidence[i] for i in selected) / len(selected)
            avg_acc = sum(outcome[i] for i in selected) / len(selected)
            error += len(selected) / total * abs(avg_conf - avg_acc)
    return error


def horizon_mse(predictions, targets) -> tuple[float, ...]:
    """Return MSE separately at each rollout horizon.

    Inputs are nested [batch,horizon,...feature] arrays/tensors. Values are
    flattened only within each horizon, preserving the time axis.
    """
    def to_nested(value, name):
        if hasattr(value, "detach"):
            value = value.detach().cpu().tolist()
        elif hasattr(value, "tolist"):
            value = value.tolist()
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError(f"{name} must be a non-empty [batch,horizon,...] sequence")
        return value
    pred, truth = to_nested(predictions, "predictions"), to_nested(targets, "targets")
    if len(pred) != len(truth):
        raise ValueError("batch dimensions must match")
    horizon = len(pred[0])
    if horizon < 1 or any(len(row) != horizon for row in pred + truth):
        raise ValueError("all sequences must have the same non-empty horizon")
    result = []
    for h in range(horizon):
        p, t = [], []
        for b in range(len(pred)):
            p.extend(_numbers_nested(pred[b][h]))
            t.extend(_numbers_nested(truth[b][h]))
        if len(p) != len(t):
            raise ValueError("prediction and target feature shapes must match")
        result.append(sum((a-b) ** 2 for a, b in zip(p, t)) / len(p))
    return tuple(result)


def _numbers_nested(value) -> list[float]:
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_numbers_nested(item))
        return out
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("prediction metrics require finite values")
    return [number]
