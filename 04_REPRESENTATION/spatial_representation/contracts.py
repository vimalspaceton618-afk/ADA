"""Typed spatial records; no spatial estimator is implied by these contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SpatialPoint:
    coordinates: tuple[float, ...]
    coordinate_frame: str
    unit: str

    def __post_init__(self) -> None:
        values = tuple(self.coordinates)
        if len(values) not in {2, 3}:
            raise ValueError("spatial point must have two or three coordinates")
        if not self.coordinate_frame or not self.unit:
            raise ValueError("coordinate_frame and unit must be non-empty")
        for index, value in enumerate(values):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"coordinate[{index}] must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"coordinate[{index}] must be finite")
        object.__setattr__(self, "coordinates", tuple(float(v) for v in values))


@dataclass(frozen=True, slots=True)
class SpatialRegion:
    minimum: SpatialPoint
    maximum: SpatialPoint

    def __post_init__(self) -> None:
        if self.minimum.coordinate_frame != self.maximum.coordinate_frame:
            raise ValueError("spatial region points must use the same coordinate frame")
        if self.minimum.unit != self.maximum.unit:
            raise ValueError("spatial region points must use the same unit")
        if len(self.minimum.coordinates) != len(self.maximum.coordinates):
            raise ValueError("spatial region points must have matching dimensions")
        if any(
            low > high
            for low, high in zip(
                self.minimum.coordinates, self.maximum.coordinates, strict=True
            )
        ):
            raise ValueError("spatial region minimum must not exceed maximum")
