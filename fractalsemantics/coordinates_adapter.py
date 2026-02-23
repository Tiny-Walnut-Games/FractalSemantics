"""Adapters for converting payloads into typed FractalSemantics coordinates."""

from __future__ import annotations

from typing import Any

from fractalsemantics.dynamic_enum import Alignment, Horizon, Polarity, Realm
from fractalsemantics.fractalsemantics_entity import FractalSemanticsCoordinates


def payload_to_fractalsemantics_coordinates(
    payload: dict[str, Any],
) -> FractalSemanticsCoordinates:
    """Convert embedding-derived payload dict into FractalSemanticsCoordinates."""
    return FractalSemanticsCoordinates(
        realm=Realm(str(payload["realm"])),
        lineage=int(payload["lineage"]),
        adjacency=float(payload["adjacency"]),
        horizon=Horizon(str(payload["horizon"])),
        luminosity=float(payload["luminosity"]),
        polarity=Polarity(str(payload["polarity"])),
        dimensionality=int(payload["dimensionality"]),
        alignment=Alignment(str(payload["alignment"])),
    )
