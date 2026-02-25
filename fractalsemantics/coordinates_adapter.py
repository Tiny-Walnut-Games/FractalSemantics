"""Adapters for converting payloads into typed FractalSemantics coordinates."""

from __future__ import annotations

from typing import Any

from fractalsemantics.dynamic_enum import Alignment, Horizon, Polarity, Realm
from fractalsemantics.fractalsemantics_entity import FractalSemanticsCoordinates

_REQUIRED_PAYLOAD_KEYS = (
    "realm",
    "lineage",
    "adjacency",
    "horizon",
    "luminosity",
    "polarity",
    "dimensionality",
    "alignment",
)


def payload_to_fractalsemantics_coordinates(
    payload: dict[str, Any],
) -> FractalSemanticsCoordinates:
    """Convert embedding-derived payload dict into FractalSemanticsCoordinates."""
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dictionary")

    missing = [key for key in _REQUIRED_PAYLOAD_KEYS if key not in payload]
    if missing:
        raise ValueError(f"payload missing required keys: {', '.join(missing)}")

    try:
        realm = Realm(str(payload["realm"]))
        lineage = int(payload["lineage"])
        adjacency = float(payload["adjacency"])
        horizon = Horizon(str(payload["horizon"]))
        luminosity = float(payload["luminosity"])
        polarity = Polarity(str(payload["polarity"]))
        dimensionality = int(payload["dimensionality"])
        alignment = Alignment(str(payload["alignment"]))
    except (TypeError, ValueError) as exc:
        raise ValueError("payload contains invalid coordinate values") from exc

    if not 0 <= lineage <= 999:
        raise ValueError("lineage must be in range [0, 999]")
    if not 0.0 <= adjacency <= 100.0:
        raise ValueError("adjacency must be in range [0.0, 100.0]")
    if not 0.0 <= luminosity <= 100.0:
        raise ValueError("luminosity must be in range [0.0, 100.0]")
    if not 0 <= dimensionality <= 9:
        raise ValueError("dimensionality must be in range [0, 9]")

    return FractalSemanticsCoordinates(
        realm=realm,
        lineage=lineage,
        adjacency=adjacency,
        horizon=horizon,
        luminosity=luminosity,
        polarity=polarity,
        dimensionality=dimensionality,
        alignment=alignment,
    )
