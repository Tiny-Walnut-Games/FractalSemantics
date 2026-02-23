from fractalsemantics.coordinates_adapter import (
    payload_to_fractalsemantics_coordinates,
)


def test_payload_to_fractalsemantics_coordinates_returns_typed_coordinates() -> None:
    payload = {
        "realm": "pattern",
        "lineage": 7,
        "adjacency": 62.5,
        "horizon": "peak",
        "luminosity": 83.0,
        "polarity": "balance",
        "dimensionality": 4,
        "alignment": "true_neutral",
    }

    coords = payload_to_fractalsemantics_coordinates(payload)

    assert coords.realm.value == "pattern"
    assert coords.lineage == 7
    assert coords.adjacency == 62.5
    assert coords.horizon.value == "peak"
    assert coords.luminosity == 83.0
    assert coords.polarity.value == "balance"
    assert coords.dimensionality == 4
    assert coords.alignment.value == "true_neutral"
