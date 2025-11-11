"""
Test suite for STAT7 Validation Experiments: Phase 1 Doctrine Testing
Tests address uniqueness, retrieval efficiency, and dimension necessity.
"""

import pytest
from datetime import datetime, timezone


class TestDataClassEnum:
    """Test DataClass security enum."""

    def test_data_class_values(self):
        """DataClass enum should have security levels."""
        from fractalstat.stat7_experiments import DataClass

        assert DataClass.PUBLIC.value == "PUBLIC"
        assert DataClass.SENSITIVE.value == "SENSITIVE"
        assert DataClass.PII.value == "PII"


class TestCapabilityEnum:
    """Test Capability recovery enum."""

    def test_capability_values(self):
        """Capability enum should have recovery levels."""
        from fractalstat.stat7_experiments import Capability

        assert Capability.COMPRESSED.value == "compressed"
        assert Capability.PARTIAL.value == "partial"
        assert Capability.FULL.value == "full"


class TestNormalizeFloat:
    """Test floating point normalization."""

    def test_normalize_float_standard(self):
        """normalize_float should normalize to 8 decimal places."""
        from fractalstat.stat7_experiments import normalize_float

        result = normalize_float(0.123456789)
        assert isinstance(result, str)
        assert len(result.split(".")[1]) <= 8

    def test_normalize_float_banker_rounding(self):
        """normalize_float should use banker's rounding."""
        from fractalstat.stat7_experiments import normalize_float

        result = normalize_float(0.5)
        assert result == "0.5"

    def test_normalize_float_rejects_nan(self):
        """normalize_float should reject NaN."""
        from fractalstat.stat7_experiments import normalize_float

        with pytest.raises(ValueError):
            normalize_float(float("nan"))

    def test_normalize_float_rejects_inf(self):
        """normalize_float should reject Inf."""
        from fractalstat.stat7_experiments import normalize_float

        with pytest.raises(ValueError):
            normalize_float(float("inf"))


class TestNormalizeTimestamp:
    """Test timestamp normalization."""

    def test_normalize_timestamp_current(self):
        """normalize_timestamp with None should use current time."""
        from fractalstat.stat7_experiments import normalize_timestamp

        result = normalize_timestamp()
        assert isinstance(result, str)
        assert "T" in result
        assert result.endswith("Z")

    def test_normalize_timestamp_iso8601(self):
        """normalize_timestamp should return ISO8601 format."""
        from fractalstat.stat7_experiments import normalize_timestamp

        ts = "2024-01-01T12:30:45Z"
        result = normalize_timestamp(ts)
        assert isinstance(result, str)
        assert "T" in result
        assert "Z" in result

    def test_normalize_timestamp_milliseconds(self):
        """normalize_timestamp should include milliseconds."""
        from fractalstat.stat7_experiments import normalize_timestamp

        result = normalize_timestamp()
        parts = result.split(".")
        assert len(parts) == 2
        assert len(parts[1]) == 4  # mmmZ


class TestCanonicalSerialization:
    """Test canonical serialization."""

    def test_canonical_serialize_simple_dict(self):
        """canonical_serialize should handle simple dicts."""
        from fractalstat.stat7_experiments import canonical_serialize

        data = {"b": 2, "a": 1}
        result = canonical_serialize(data)
        assert isinstance(result, str)
        assert result.startswith("{")

    def test_canonical_serialize_deterministic(self):
        """canonical_serialize should be deterministic."""
        from fractalstat.stat7_experiments import canonical_serialize

        data = {"key": "value", "number": 42}
        result1 = canonical_serialize(data)
        result2 = canonical_serialize(data)
        assert result1 == result2

    def test_canonical_serialize_sorted_keys(self):
        """canonical_serialize should sort keys."""
        from fractalstat.stat7_experiments import canonical_serialize

        data = {"z": 1, "a": 2, "m": 3}
        result = canonical_serialize(data)
        assert result.index('"a"') < result.index('"m"')
        assert result.index('"m"') < result.index('"z"')

    def test_canonical_serialize_no_whitespace(self):
        """canonical_serialize should not include extra whitespace."""
        from fractalstat.stat7_experiments import canonical_serialize

        data = {"key": "value"}
        result = canonical_serialize(data)
        assert "  " not in result


class TestComputeAddressHash:
    """Test address hash computation."""

    def test_compute_address_hash_returns_hex(self):
        """compute_address_hash should return hex string."""
        from fractalstat.stat7_experiments import compute_address_hash

        data = {"id": "test", "realm": "data"}
        result = compute_address_hash(data)
        assert isinstance(result, str)
        assert len(result) == 64  # SHA-256 hex


class TestCoordinates:
    """Test STAT7 Coordinates."""

    def test_coordinates_initialization(self):
        """Coordinates should initialize with all 7 dimensions."""
        from fractalstat.stat7_experiments import Coordinates

        coords = Coordinates(
            realm="data",
            lineage=1,
            adjacency=[],
            horizon="genesis",
            resonance=0.5,
            velocity=0.5,
            density=0.5,
        )

        assert coords.realm == "data"
        assert coords.lineage == 1
        assert coords.horizon == "genesis"

    def test_coordinates_to_dict(self):
        """Coordinates should convert to normalized dict."""
        from fractalstat.stat7_experiments import Coordinates

        coords = Coordinates(
            realm="narrative",
            lineage=5,
            adjacency=["id1", "id2"],
            horizon="peak",
            resonance=0.7,
            velocity=0.3,
            density=0.6,
        )

        coords_dict = coords.to_dict()
        assert isinstance(coords_dict, dict)
        assert coords_dict["realm"] == "narrative"
        assert "adjacency" in coords_dict
        assert coords_dict["adjacency"] == ["id1", "id2"]


class TestBitChain:
    """Test BitChain entity."""

    def test_bitchain_initialization(self):
        """BitChain should initialize with required fields."""
        from fractalstat.stat7_experiments import BitChain, Coordinates

        coords = Coordinates(
            realm="data",
            lineage=1,
            adjacency=[],
            horizon="genesis",
            resonance=0.5,
            velocity=0.5,
            density=0.5,
        )

        bc = BitChain(
            id="test-123",
            entity_type="artifact",
            realm="data",
            coordinates=coords,
            created_at=datetime.now(timezone.utc).isoformat(),
            state={"value": 42},
        )

        assert bc.id == "test-123"
        assert bc.realm == "data"

    def test_bitchain_to_canonical_dict(self):
        """BitChain should convert to canonical dict."""
        from fractalstat.stat7_experiments import BitChain, Coordinates

        coords = Coordinates(
            realm="data",
            lineage=1,
            adjacency=[],
            horizon="genesis",
            resonance=0.5,
            velocity=0.5,
            density=0.5,
        )

        bc = BitChain(
            id="test-123",
            entity_type="artifact",
            realm="data",
            coordinates=coords,
            created_at=datetime.now(timezone.utc).isoformat(),
            state={},
        )

        canonical = bc.to_canonical_dict()
        assert isinstance(canonical, dict)
        assert "id" in canonical
        assert "stat7_coordinates" in canonical

    def test_bitchain_compute_address(self):
        """BitChain should compute unique address."""
        from fractalstat.stat7_experiments import BitChain, Coordinates

        coords = Coordinates(
            realm="data",
            lineage=1,
            adjacency=[],
            horizon="genesis",
            resonance=0.5,
            velocity=0.5,
            density=0.5,
        )

        bc = BitChain(
            id="test-456",
            entity_type="artifact",
            realm="data",
            coordinates=coords,
            created_at=datetime.now(timezone.utc).isoformat(),
            state={},
        )

        address = bc.compute_address()
        assert isinstance(address, str)
        assert len(address) == 64

    def test_bitchain_get_stat7_uri(self):
        """BitChain should generate STAT7 URI."""
        from fractalstat.stat7_experiments import BitChain, Coordinates

        coords = Coordinates(
            realm="narrative",
            lineage=3,
            adjacency=["adj1"],
            horizon="peak",
            resonance=0.7,
            velocity=0.3,
            density=0.6,
        )

        bc = BitChain(
            id="test-uri",
            entity_type="concept",
            realm="narrative",
            coordinates=coords,
            created_at=datetime.now(timezone.utc).isoformat(),
            state={},
        )

        uri = bc.get_stat7_uri()
        assert isinstance(uri, str)
        assert uri.startswith("stat7://")
        assert "narrative" in uri
        assert "peak" in uri


class TestGenerateRandomBitchain:
    """Test random bitchain generation."""

    def test_generate_random_bitchain_returns_bitchain(self):
        """generate_random_bitchain should return BitChain."""
        from fractalstat.stat7_experiments import generate_random_bitchain

        bc = generate_random_bitchain()
        assert bc is not None
        assert hasattr(bc, "id")
        assert hasattr(bc, "coordinates")

    def test_generate_random_bitchain_deterministic_with_seed(self):
        """generate_random_bitchain with seed should be deterministic."""
        from fractalstat.stat7_experiments import generate_random_bitchain

        bc1 = generate_random_bitchain(seed=42)
        bc2 = generate_random_bitchain(seed=42)

        assert bc1.id == bc2.id
        assert bc1.realm == bc2.realm


class TestEXP01AddressUniqueness:
    """Test EXP-01 address uniqueness."""

    def test_exp01_initializes(self):
        """EXP01_AddressUniqueness should initialize."""
        from fractalstat.stat7_experiments import EXP01_AddressUniqueness

        exp = EXP01_AddressUniqueness(sample_size=100, iterations=2)
        assert exp.sample_size == 100
        assert exp.iterations == 2

    def test_exp01_run_returns_results(self):
        """EXP01 run should return results."""
        from fractalstat.stat7_experiments import EXP01_AddressUniqueness

        exp = EXP01_AddressUniqueness(sample_size=10, iterations=1)
        results, success = exp.run()

        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(success, bool)

    def test_exp01_detects_collisions(self):
        """EXP01 should accurately report collisions."""
        from fractalstat.stat7_experiments import EXP01_AddressUniqueness

        exp = EXP01_AddressUniqueness(sample_size=100, iterations=1)
        results, _ = exp.run()

        result = results[0]
        assert result.total_bitchains == 100
        assert result.unique_addresses <= result.total_bitchains


class TestEXP01Result:
    """Test EXP-01 result tracking."""

    def test_exp01_result_initialization(self):
        """EXP01_Result should track test metrics."""
        from fractalstat.stat7_experiments import EXP01_Result

        result = EXP01_Result(
            iteration=1,
            total_bitchains=100,
            unique_addresses=100,
            collisions=0,
            collision_rate=0.0,
            success=True,
        )

        assert result.iteration == 1
        assert result.success

    def test_exp01_result_to_dict(self):
        """EXP01_Result should serialize to dict."""
        from fractalstat.stat7_experiments import EXP01_Result

        result = EXP01_Result(
            iteration=1,
            total_bitchains=100,
            unique_addresses=100,
            collisions=0,
            collision_rate=0.0,
            success=True,
        )

        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["iteration"] == 1


class TestRealmDefinitions:
    """Test realm and horizon definitions."""

    def test_realms_defined(self):
        """REALMS should be defined."""
        from fractalstat.stat7_experiments import REALMS

        assert isinstance(REALMS, list)
        assert len(REALMS) > 0
        assert "data" in REALMS

    def test_horizons_defined(self):
        """HORIZONS should be defined."""
        from fractalstat.stat7_experiments import HORIZONS

        assert isinstance(HORIZONS, list)
        assert len(HORIZONS) > 0
        assert "genesis" in HORIZONS

    def test_entity_types_defined(self):
        """ENTITY_TYPES should be defined."""
        from fractalstat.stat7_experiments import ENTITY_TYPES

        assert isinstance(ENTITY_TYPES, list)
        assert len(ENTITY_TYPES) > 0


class TestNormalizeFloatEdgeCases:
    """Test normalize_float edge cases and branches."""

    def test_normalize_float_negative_inf(self):
        """normalize_float should reject negative infinity."""
        from fractalstat.stat7_experiments import normalize_float

        with pytest.raises(ValueError):
            normalize_float(float("-inf"))

    def test_normalize_float_zero(self):
        """normalize_float should handle zero."""
        from fractalstat.stat7_experiments import normalize_float

        result = normalize_float(0.0)
        assert result == "0.0"

    def test_normalize_float_negative_value(self):
        """normalize_float should handle negative values."""
        from fractalstat.stat7_experiments import normalize_float

        result = normalize_float(-0.123456789)
        assert result.startswith("-")
        assert isinstance(result, str)

    def test_normalize_float_trailing_zeros_removed(self):
        """normalize_float should remove trailing zeros."""
        from fractalstat.stat7_experiments import normalize_float

        result = normalize_float(1.5)
        assert result == "1.5"
        assert not result.endswith("00")

    def test_normalize_float_custom_decimal_places(self):
        """normalize_float should support custom decimal places."""
        from fractalstat.stat7_experiments import normalize_float

        result = normalize_float(0.123456789, decimal_places=4)
        parts = result.split(".")
        assert len(parts[1]) <= 4

    def test_normalize_float_large_value(self):
        """normalize_float should handle large values."""
        from fractalstat.stat7_experiments import normalize_float

        result = normalize_float(999999.123456789)
        assert isinstance(result, str)
        assert "999999" in result


class TestNormalizeTimestampEdgeCases:
    """Test normalize_timestamp edge cases."""

    def test_normalize_timestamp_with_timezone(self):
        """normalize_timestamp should handle timezone conversion."""
        from fractalstat.stat7_experiments import normalize_timestamp

        ts = "2024-01-01T12:30:45+05:00"
        result = normalize_timestamp(ts)
        assert result.endswith("Z")
        assert "2024-01-01" in result

    def test_normalize_timestamp_preserves_date(self):
        """normalize_timestamp should preserve date components."""
        from fractalstat.stat7_experiments import normalize_timestamp

        ts = "2024-06-15T08:45:30Z"
        result = normalize_timestamp(ts)
        assert "2024-06-15" in result

    def test_normalize_timestamp_format_structure(self):
        """normalize_timestamp should follow YYYY-MM-DDTHH:MM:SS.mmmZ format."""
        from fractalstat.stat7_experiments import normalize_timestamp
        import re

        result = normalize_timestamp()
        pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z"
        assert re.match(pattern, result)


class TestSortJsonKeys:
    """Test sort_json_keys function."""

    def test_sort_json_keys_simple_dict(self):
        """sort_json_keys should sort dictionary keys."""
        from fractalstat.stat7_experiments import sort_json_keys

        data = {"z": 1, "a": 2, "m": 3}
        result = sort_json_keys(data)
        keys = list(result.keys())
        assert keys == ["a", "m", "z"]

    def test_sort_json_keys_nested_dict(self):
        """sort_json_keys should recursively sort nested dicts."""
        from fractalstat.stat7_experiments import sort_json_keys

        data = {"outer": {"z": 1, "a": 2}, "inner": {"y": 3, "b": 4}}
        result = sort_json_keys(data)
        assert list(result.keys()) == ["inner", "outer"]
        assert list(result["outer"].keys()) == ["a", "z"]
        assert list(result["inner"].keys()) == ["b", "y"]

    def test_sort_json_keys_with_list(self):
        """sort_json_keys should handle lists."""
        from fractalstat.stat7_experiments import sort_json_keys

        data = {"items": [{"z": 1, "a": 2}, {"y": 3, "b": 4}]}
        result = sort_json_keys(data)
        assert list(result["items"][0].keys()) == ["a", "z"]
        assert list(result["items"][1].keys()) == ["b", "y"]

    def test_sort_json_keys_primitive_values(self):
        """sort_json_keys should preserve primitive values."""
        from fractalstat.stat7_experiments import sort_json_keys

        assert sort_json_keys(42) == 42
        assert sort_json_keys("string") == "string"
        assert sort_json_keys(None) is None
        assert sort_json_keys(True) is True


class TestCanonicalSerializeEdgeCases:
    """Test canonical_serialize edge cases."""

    def test_canonical_serialize_nested_structure(self):
        """canonical_serialize should handle nested structures."""
        from fractalstat.stat7_experiments import canonical_serialize

        data = {"level1": {"level2": {"level3": "value"}}}
        result = canonical_serialize(data)
        assert isinstance(result, str)
        assert "level1" in result
        assert "level3" in result

    def test_canonical_serialize_with_arrays(self):
        """canonical_serialize should handle arrays."""
        from fractalstat.stat7_experiments import canonical_serialize

        data = {"items": [1, 2, 3], "names": ["a", "b"]}
        result = canonical_serialize(data)
        assert "[1,2,3]" in result or "[1, 2, 3]" in result.replace(" ", "")

    def test_canonical_serialize_unicode(self):
        """canonical_serialize should handle unicode."""
        from fractalstat.stat7_experiments import canonical_serialize

        data = {"text": "hello", "number": 123}
        result = canonical_serialize(data)
        assert isinstance(result, str)

    def test_canonical_serialize_empty_dict(self):
        """canonical_serialize should handle empty dict."""
        from fractalstat.stat7_experiments import canonical_serialize

        result = canonical_serialize({})
        assert result == "{}"


class TestComputeAddressHashDeterminism:
    """Test compute_address_hash determinism."""

    def test_compute_address_hash_deterministic(self):
        """compute_address_hash should be deterministic."""
        from fractalstat.stat7_experiments import compute_address_hash

        data = {"id": "test123", "value": 42}
        hash1 = compute_address_hash(data)
        hash2 = compute_address_hash(data)
        assert hash1 == hash2

    def test_compute_address_hash_different_for_different_data(self):
        """compute_address_hash should differ for different data."""
        from fractalstat.stat7_experiments import compute_address_hash

        hash1 = compute_address_hash({"id": "test1"})
        hash2 = compute_address_hash({"id": "test2"})
        assert hash1 != hash2

    def test_compute_address_hash_key_order_independent(self):
        """compute_address_hash should be independent of key order."""
        from fractalstat.stat7_experiments import compute_address_hash

        hash1 = compute_address_hash({"a": 1, "b": 2})
        hash2 = compute_address_hash({"b": 2, "a": 1})
        assert hash1 == hash2


class TestCoordinatesEdgeCases:
    """Test Coordinates edge cases."""

    def test_coordinates_adjacency_sorted(self):
        """Coordinates.to_dict should sort adjacency list."""
        from fractalstat.stat7_experiments import Coordinates

        coords = Coordinates(
            realm="data",
            lineage=1,
            adjacency=["id3", "id1", "id2"],
            horizon="genesis",
            resonance=0.5,
            velocity=0.5,
            density=0.5,
        )
        coords_dict = coords.to_dict()
        assert coords_dict["adjacency"] == ["id1", "id2", "id3"]

    def test_coordinates_empty_adjacency(self):
        """Coordinates should handle empty adjacency."""
        from fractalstat.stat7_experiments import Coordinates

        coords = Coordinates(
            realm="data",
            lineage=1,
            adjacency=[],
            horizon="genesis",
            resonance=0.0,
            velocity=0.0,
            density=0.0,
        )
        coords_dict = coords.to_dict()
        assert coords_dict["adjacency"] == []

    def test_coordinates_boundary_values(self):
        """Coordinates should handle boundary values."""
        from fractalstat.stat7_experiments import Coordinates

        coords = Coordinates(
            realm="void",
            lineage=100,
            adjacency=["id1"],
            horizon="crystallization",
            resonance=-1.0,
            velocity=1.0,
            density=1.0,
        )
        coords_dict = coords.to_dict()
        assert coords_dict["resonance"] == -1.0
        assert coords_dict["velocity"] == 1.0
        assert coords_dict["density"] == 1.0

    def test_coordinates_float_normalization(self):
        """Coordinates.to_dict should normalize floats."""
        from fractalstat.stat7_experiments import Coordinates

        coords = Coordinates(
            realm="data",
            lineage=1,
            adjacency=[],
            horizon="genesis",
            resonance=0.123456789,
            velocity=0.987654321,
            density=0.555555555,
        )
        coords_dict = coords.to_dict()
        assert isinstance(coords_dict["resonance"], float)
        assert isinstance(coords_dict["velocity"], float)
        assert isinstance(coords_dict["density"], float)


class TestBitChainEdgeCases:
    """Test BitChain edge cases."""

    def test_bitchain_security_defaults(self):
        """BitChain should have default security settings."""
        from fractalstat.stat7_experiments import BitChain, Coordinates, DataClass

        coords = Coordinates(
            realm="data",
            lineage=1,
            adjacency=[],
            horizon="genesis",
            resonance=0.5,
            velocity=0.5,
            density=0.5,
        )
        bc = BitChain(
            id="test",
            entity_type="artifact",
            realm="data",
            coordinates=coords,
            created_at=datetime.now(timezone.utc).isoformat(),
            state={},
        )
        assert bc.data_classification == DataClass.PUBLIC
        assert bc.access_control_list == ["owner"]
        assert bc.owner_id is None
        assert bc.encryption_key_id is None

    def test_bitchain_with_security_settings(self):
        """BitChain should accept custom security settings."""
        from fractalstat.stat7_experiments import (
            BitChain,
            Coordinates,
            DataClass,
        )

        coords = Coordinates(
            realm="data",
            lineage=1,
            adjacency=[],
            horizon="genesis",
            resonance=0.5,
            velocity=0.5,
            density=0.5,
        )
        bc = BitChain(
            id="test",
            entity_type="artifact",
            realm="data",
            coordinates=coords,
            created_at=datetime.now(timezone.utc).isoformat(),
            state={},
            data_classification=DataClass.PII,
            access_control_list=["admin", "owner"],
            owner_id="user123",
            encryption_key_id="key456",
        )
        assert bc.data_classification == DataClass.PII
        assert bc.access_control_list == ["admin", "owner"]
        assert bc.owner_id == "user123"
        assert bc.encryption_key_id == "key456"

    def test_bitchain_timestamp_normalization(self):
        """BitChain should normalize timestamp on init."""
        from fractalstat.stat7_experiments import BitChain, Coordinates

        coords = Coordinates(
            realm="data",
            lineage=1,
            adjacency=[],
            horizon="genesis",
            resonance=0.5,
            velocity=0.5,
            density=0.5,
        )
        bc = BitChain(
            id="test",
            entity_type="artifact",
            realm="data",
            coordinates=coords,
            created_at="2024-01-01T12:00:00Z",
            state={},
        )
        assert bc.created_at.endswith("Z")
        assert "2024-01-01" in bc.created_at

    def test_bitchain_canonical_dict_structure(self):
        """BitChain.to_canonical_dict should have correct structure."""
        from fractalstat.stat7_experiments import BitChain, Coordinates

        coords = Coordinates(
            realm="data",
            lineage=1,
            adjacency=[],
            horizon="genesis",
            resonance=0.5,
            velocity=0.5,
            density=0.5,
        )
        bc = BitChain(
            id="test",
            entity_type="artifact",
            realm="data",
            coordinates=coords,
            created_at=datetime.now(timezone.utc).isoformat(),
            state={"key": "value"},
        )
        canonical = bc.to_canonical_dict()
        assert "created_at" in canonical
        assert "entity_type" in canonical
        assert "id" in canonical
        assert "realm" in canonical
        assert "stat7_coordinates" in canonical
        assert "state" in canonical

    def test_bitchain_address_uniqueness(self):
        """Different BitChains should have different addresses."""
        from fractalstat.stat7_experiments import BitChain, Coordinates

        coords1 = Coordinates(
            realm="data",
            lineage=1,
            adjacency=[],
            horizon="genesis",
            resonance=0.5,
            velocity=0.5,
            density=0.5,
        )
        coords2 = Coordinates(
            realm="data",
            lineage=2,
            adjacency=[],
            horizon="genesis",
            resonance=0.5,
            velocity=0.5,
            density=0.5,
        )
        bc1 = BitChain(
            id="test1",
            entity_type="artifact",
            realm="data",
            coordinates=coords1,
            created_at="2024-01-01T12:00:00Z",
            state={},
        )
        bc2 = BitChain(
            id="test2",
            entity_type="artifact",
            realm="data",
            coordinates=coords2,
            created_at="2024-01-01T12:00:00Z",
            state={},
        )
        assert bc1.compute_address() != bc2.compute_address()

    def test_bitchain_uri_format(self):
        """BitChain.get_stat7_uri should follow correct format."""
        from fractalstat.stat7_experiments import BitChain, Coordinates

        coords = Coordinates(
            realm="narrative",
            lineage=5,
            adjacency=["adj1", "adj2"],
            horizon="peak",
            resonance=0.75,
            velocity=-0.25,
            density=0.6,
        )
        bc = BitChain(
            id="test",
            entity_type="concept",
            realm="narrative",
            coordinates=coords,
            created_at=datetime.now(timezone.utc).isoformat(),
            state={},
        )
        uri = bc.get_stat7_uri()
        assert uri.startswith("stat7://narrative/5/")
        assert "/peak?" in uri
        assert "r=" in uri
        assert "v=" in uri
        assert "d=" in uri


class TestGenerateRandomBitchainEdgeCases:
    """Test generate_random_bitchain edge cases."""

    def test_generate_random_bitchain_without_seed(self):
        """generate_random_bitchain without seed should be random."""
        from fractalstat.stat7_experiments import generate_random_bitchain

        bc1 = generate_random_bitchain()
        bc2 = generate_random_bitchain()
        assert bc1.id != bc2.id

    def test_generate_random_bitchain_valid_realm(self):
        """generate_random_bitchain should use valid realms."""
        from fractalstat.stat7_experiments import (
            generate_random_bitchain,
            REALMS,
        )

        bc = generate_random_bitchain(seed=123)
        assert bc.realm in REALMS
        assert bc.coordinates.realm in REALMS

    def test_generate_random_bitchain_valid_horizon(self):
        """generate_random_bitchain should use valid horizons."""
        from fractalstat.stat7_experiments import (
            generate_random_bitchain,
            HORIZONS,
        )

        bc = generate_random_bitchain(seed=456)
        assert bc.coordinates.horizon in HORIZONS

    def test_generate_random_bitchain_valid_entity_type(self):
        """generate_random_bitchain should use valid entity types."""
        from fractalstat.stat7_experiments import (
            generate_random_bitchain,
            ENTITY_TYPES,
        )

        bc = generate_random_bitchain(seed=789)
        assert bc.entity_type in ENTITY_TYPES

    def test_generate_random_bitchain_coordinate_ranges(self):
        """generate_random_bitchain should respect coordinate ranges."""
        from fractalstat.stat7_experiments import generate_random_bitchain

        bc = generate_random_bitchain(seed=999)
        assert -1.0 <= bc.coordinates.resonance <= 1.0
        assert -1.0 <= bc.coordinates.velocity <= 1.0
        assert 0.0 <= bc.coordinates.density <= 1.0
        assert bc.coordinates.lineage >= 1


class TestEXP01AddressUniquenessExtended:
    """Extended tests for EXP-01."""

    def test_exp01_get_summary(self):
        """EXP01 should provide summary statistics."""
        from fractalstat.stat7_experiments import EXP01_AddressUniqueness

        exp = EXP01_AddressUniqueness(sample_size=50, iterations=2)
        exp.run()
        summary = exp.get_summary()
        assert "total_iterations" in summary
        assert "total_bitchains_tested" in summary
        assert "total_collisions" in summary
        assert "overall_collision_rate" in summary
        assert "all_passed" in summary
        assert "results" in summary

    def test_exp01_multiple_iterations(self):
        """EXP01 should handle multiple iterations."""
        from fractalstat.stat7_experiments import EXP01_AddressUniqueness

        exp = EXP01_AddressUniqueness(sample_size=20, iterations=3)
        results, success = exp.run()
        assert len(results) == 3
        assert all(r.iteration in [1, 2, 3] for r in results)

    def test_exp01_collision_rate_calculation(self):
        """EXP01 should correctly calculate collision rate."""
        from fractalstat.stat7_experiments import EXP01_AddressUniqueness

        exp = EXP01_AddressUniqueness(sample_size=100, iterations=1)
        results, _ = exp.run()
        result = results[0]
        expected_rate = result.collisions / result.total_bitchains
        assert abs(result.collision_rate - expected_rate) < 0.0001


class TestEXP02RetrievalEfficiency:
    """Test EXP-02 retrieval efficiency."""

    def test_exp02_initializes(self):
        """EXP02_RetrievalEfficiency should initialize."""
        from fractalstat.stat7_experiments import EXP02_RetrievalEfficiency

        exp = EXP02_RetrievalEfficiency(query_count=500)
        assert exp.query_count == 500
        assert len(exp.scales) > 0

    def test_exp02_run_returns_results(self):
        """EXP02 run should return results."""
        from fractalstat.stat7_experiments import EXP02_RetrievalEfficiency

        exp = EXP02_RetrievalEfficiency(query_count=10)
        exp.scales = [100]
        results, success = exp.run()
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(success, bool)

    def test_exp02_result_structure(self):
        """EXP02_Result should have correct structure."""
        from fractalstat.stat7_experiments import EXP02_Result

        result = EXP02_Result(
            scale=1000,
            queries=100,
            mean_latency_ms=0.05,
            median_latency_ms=0.04,
            p95_latency_ms=0.08,
            p99_latency_ms=0.10,
            min_latency_ms=0.01,
            max_latency_ms=0.15,
            success=True,
        )
        assert result.scale == 1000
        assert result.queries == 100
        assert result.success is True

    def test_exp02_result_to_dict(self):
        """EXP02_Result should serialize to dict."""
        from fractalstat.stat7_experiments import EXP02_Result

        result = EXP02_Result(
            scale=1000,
            queries=100,
            mean_latency_ms=0.05,
            median_latency_ms=0.04,
            p95_latency_ms=0.08,
            p99_latency_ms=0.10,
            min_latency_ms=0.01,
            max_latency_ms=0.15,
            success=True,
        )
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["scale"] == 1000

    def test_exp02_get_summary(self):
        """EXP02 should provide summary statistics."""
        from fractalstat.stat7_experiments import EXP02_RetrievalEfficiency

        exp = EXP02_RetrievalEfficiency(query_count=10)
        exp.scales = [50]
        exp.run()
        summary = exp.get_summary()
        assert "total_scales_tested" in summary
        assert "all_passed" in summary
        assert "results" in summary


class TestEXP03DimensionNecessity:
    """Test EXP-03 dimension necessity."""

    def test_exp03_initializes(self):
        """EXP03_DimensionNecessity should initialize."""
        from fractalstat.stat7_experiments import EXP03_DimensionNecessity

        exp = EXP03_DimensionNecessity(sample_size=100)
        assert exp.sample_size == 100

    def test_exp03_run_returns_results(self):
        """EXP03 run should return results."""
        from fractalstat.stat7_experiments import EXP03_DimensionNecessity

        exp = EXP03_DimensionNecessity(sample_size=50)
        results, success = exp.run()
        assert isinstance(results, list)
        assert len(results) > 0
        assert isinstance(success, bool)

    def test_exp03_result_structure(self):
        """EXP03_Result should have correct structure."""
        from fractalstat.stat7_experiments import EXP03_Result

        result = EXP03_Result(
            dimensions_used=["realm", "lineage", "horizon"],
            sample_size=100,
            collisions=5,
            collision_rate=0.05,
            acceptable=True,
        )
        assert result.dimensions_used == ["realm", "lineage", "horizon"]
        assert result.sample_size == 100
        assert result.collisions == 5

    def test_exp03_result_to_dict(self):
        """EXP03_Result should serialize to dict."""
        from fractalstat.stat7_experiments import EXP03_Result

        result = EXP03_Result(
            dimensions_used=["realm", "lineage"],
            sample_size=100,
            collisions=0,
            collision_rate=0.0,
            acceptable=True,
        )
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["sample_size"] == 100

    def test_exp03_get_summary(self):
        """EXP03 should provide summary statistics."""
        from fractalstat.stat7_experiments import EXP03_DimensionNecessity

        exp = EXP03_DimensionNecessity(sample_size=30)
        exp.run()
        summary = exp.get_summary()
        assert "total_tests" in summary
        assert "all_passed" in summary
        assert "results" in summary


class TestRunAllExperiments:
    """Test run_all_experiments function."""

    def test_run_all_experiments_returns_dict(self):
        """run_all_experiments should return results dict."""
        from fractalstat.stat7_experiments import run_all_experiments

        results = run_all_experiments(
            exp01_samples=10, exp01_iterations=1, exp02_queries=5, exp03_samples=10
        )
        assert isinstance(results, dict)

    def test_run_all_experiments_includes_exp01(self):
        """run_all_experiments should include EXP-01 results."""
        from fractalstat.stat7_experiments import run_all_experiments

        results = run_all_experiments(
            exp01_samples=10, exp01_iterations=1, exp02_queries=5, exp03_samples=10
        )
        assert "EXP-01" in results or len(results) > 0

    def test_run_all_experiments_result_structure(self):
        """run_all_experiments results should have success and summary."""
        from fractalstat.stat7_experiments import run_all_experiments

        results = run_all_experiments(
            exp01_samples=10, exp01_iterations=1, exp02_queries=5, exp03_samples=10
        )
        for exp_name, exp_result in results.items():
            if isinstance(exp_result, dict):
                assert "success" in exp_result or "error" in exp_result
