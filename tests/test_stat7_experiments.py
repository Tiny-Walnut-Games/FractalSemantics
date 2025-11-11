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
