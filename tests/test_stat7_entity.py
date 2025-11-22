"""
Comprehensive unit tests for stat7_entity.py core module
"""

import pytest
import json
from datetime import datetime
from pathlib import Path
import tempfile


class TestRealmEnum:
    """Test Realm enum values and behavior."""

    def test_realm_enum_values(self):
        """Realm enum should have all expected values."""
        from fractalstat.stat7_entity import Realm

        assert Realm.COMPANION.value == "companion"
        assert Realm.BADGE.value == "badge"
        assert Realm.SPONSOR_RING.value == "sponsor_ring"
        assert Realm.ACHIEVEMENT.value == "achievement"
        assert Realm.PATTERN.value == "pattern"
        assert Realm.FACULTY.value == "faculty"
        assert Realm.VOID.value == "void"

    def test_realm_enum_membership(self):
        """Realm enum should support membership testing."""
        from fractalstat.stat7_entity import Realm

        assert Realm.COMPANION in Realm
        assert Realm.BADGE in Realm

    def test_realm_enum_iteration(self):
        """Realm enum should be iterable."""
        from fractalstat.stat7_entity import Realm

        realms = list(Realm)
        assert len(realms) == 7
        assert Realm.COMPANION in realms


class TestHorizonEnum:
    """Test Horizon enum values and behavior."""

    def test_horizon_enum_values(self):
        """Horizon enum should have all lifecycle stages."""
        from fractalstat.stat7_entity import Horizon

        assert Horizon.GENESIS.value == "genesis"
        assert Horizon.EMERGENCE.value == "emergence"
        assert Horizon.PEAK.value == "peak"
        assert Horizon.DECAY.value == "decay"
        assert Horizon.CRYSTALLIZATION.value == "crystallization"
        assert Horizon.ARCHIVED.value == "archived"

    def test_horizon_enum_count(self):
        """Horizon enum should have exactly 6 stages."""
        from fractalstat.stat7_entity import Horizon

        assert len(list(Horizon)) == 6


class TestPolarityEnum:
    """Test Polarity enum values and behavior."""

    def test_polarity_companion_values(self):
        """Polarity should have companion elemental values."""
        from fractalstat.stat7_entity import Polarity

        assert Polarity.LOGIC.value == "logic"
        assert Polarity.CREATIVITY.value == "creativity"
        assert Polarity.ORDER.value == "order"
        assert Polarity.CHAOS.value == "chaos"
        assert Polarity.BALANCE.value == "balance"

    def test_polarity_badge_values(self):
        """Polarity should have badge category values."""
        from fractalstat.stat7_entity import Polarity

        assert Polarity.ACHIEVEMENT.value == "achievement"
        assert Polarity.CONTRIBUTION.value == "contribution"
        assert Polarity.COMMUNITY.value == "community"
        assert Polarity.TECHNICAL.value == "technical"
        assert Polarity.CREATIVE.value == "creative"
        assert Polarity.UNITY.value == "unity"

    def test_polarity_neutral_value(self):
        """Polarity should have void neutral value."""
        from fractalstat.stat7_entity import Polarity

        assert Polarity.VOID.value == "void"


class TestSTAT7Coordinates:
    """Test STAT7Coordinates dataclass."""

    def test_coordinates_initialization(self):
        """STAT7Coordinates should initialize with all dimensions."""
        from fractalstat.stat7_entity import (
            STAT7Coordinates,
            Realm,
            Horizon,
            Polarity,
        )

        coords = STAT7Coordinates(
            realm=Realm.COMPANION,
            lineage=5,
            adjacency=75.5,
            horizon=Horizon.PEAK,
            luminosity=90.0,
            polarity=Polarity.LOGIC,
            dimensionality=3,
        )

        assert coords.realm == Realm.COMPANION
        assert coords.lineage == 5
        assert coords.adjacency == 75.5
        assert coords.horizon == Horizon.PEAK
        assert coords.luminosity == 90.0
        assert coords.polarity == Polarity.LOGIC
        assert coords.dimensionality == 3

    def test_coordinates_address_generation(self):
        """address property should generate canonical STAT7 address."""
        from fractalstat.stat7_entity import (
            STAT7Coordinates,
            Realm,
            Horizon,
            Polarity,
        )

        coords = STAT7Coordinates(
            realm=Realm.COMPANION,
            lineage=5,
            adjacency=75.5,
            horizon=Horizon.PEAK,
            luminosity=90.0,
            polarity=Polarity.LOGIC,
            dimensionality=3,
        )

        address = coords.address
        assert address == "STAT7-C-005-75-P-90-L-3"

    def test_coordinates_address_format(self):
        """address should follow STAT7-R-LLL-AA-H-LL-P-D format."""
        from fractalstat.stat7_entity import (
            STAT7Coordinates,
            Realm,
            Horizon,
            Polarity,
        )

        coords = STAT7Coordinates(
            realm=Realm.BADGE,
            lineage=0,
            adjacency=0.0,
            horizon=Horizon.GENESIS,
            luminosity=0.0,
            polarity=Polarity.ACHIEVEMENT,
            dimensionality=0,
        )

        address = coords.address
        parts = address.split("-")

        assert len(parts) == 8
        assert parts[0] == "STAT7"
        assert parts[1] == "B"
        assert parts[2] == "000"
        assert parts[3] == "00"
        assert parts[4] == "G"
        assert parts[5] == "00"
        assert parts[6] == "A"
        assert parts[7] == "0"

    def test_coordinates_from_address_valid(self):
        """from_address should parse valid STAT7 address."""
        from fractalstat.stat7_entity import (
            STAT7Coordinates,
            Realm,
            Horizon,
            Polarity,
        )

        address = "STAT7-C-005-75-P-90-L-3"
        coords = STAT7Coordinates.from_address(address)

        assert coords.realm == Realm.COMPANION
        assert coords.lineage == 5
        assert coords.adjacency == 75.0
        assert coords.horizon == Horizon.PEAK
        assert coords.luminosity == 90.0
        assert coords.polarity == Polarity.LOGIC
        assert coords.dimensionality == 3

    def test_coordinates_from_address_invalid_prefix(self):
        """from_address should raise ValueError for invalid prefix."""
        from fractalstat.stat7_entity import STAT7Coordinates

        with pytest.raises(ValueError, match="Invalid STAT7 address"):
            STAT7Coordinates.from_address("INVALID-C-005-75-P-90-L-3")

    def test_coordinates_from_address_invalid_parts(self):
        """from_address should raise ValueError for wrong number of parts."""
        from fractalstat.stat7_entity import STAT7Coordinates

        with pytest.raises(ValueError, match="Invalid STAT7 address"):
            STAT7Coordinates.from_address("STAT7-C-005")

    def test_coordinates_roundtrip(self):
        """Coordinates should survive address roundtrip."""
        from fractalstat.stat7_entity import (
            STAT7Coordinates,
            Realm,
            Horizon,
            Polarity,
        )

        original = STAT7Coordinates(
            realm=Realm.SPONSOR_RING,
            lineage=10,
            adjacency=50.0,
            horizon=Horizon.CRYSTALLIZATION,
            luminosity=100.0,
            polarity=Polarity.UNITY,
            dimensionality=7,
        )

        address = original.address
        restored = STAT7Coordinates.from_address(address)

        assert restored.realm == original.realm
        assert restored.lineage == original.lineage
        assert restored.adjacency == original.adjacency
        assert restored.horizon == original.horizon
        assert restored.luminosity == original.luminosity
        assert restored.polarity == original.polarity
        assert restored.dimensionality == original.dimensionality

    def test_coordinates_to_dict(self):
        """to_dict should convert coordinates to dictionary."""
        from fractalstat.stat7_entity import (
            STAT7Coordinates,
            Realm,
            Horizon,
            Polarity,
        )

        coords = STAT7Coordinates(
            realm=Realm.PATTERN,
            lineage=2,
            adjacency=33.3,
            horizon=Horizon.EMERGENCE,
            luminosity=66.6,
            polarity=Polarity.CHAOS,
            dimensionality=1,
        )

        data = coords.to_dict()

        assert data["realm"] == "pattern"
        assert data["lineage"] == 2
        assert data["adjacency"] == 33.3
        assert data["horizon"] == "emergence"
        assert data["luminosity"] == 66.6
        assert data["polarity"] == "chaos"
        assert data["dimensionality"] == 1
        assert "address" in data


class TestLifecycleEvent:
    """Test LifecycleEvent dataclass."""

    def test_lifecycle_event_initialization(self):
        """LifecycleEvent should initialize with required fields."""
        from fractalstat.stat7_entity import LifecycleEvent

        timestamp = datetime.now()
        event = LifecycleEvent(
            timestamp=timestamp,
            event_type="birth",
            description="Entity created",
        )

        assert event.timestamp == timestamp
        assert event.event_type == "birth"
        assert event.description == "Entity created"
        assert event.metadata == {}

    def test_lifecycle_event_with_metadata(self):
        """LifecycleEvent should accept metadata."""
        from fractalstat.stat7_entity import LifecycleEvent

        metadata = {"key": "value", "count": 42}
        event = LifecycleEvent(
            timestamp=datetime.now(),
            event_type="evolution",
            description="Entity evolved",
            metadata=metadata,
        )

        assert event.metadata == metadata

    def test_lifecycle_event_to_dict(self):
        """to_dict should convert event to dictionary."""
        from fractalstat.stat7_entity import LifecycleEvent

        timestamp = datetime(2025, 1, 1, 12, 0, 0)
        event = LifecycleEvent(
            timestamp=timestamp,
            event_type="mint",
            description="NFT minted",
            metadata={"token_id": 123},
        )

        data = event.to_dict()

        assert data["timestamp"] == timestamp.isoformat()
        assert data["event_type"] == "mint"
        assert data["description"] == "NFT minted"
        assert data["metadata"]["token_id"] == 123


class TestSTAT7EntityBase:
    """Test STAT7Entity abstract base class."""

    def create_concrete_entity(self):
        """Helper to create a concrete STAT7Entity subclass."""
        from fractalstat.stat7_entity import (
            STAT7Entity,
            STAT7Coordinates,
            Realm,
            Horizon,
            Polarity,
        )

        class ConcreteEntity(STAT7Entity):
            def _compute_stat7_coordinates(self):
                return STAT7Coordinates(
                    realm=Realm.COMPANION,
                    lineage=0,
                    adjacency=0.0,
                    horizon=Horizon.GENESIS,
                    luminosity=0.0,
                    polarity=Polarity.BALANCE,
                    dimensionality=0,
                )

            def to_collectible_card_data(self):
                return {
                    "title": "Test Entity",
                    "fluff_text": "Test description",
                    "icon_url": "http://example.com/icon.png",
                    "artwork_url": "http://example.com/art.png",
                    "rarity": "common",
                    "key_stats": {"stat1": 10},
                    "properties": {"prop1": "value1"},
                }

            def validate_hybrid_encoding(self):
                return (True, "Valid")

            def get_luca_trace(self):
                return []

        return ConcreteEntity

    def test_entity_default_initialization(self):
        """STAT7Entity should initialize with default values."""
        ConcreteEntity = self.create_concrete_entity()
        entity = ConcreteEntity()

        assert entity.entity_id is not None
        assert len(entity.entity_id) > 0
        assert entity.entity_type == ""
        assert entity.stat7 is not None
        assert entity.legacy_data == {}
        assert entity.migration_source is None
        assert entity.nft_minted is False
        assert entity.nft_contract is None
        assert entity.nft_token_id is None
        assert entity.nft_metadata_ipfs is None
        assert entity.entangled_entities == []
        assert entity.entanglement_strength == []
        assert isinstance(entity.created_at, datetime)
        assert isinstance(entity.last_activity, datetime)
        assert len(entity.lifecycle_events) >= 1
        assert entity.owner_id == ""
        assert entity.opt_in_stat7_nft is True
        assert entity.opt_in_blockchain is False
        assert entity.preferred_zoom_level == 1

    def test_entity_custom_entity_id(self):
        """STAT7Entity should accept custom entity_id."""
        ConcreteEntity = self.create_concrete_entity()
        custom_id = "custom-entity-123"
        entity = ConcreteEntity(entity_id=custom_id)

        assert entity.entity_id == custom_id
        assert entity.owner_id is None or isinstance(entity.owner_id, str)

    def test_entity_record_event(self):
        """_record_event should add lifecycle event."""
        ConcreteEntity = self.create_concrete_entity()
        entity = ConcreteEntity()

        initial_count = len(entity.lifecycle_events)
        entity._record_event("test_event", "Test description", {"key": "value"})

        assert len(entity.lifecycle_events) == initial_count + 1
        event = entity.lifecycle_events[-1]
        assert event.event_type == "test_event"
        assert event.description == "Test description"
        assert event.metadata["key"] == "value"

    def test_entity_last_activity_tracking(self):
        """Entity should track last_activity timestamp."""
        from datetime import timezone

        ConcreteEntity = self.create_concrete_entity()
        entity = ConcreteEntity()

        assert isinstance(entity.last_activity, datetime)
        now_utc = datetime.now(timezone.utc)
        assert entity.last_activity <= now_utc

    def test_entity_add_entanglement(self):
        """add_entanglement should track entangled entities."""
        ConcreteEntity = self.create_concrete_entity()
        entity = ConcreteEntity()

        entity.add_entanglement("entity-123", 0.75)

        assert "entity-123" in entity.entangled_entities
        idx = entity.entangled_entities.index("entity-123")
        assert entity.entanglement_strength[idx] == 0.75

    def test_entity_add_entanglement_duplicate(self):
        """add_entanglement should not add duplicate entities."""
        ConcreteEntity = self.create_concrete_entity()
        entity = ConcreteEntity()

        entity.add_entanglement("entity-123", 0.5)
        entity.add_entanglement("entity-123", 0.9)

        assert entity.entangled_entities.count("entity-123") == 1

    def test_entity_remove_entanglement(self):
        """remove_entanglement should remove entangled entity."""
        ConcreteEntity = self.create_concrete_entity()
        entity = ConcreteEntity()

        entity.add_entanglement("entity-123", 0.75)
        entity.remove_entanglement("entity-123")

        assert "entity-123" not in entity.entangled_entities
        assert len(entity.entangled_entities) == len(entity.entanglement_strength)

    def test_entity_remove_entanglement_nonexistent(self):
        """remove_entanglement should handle nonexistent entity gracefully."""
        ConcreteEntity = self.create_concrete_entity()
        entity = ConcreteEntity()

        entity.remove_entanglement("nonexistent")

    def test_entity_get_entanglements(self):
        """get_entanglements should return list of entanglement tuples."""
        ConcreteEntity = self.create_concrete_entity()
        entity = ConcreteEntity()

        entity.add_entanglement("entity-1", 0.5)
        entity.add_entanglement("entity-2", 0.9)

        entanglements = entity.get_entanglements()

        assert len(entanglements) == 2
        assert ("entity-1", 0.5) in entanglements
        assert ("entity-2", 0.9) in entanglements

    def test_entity_collectible_card_data(self):
        """to_collectible_card_data should return card display data."""
        ConcreteEntity = self.create_concrete_entity()
        entity = ConcreteEntity()

        card_data = entity.to_collectible_card_data()

        assert card_data["title"] == "Test Entity"
        assert card_data["fluff_text"] == "Test description"
        assert "icon_url" in card_data
        assert "artwork_url" in card_data

    def test_entity_record_mint(self):
        """record_mint should update NFT status."""
        ConcreteEntity = self.create_concrete_entity()
        entity = ConcreteEntity()

        initial_events = len(entity.lifecycle_events)
        entity.record_mint("0x123abc", 42, "QmHash123")

        assert entity.nft_minted is True
        assert entity.nft_contract == "0x123abc"
        assert entity.nft_token_id == 42
        assert entity.nft_metadata_ipfs == "QmHash123"
        assert len(entity.lifecycle_events) == initial_events + 1
        assert entity.lifecycle_events[-1].event_type == "nft_minted"

    def test_entity_to_dict(self):
        """to_dict should convert entity to dictionary."""
        from fractalstat.stat7_entity import (
            STAT7Coordinates,
            Realm,
            Horizon,
            Polarity,
        )

        ConcreteEntity = self.create_concrete_entity()
        entity = ConcreteEntity()
        entity.entity_type = "TestEntity"
        entity.stat7 = STAT7Coordinates(
            realm=Realm.COMPANION,
            lineage=5,
            adjacency=75.0,
            horizon=Horizon.PEAK,
            luminosity=90.0,
            polarity=Polarity.LOGIC,
            dimensionality=3,
        )
        entity.owner_id = "user-123"

        data = entity.to_dict()

        assert data["entity_id"] == entity.entity_id
        assert data["entity_type"] == "TestEntity"
        assert data["stat7"] is not None
        assert data["owner_id"] == "user-123"
        assert "created_at" in data
        assert "last_activity" in data

    def test_entity_save_to_file(self):
        """save_to_file should persist entity to JSON."""
        ConcreteEntity = self.create_concrete_entity()
        entity = ConcreteEntity()
        entity.entity_type = "TestEntity"

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test_entity.json"
            entity.save_to_file(file_path)

            assert file_path.exists()

            with open(file_path, "r") as f:
                data = json.load(f)

            assert data["entity_id"] == entity.entity_id
            assert data["entity_type"] == "TestEntity"

    def test_entity_save_to_file_creates_directory(self):
        """save_to_file should create parent directories."""
        ConcreteEntity = self.create_concrete_entity()
        entity = ConcreteEntity()

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "subdir" / "test_entity.json"
            entity.save_to_file(file_path)

            assert file_path.exists()

    def test_entity_load_from_file_raises_not_implemented(self):
        """load_from_file should raise NotImplementedError."""
        ConcreteEntity = self.create_concrete_entity()

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.json"
            with open(file_path, "w") as f:
                json.dump({"entity_type": "test"}, f)

            with pytest.raises(NotImplementedError):
                ConcreteEntity.load_from_file(file_path)

    def test_entity_render_zoom_level_1_badge(self):
        """render_zoom_level(1) should return badge view."""
        from fractalstat.stat7_entity import (
            STAT7Coordinates,
            Realm,
            Horizon,
            Polarity,
        )

        ConcreteEntity = self.create_concrete_entity()
        entity = ConcreteEntity()
        entity.stat7 = STAT7Coordinates(
            realm=Realm.BADGE,
            lineage=0,
            adjacency=0.0,
            horizon=Horizon.GENESIS,
            luminosity=0.0,
            polarity=Polarity.ACHIEVEMENT,
            dimensionality=0,
        )

        view = entity.render_zoom_level(1)

        assert view["type"] == "badge"
        assert view["zoom_level"] == 1
        assert "icon" in view
        assert "rarity" in view

    def test_entity_render_zoom_level_2_dog_tag(self):
        """render_zoom_level(2) should return dog-tag view."""
        from fractalstat.stat7_entity import (
            STAT7Coordinates,
            Realm,
            Horizon,
            Polarity,
        )

        ConcreteEntity = self.create_concrete_entity()
        entity = ConcreteEntity()
        entity.stat7 = STAT7Coordinates(
            realm=Realm.COMPANION,
            lineage=1,
            adjacency=50.0,
            horizon=Horizon.EMERGENCE,
            luminosity=50.0,
            polarity=Polarity.BALANCE,
            dimensionality=1,
        )

        view = entity.render_zoom_level(2)

        assert view["type"] == "dog_tag"
        assert view["zoom_level"] == 2
        assert "icon" in view
        assert "title" in view
        assert "stats" in view

    def test_entity_render_zoom_level_3_card(self):
        """render_zoom_level(3) should return collectible card view."""
        from fractalstat.stat7_entity import (
            STAT7Coordinates,
            Realm,
            Horizon,
            Polarity,
        )

        ConcreteEntity = self.create_concrete_entity()
        entity = ConcreteEntity()
        entity.stat7 = STAT7Coordinates(
            realm=Realm.COMPANION,
            lineage=2,
            adjacency=75.0,
            horizon=Horizon.PEAK,
            luminosity=90.0,
            polarity=Polarity.LOGIC,
            dimensionality=2,
        )

        view = entity.render_zoom_level(3)

        assert view["type"] == "collectible_card"
        assert view["zoom_level"] == 3

    def test_entity_render_zoom_level_4_profile_panel(self):
        """render_zoom_level(4) should return profile panel view."""
        from fractalstat.stat7_entity import (
            STAT7Coordinates,
            Realm,
            Horizon,
            Polarity,
        )

        ConcreteEntity = self.create_concrete_entity()
        entity = ConcreteEntity()
        entity.stat7 = STAT7Coordinates(
            realm=Realm.COMPANION,
            lineage=3,
            adjacency=80.0,
            horizon=Horizon.PEAK,
            luminosity=95.0,
            polarity=Polarity.CREATIVITY,
            dimensionality=3,
        )
        entity.owner_id = "user-123"
        entity.add_entanglement("entity-1", 0.5)

        view = entity.render_zoom_level(4)

        assert view["type"] == "profile_panel"
        assert view["zoom_level"] == 4
        assert view["owner"] == "user-123"
        assert view["entangled_count"] == 1

    def test_entity_render_zoom_level_5_full_profile(self):
        """render_zoom_level(5) should return full entity profile."""
        from fractalstat.stat7_entity import (
            STAT7Coordinates,
            Realm,
            Horizon,
            Polarity,
        )

        ConcreteEntity = self.create_concrete_entity()
        entity = ConcreteEntity()
        entity.stat7 = STAT7Coordinates(
            realm=Realm.COMPANION,
            lineage=4,
            adjacency=85.0,
            horizon=Horizon.CRYSTALLIZATION,
            luminosity=100.0,
            polarity=Polarity.ORDER,
            dimensionality=4,
        )
        entity._record_event("test", "Test event")

        view = entity.render_zoom_level(5)

        assert view["type"] == "entity_profile"
        assert view["zoom_level"] == 5
        assert "lifecycle_events" in view
        assert "entanglements" in view
        assert "luca_trace" in view

    def test_entity_render_zoom_level_6_fractal_descent(self):
        """render_zoom_level(6+) should return fractal descent view."""
        from fractalstat.stat7_entity import (
            STAT7Coordinates,
            Realm,
            Horizon,
            Polarity,
        )

        ConcreteEntity = self.create_concrete_entity()
        entity = ConcreteEntity()
        entity.stat7 = STAT7Coordinates(
            realm=Realm.PATTERN,
            lineage=5,
            adjacency=90.0,
            horizon=Horizon.PEAK,
            luminosity=100.0,
            polarity=Polarity.CHAOS,
            dimensionality=5,
        )

        view = entity.render_zoom_level(6)

        assert view["type"] == "fractal_descent"
        assert view["zoom_level"] == 6
        assert "stat7_dimensions" in view
        assert "realm_details" in view
        assert "entanglement_network" in view
        assert "event_chronology" in view

    def test_entity_render_zoom_level_invalid_low(self):
        """render_zoom_level should raise ValueError for level < 1."""
        from fractalstat.stat7_entity import (
            STAT7Coordinates,
            Realm,
            Horizon,
            Polarity,
        )

        ConcreteEntity = self.create_concrete_entity()
        entity = ConcreteEntity()
        entity.stat7 = STAT7Coordinates(
            realm=Realm.COMPANION,
            lineage=0,
            adjacency=0.0,
            horizon=Horizon.GENESIS,
            luminosity=0.0,
            polarity=Polarity.BALANCE,
            dimensionality=0,
        )

        with pytest.raises(ValueError, match="Invalid zoom level"):
            entity.render_zoom_level(0)

    def test_entity_render_zoom_level_invalid_high(self):
        """render_zoom_level should raise ValueError for level > 7."""
        from fractalstat.stat7_entity import (
            STAT7Coordinates,
            Realm,
            Horizon,
            Polarity,
        )

        ConcreteEntity = self.create_concrete_entity()
        entity = ConcreteEntity()
        entity.stat7 = STAT7Coordinates(
            realm=Realm.COMPANION,
            lineage=0,
            adjacency=0.0,
            horizon=Horizon.GENESIS,
            luminosity=0.0,
            polarity=Polarity.BALANCE,
            dimensionality=0,
        )

        with pytest.raises(ValueError, match="Invalid zoom level"):
            entity.render_zoom_level(8)


class TestHelperFunctions:
    """Test helper functions."""

    def test_hash_for_coordinates_deterministic(self):
        """hash_for_coordinates should be deterministic."""
        from fractalstat.stat7_entity import hash_for_coordinates

        data = {"key1": "value1", "key2": 42}
        hash1 = hash_for_coordinates(data)
        hash2 = hash_for_coordinates(data)

        assert hash1 == hash2

    def test_hash_for_coordinates_different_data(self):
        """hash_for_coordinates should produce different hashes for different data."""
        from fractalstat.stat7_entity import hash_for_coordinates

        data1 = {"key": "value1"}
        data2 = {"key": "value2"}

        hash1 = hash_for_coordinates(data1)
        hash2 = hash_for_coordinates(data2)

        assert hash1 != hash2

    def test_hash_for_coordinates_order_independent(self):
        """hash_for_coordinates should be order-independent."""
        from fractalstat.stat7_entity import hash_for_coordinates

        data1 = {"a": 1, "b": 2, "c": 3}
        data2 = {"c": 3, "a": 1, "b": 2}

        hash1 = hash_for_coordinates(data1)
        hash2 = hash_for_coordinates(data2)

        assert hash1 == hash2

    def test_compute_adjacency_score_identical_tags(self):
        """compute_adjacency_score should return 100 for identical tags."""
        from fractalstat.stat7_entity import compute_adjacency_score

        tags = ["tag1", "tag2", "tag3"]
        score = compute_adjacency_score(tags, tags)

        assert score == 100.0

    def test_compute_adjacency_score_no_overlap(self):
        """compute_adjacency_score should return 0 for no overlap."""
        from fractalstat.stat7_entity import compute_adjacency_score

        tags1 = ["tag1", "tag2"]
        tags2 = ["tag3", "tag4"]
        score = compute_adjacency_score(tags1, tags2)

        assert score == 0.0

    def test_compute_adjacency_score_partial_overlap(self):
        """compute_adjacency_score should calculate partial overlap correctly."""
        from fractalstat.stat7_entity import compute_adjacency_score

        tags1 = ["tag1", "tag2", "tag3"]
        tags2 = ["tag2", "tag3", "tag4"]
        score = compute_adjacency_score(tags1, tags2)

        assert 0.0 < score < 100.0
        assert abs(score - 50.0) < 1.0

    def test_compute_adjacency_score_empty_tags1(self):
        """compute_adjacency_score should return 0 for empty first list."""
        from fractalstat.stat7_entity import compute_adjacency_score

        score = compute_adjacency_score([], ["tag1", "tag2"])

        assert score == 0.0

    def test_compute_adjacency_score_empty_tags2(self):
        """compute_adjacency_score should return 0 for empty second list."""
        from fractalstat.stat7_entity import compute_adjacency_score

        score = compute_adjacency_score(["tag1", "tag2"], [])

        assert score == 0.0

    def test_compute_adjacency_score_both_empty(self):
        """compute_adjacency_score should return 0 for both empty lists."""
        from fractalstat.stat7_entity import compute_adjacency_score

        score = compute_adjacency_score([], [])

        assert score == 0.0

    def test_compute_adjacency_score_duplicate_tags(self):
        """compute_adjacency_score should handle duplicate tags."""
        from fractalstat.stat7_entity import compute_adjacency_score

        tags1 = ["tag1", "tag1", "tag2"]
        tags2 = ["tag1", "tag2", "tag2"]
        score = compute_adjacency_score(tags1, tags2)

        assert score == 100.0
