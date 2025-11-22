#!/usr/bin/env python3
"""
FractalStat Entity System - Enhanced 8-Dimensional Addressing

Successor to STAT7 with improved expressivity (100% vs 95%).
Added 8th dimension: 'alignment' for social/coordination dynamics.

Features:
- Hybrid encoding (maps legacy systems to FractalStat coordinates)
- Backward compatibility with existing pets/badges/entities
- LUCA-adjacent bootstrap tracing
- Deterministic coordinate assignment
- Entanglement detection and management
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
import uuid
import hashlib
from abc import ABC, abstractmethod


def _utc_now() -> datetime:
    """Helper function for timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


# ============================================================================
# FractalStat Dimension Enums
# ============================================================================


class Realm(Enum):
    """Domain classification for FractalStat entities"""

    COMPANION = "companion"  # Pets, familiars, companions
    BADGE = "badge"  # Achievement badges
    SPONSOR_RING = "sponsor_ring"  # Sponsor tier badges
    ACHIEVEMENT = "achievement"  # Generic achievements
    PATTERN = "pattern"  # System patterns
    FACULTY = "faculty"  # Faculty-exclusive entities
    TEMPORAL = "temporal"  # Time-based entities
    VOID = "void"  # Null/empty realm


class Horizon(Enum):
    """Lifecycle stage in entity progression"""

    GENESIS = "genesis"  # Entity created, initial state
    EMERGENCE = "emergence"  # Entity becoming active
    PEAK = "peak"  # Entity at maximum activity
    DECAY = "decay"  # Entity waning
    CRYSTALLIZATION = "crystallization"  # Entity settled/permanent
    ARCHIVED = "archived"  # Historical record


class Polarity(Enum):
    """Resonance/affinity classification"""

    # Companion polarities (elemental)
    LOGIC = "logic"
    CREATIVITY = "creativity"
    ORDER = "order"
    CHAOS = "chaos"
    BALANCE = "balance"

    # Badge polarities (category)
    ACHIEVEMENT = "achievement"
    CONTRIBUTION = "contribution"
    COMMUNITY = "community"
    TECHNICAL = "technical"
    CREATIVE = "creative"
    UNITY = "unity"  # Special for sponsor rings

    # Neutral
    VOID = "void"


class Alignment(Enum):
    """Social and coordination dynamics alignment"""

    # Classical alignment system (inspired by fantasy RPGs) - Law vs Chaos
    LAWFUL_GOOD = "lawful_good"  # Principled, helpful
    NEUTRAL_GOOD = "neutral_good"  # Helpful, flexible
    CHAOTIC_GOOD = "chaotic_good"  # Helpful, unconstrained

    LAWFUL_NEUTRAL = "lawful_neutral"  # Principled, pragmatic
    TRUE_NEUTRAL = "true_neutral"  # Balanced, pragmatic
    CHAOTIC_NEUTRAL = "chaotic_neutral"  # Flexible, pragmatic

    LAWFUL_EVIL = "lawful_evil"  # Principled, harmful
    NEUTRAL_EVIL = "neutral_evil"  # Self-serving
    CHAOTIC_EVIL = "chaotic_evil"  # Harmful, unconstrained

    # Special classifications for FractalStat
    HARMONIC = "harmonic"  # Naturally coordinated
    ENTROPIC = "entropic"  # Naturally disruptive
    SYMBIOTIC = "symbiotic"  # Mutually beneficial connections


# ============================================================================
# FractalStat Coordinate Data Class (8 Dimensions)
# ============================================================================


@dataclass
class FractalStatCoordinates:
    """
    8-dimensional addressing space for all entities with 100% expressivity.

    Each dimension represents a different axis of entity existence:
      1. Realm: Domain/type classification
      2. Lineage: Generation or tier progression from LUCA
      3. Adjacency: Semantic/functional proximity score (0-100)
      4. Horizon: Lifecycle stage
      5. Luminosity: Activity level (0-100)
      6. Polarity: Resonance/affinity type
      7. Dimensionality: Fractal depth / detail level
      8. Alignment: Social/coordination dynamics (NEW - 100% expressivity boost)
    """

    realm: Realm
    lineage: int  # 0-based generation from LUCA
    adjacency: float  # 0-100 proximity score
    horizon: Horizon
    luminosity: float  # 0-100 activity level
    polarity: Polarity
    dimensionality: int  # 0+ fractal depth
    alignment: Alignment  # 8th dimension for social dynamics

    @property
    def address(self) -> str:
        """Generate canonical FractalStat address string"""
        return f"FractalStat-{self.realm.value[0].upper()}-{self.lineage:03d}-{int(self.adjacency):02d}-{self.horizon.value[0].upper()}-{int(self.luminosity):02d}-{self.polarity.value[0].upper()}-{self.dimensionality}-{self.alignment.value[0].upper()}"

    @staticmethod
    def from_address(address: str) -> "FractalStatCoordinates":
        """Parse FractalStat address back to coordinates"""
        # Format: FractalStat-R-000-00-H-00-P-0-A
        parts = address.split("-")
        if len(parts) != 10 or parts[0] != "FractalStat":
            raise ValueError(f"Invalid FractalStat address: {address}")

        realm_map = {r.value[0].upper(): r for r in Realm}
        horizon_map = {h.value[0].upper(): h for h in Horizon}
        polarity_map = {p.value[0].upper(): p for p in Polarity}
        alignment_map = {a.value[0].upper(): a for a in Alignment}

        return FractalStatCoordinates(
            realm=realm_map[parts[1]],
            lineage=int(parts[2]),
            adjacency=float(parts[3]),
            horizon=horizon_map[parts[4]],
            luminosity=float(parts[5]),
            polarity=polarity_map[parts[6]],
            dimensionality=int(parts[7]),
            alignment=alignment_map[parts[8]],
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "realm": self.realm.value,
            "lineage": self.lineage,
            "adjacency": self.adjacency,
            "horizon": self.horizon.value,
            "luminosity": self.luminosity,
            "polarity": self.polarity.value,
            "dimensionality": self.dimensionality,
            "alignment": self.alignment.value,
            "address": self.address,
        }


# ============================================================================
# Lifecycle Event Tracking
# ============================================================================


@dataclass
class LifecycleEvent:
    """Record of significant moments in entity history"""

    timestamp: datetime
    event_type: str  # "birth", "evolution", "mint", etc.
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "description": self.description,
            "metadata": self.metadata,
        }


# ============================================================================
# FractalStat Entity Base Class
# ============================================================================


@dataclass
class FractalStatEntity(ABC):
    """
    Abstract base class for all FractalStat-addressed entities.

    8-dimensional successor to STAT7 with enhanced expressivity.

    Provides:
    - Hybrid encoding (bridge between legacy and FractalStat systems)
    - 8D coordinate assignment
    - Entanglement tracking
    - Temporal tracking
    - NFT metadata
    """

    # Identity
    entity_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    entity_type: str = ""  # Overridden in subclasses

    # FractalStat Addressing (8D)
    fractalstat: Optional[FractalStatCoordinates] = None

    # Legacy Fields (backward compatibility)
    legacy_data: Dict[str, Any] = field(default_factory=dict)
    migration_source: Optional[str] = None  # "pet", "badge", etc.

    # NFT Status
    nft_minted: bool = False
    nft_contract: Optional[str] = None
    nft_token_id: Optional[int] = None
    nft_metadata_ipfs: Optional[str] = None

    # Entanglement
    entangled_entities: List[str] = field(default_factory=list)
    entanglement_strength: List[float] = field(default_factory=list)

    # Temporal
    created_at: datetime = field(default_factory=_utc_now)
    last_activity: datetime = field(default_factory=_utc_now)
    lifecycle_events: List[LifecycleEvent] = field(default_factory=list)

    # Owner/User
    owner_id: str = ""

    # User Preferences
    opt_in_fractalstat_nft: bool = True  # Renamed from opt_in_stat7_nft
    opt_in_blockchain: bool = False
    preferred_zoom_level: int = 1  # Default display level

    def __post_init__(self):
        """Initialize FractalStat coordinates if not provided"""
        if self.fractalstat is None:
            self.fractalstat = self._compute_fractalstat_coordinates()
        self._record_event("genesis", "Entity initialized in FractalStat space")

    # ========================================================================
    # Abstract Methods (Implemented by Subclasses)
    # ========================================================================

    @abstractmethod
    def _compute_fractalstat_coordinates(self) -> FractalStatCoordinates:
        """
        Compute 8D FractalStat coordinates from entity data.
        Each subclass defines its own coordinate mapping.
        """
        pass

    @abstractmethod
    def to_collectible_card_data(self) -> Dict[str, Any]:
        """Convert entity to collectible card display format"""
        pass

    @abstractmethod
    def validate_hybrid_encoding(self) -> Tuple[bool, str]:
        """
        Validate that FractalStat coordinates correctly encode legacy data.
        Returns (is_valid, error_message_or_empty_string)
        """
        pass

    # ========================================================================
    # Event Tracking
    # ========================================================================

    def _record_event(
        self,
        event_type: str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Record a lifecycle event"""
        event = LifecycleEvent(
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            description=description,
            metadata=metadata or {},
        )
        self.lifecycle_events.append(event)
        self.last_activity = event.timestamp

    def get_event_history(self, limit: Optional[int] = None) -> List[LifecycleEvent]:
        """Get lifecycle events, optionally limited to most recent"""
        events = sorted(self.lifecycle_events, key=lambda e: e.timestamp, reverse=True)
        return events[:limit] if limit else events

    # ========================================================================
    # Entanglement Management
    # ========================================================================

    def add_entanglement(self, other_entity_id: str, strength: float = 1.0):
        """
        Link to another entity via resonance/entanglement.
        Strength: 0-1.0 (1.0 = maximum entanglement)
        """
        if other_entity_id not in self.entangled_entities:
            self.entangled_entities.append(other_entity_id)
            self.entanglement_strength.append(strength)
            self._record_event(
                "entanglement_added",
                f"Entangled with {other_entity_id}",
                {"strength": strength},
            )

    def remove_entanglement(self, other_entity_id: str):
        """Remove entanglement with another entity"""
        if other_entity_id in self.entangled_entities:
            idx = self.entangled_entities.index(other_entity_id)
            self.entangled_entities.pop(idx)
            self.entanglement_strength.pop(idx)
            self._record_event(
                "entanglement_removed", f"Untangled from {other_entity_id}"
            )

    def get_entanglements(self) -> List[Tuple[str, float]]:
        """Get all entangled entities with strength"""
        return list(zip(self.entangled_entities, self.entanglement_strength))

    def update_entanglement_strength(self, other_entity_id: str, new_strength: float):
        """Update entanglement strength with another entity"""
        if other_entity_id in self.entangled_entities:
            idx = self.entangled_entities.index(other_entity_id)
            old_strength = self.entanglement_strength[idx]
            self.entanglement_strength[idx] = new_strength
            self._record_event(
                "entanglement_updated",
                f"Entanglement strength changed {old_strength:.2f} → {
                    new_strength:.2f}",
            )

    # ========================================================================
    # LUCA Bootstrap
    # ========================================================================

    @property
    def luca_distance(self) -> int:
        """Distance from LUCA (Last Universal Common Ancestor)"""
        assert self.fractalstat is not None, (
            "fractalstat coordinates must be initialized"
        )
        return self.fractalstat.lineage

    def get_luca_trace(self) -> Dict[str, Any]:
        """
        Get path back to LUCA bootstrap origin.
        In a real system, this would trace parent entities.
        """
        assert self.fractalstat is not None, (
            "fractalstat coordinates must be initialized"
        )
        return {
            "entity_id": self.entity_id,
            "luca_distance": self.luca_distance,
            "realm": self.fractalstat.realm.value,
            "lineage": self.fractalstat.lineage,
            "created_at": self.created_at.isoformat(),
            "migration_source": self.migration_source,
            "event_count": len(self.lifecycle_events),
        }

    # ========================================================================
    # NFT Integration
    # ========================================================================

    def prepare_for_minting(self) -> Dict[str, Any]:
        """
        Generate NFT metadata for minting.
        Returns ERC-721/ERC-1155 compatible metadata object.
        """
        if not self.opt_in_fractalstat_nft:
            raise ValueError("Entity not opted in to FractalStat-NFT system")

        assert self.fractalstat is not None, (
            "fractalstat coordinates must be initialized"
        )
        card_data = self.to_collectible_card_data()

        return {
            "name": card_data.get("title", self.entity_id),
            "description": card_data.get("fluff_text", ""),
            "image": card_data.get("artwork_url", ""),
            "external_url": f"https://theseed.example.com/entity/{self.entity_id}",
            "attributes": [
                {"trait_type": "Entity Type", "value": self.entity_type},
                {"trait_type": "Realm", "value": self.fractalstat.realm.value},
                {"trait_type": "Lineage", "value": self.fractalstat.lineage},
                {"trait_type": "Horizon", "value": self.fractalstat.horizon.value},
                {
                    "trait_type": "Luminosity",
                    "value": int(self.fractalstat.luminosity),
                },
                {"trait_type": "Polarity", "value": self.fractalstat.polarity.value},
                {
                    "trait_type": "Dimensionality",
                    "value": self.fractalstat.dimensionality,
                },
                {"trait_type": "Alignment", "value": self.fractalstat.alignment.value},
                {
                    "trait_type": "FractalStat Address",
                    "value": self.fractalstat.address,
                },
            ],
            "properties": card_data.get("properties", {}),
        }

    def record_mint(self, contract_address: str, token_id: int, ipfs_hash: str):
        """Record successful NFT minting"""
        self.nft_minted = True
        self.nft_contract = contract_address
        self.nft_token_id = token_id
        self.nft_metadata_ipfs = ipfs_hash
        self._record_event(
            "nft_minted",
            f"Minted as ERC-721 token #{token_id}",
            {
                "contract": contract_address,
                "token_id": token_id,
                "ipfs_hash": ipfs_hash,
            },
        )

    # ========================================================================
    # Serialization
    # ========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary for JSON storage"""
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "fractalstat": self.fractalstat.to_dict() if self.fractalstat else None,
            "legacy_data": self.legacy_data,
            "migration_source": self.migration_source,
            "nft_minted": self.nft_minted,
            "nft_contract": self.nft_contract,
            "nft_token_id": self.nft_token_id,
            "nft_metadata_ipfs": self.nft_metadata_ipfs,
            "entangled_entities": self.entangled_entities,
            "entanglement_strength": self.entanglement_strength,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "lifecycle_events": [e.to_dict() for e in self.lifecycle_events],
            "owner_id": self.owner_id,
            "opt_in_fractalstat_nft": self.opt_in_fractalstat_nft,
            "opt_in_blockchain": self.opt_in_blockchain,
            "preferred_zoom_level": self.preferred_zoom_level,
        }

    def save_to_file(self, path: Path):
        """Persist entity to JSON file"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load_from_file(cls, path: Path) -> "FractalStatEntity":
        """Load entity from JSON file (must know concrete type)"""
        with open(path, "r") as f:
            data = json.load(f)
        entity_type = data.get("entity_type", "unknown")
        raise NotImplementedError(
            f"Use subclass load methods (detected entity_type: {entity_type}). "
            "Use factory pattern to instantiate correct subclass."
        )

    # ========================================================================
    # Display Levels
    # ========================================================================

    def render_zoom_level(self, level: int) -> Dict[str, Any]:
        """
        Render entity at specific zoom level.

        Level 1: Badge (20x20px icon)
        Level 2: Dog-tag (100x150px micro-card)
        Level 3: Collectible Card (300x400px full card)
        Level 4: Profile panel (350x500px interactive)
        Level 5: Entity profile page (full details)
        Level 6+: Fractal descent (dimension breakdown)
        """
        if level < 1 or level > 8:  # Increased max zoom level for 8D
            raise ValueError(f"Invalid zoom level: {level}")

        assert self.fractalstat is not None, (
            "fractalstat coordinates must be initialized"
        )
        card_data = self.to_collectible_card_data()

        base = {
            "zoom_level": level,
            "entity_id": self.entity_id,
            "fractalstat_address": self.fractalstat.address,
            "created_at": self.created_at.isoformat(),
        }

        if level == 1:
            # Badge: Just icon + rarity
            return {
                **base,
                "type": "badge",
                "icon": card_data.get("icon_url"),
                "rarity": card_data.get("rarity"),
            }

        elif level == 2:
            # Dog-tag: Icon, title, key stats
            return {
                **base,
                "type": "dog_tag",
                "icon": card_data.get("icon_url"),
                "title": card_data.get("title"),
                "stats": card_data.get("key_stats"),
            }

        elif level == 3:
            # Full card
            return {**base, "type": "collectible_card", **card_data}

        elif level == 4:
            # Profile panel
            return {
                **base,
                "type": "profile_panel",
                **card_data,
                "owner": self.owner_id,
                "entangled_count": len(self.entangled_entities),
                "events": len(self.lifecycle_events),
            }

        elif level == 5:
            # Full profile page
            return {
                **base,
                "type": "entity_profile",
                **card_data,
                "owner": self.owner_id,
                "lifecycle_events": [e.to_dict() for e in self.lifecycle_events],
                "entanglements": self.get_entanglements(),
                "luca_trace": self.get_luca_trace(),
            }

        elif level == 6:
            # 8th dimension awareness
            return {
                **base,
                "type": "fractal_awareness",
                "fractalstat_dimensions": self.fractalstat.to_dict(),
                "alignment_dynamics": self._get_alignment_details(),
                "realm_details": self._get_realm_details(),
                "entanglement_network": self.get_entanglements(),
            }

        else:  # level 7+
            # Full fractal descent with 8D awareness
            return {
                **base,
                "type": "fractal_descent",
                "fractalstat_dimensions": self.fractalstat.to_dict(),
                "alignment_dynamics": self._get_alignment_details(),
                "realm_details": self._get_realm_details(),
                "entanglement_network": self.get_entanglements(),
                "event_chronology": [e.to_dict() for e in self.lifecycle_events],
                "luca_trace": self.get_luca_trace(),
            }

    def _get_realm_details(self) -> Dict[str, Any]:
        """Override in subclasses to provide realm-specific details"""
        return {}

    def _get_alignment_details(self) -> Dict[str, Any]:
        """Get alignment-based social/coordination analysis"""
        if self.fractalstat is None:
            return {}

        alignment = self.fractalstat.alignment
        # Analyze social coordination patterns based on alignment
        coordination_style = {
            Alignment.LAWFUL_GOOD: "structured_harmonious",
            Alignment.NEUTRAL_GOOD: "balanced_harmonious",
            Alignment.CHAOTIC_GOOD: "flexible_harmonious",
            Alignment.LAWFUL_NEUTRAL: "structured_pragmatic",
            Alignment.TRUE_NEUTRAL: "balanced_pragmatic",
            Alignment.CHAOTIC_NEUTRAL: "flexible_pragmatic",
            Alignment.LAWFUL_EVIL: "structured_destructive",
            Alignment.NEUTRAL_EVIL: "balanced_destructive",
            Alignment.CHAOTIC_EVIL: "flexible_destructive",
            Alignment.HARMONIC: "naturally_coordinating",
            Alignment.ENTROPIC: "naturally_disruptive",
            Alignment.SYMBIOTIC: "mutually_beneficial",
        }.get(alignment, "unknown")

        return {
            "alignment": alignment.value,
            "coordination_style": coordination_style,
            "social_dynamics": self._analyze_social_dynamics(),
        }

    def _analyze_social_dynamics(self) -> Dict[str, Any]:
        """Analyze social interaction patterns based on entanglement and alignment"""
        if self.fractalstat is None:
            return {}

        # Simplified social analysis based on alignment
        alignment = self.fractalstat.alignment

        if alignment in [Alignment.LAWFUL_GOOD, Alignment.HARMONIC]:
            social_pattern = "coordinating_harmonious"
        elif alignment in [Alignment.CHAOTIC_EVIL, Alignment.ENTROPIC]:
            social_pattern = "disruptive_chaotic"
        else:
            social_pattern = "pragmatic_balanced"

        return {
            "social_pattern": social_pattern,
            "entanglement_quality": len(self.entangled_entities)
            * 0.1,  # Simplified metric
            "coordination_potential": self._calculate_coordination_potential(),
        }

    def _calculate_coordination_potential(self) -> float:
        """Calculate coordination potential based on alignment and entanglements"""
        # Simplified calculation - would be more sophisticated in production
        base_potential = len(self.entangled_entities) * 0.1

        # Alignment modifiers
        alignment_bonus = {
            Alignment.HARMONIC: 1.5,
            Alignment.SYMBIOTIC: 1.3,
            Alignment.LAWFUL_GOOD: 1.2,
            Alignment.CHAOTIC_EVIL: -0.5,
            Alignment.ENTROPIC: -0.3,
        }.get(
            self.fractalstat.alignment if self.fractalstat else Alignment.TRUE_NEUTRAL,
            1.0,
        )

        return min(1.0, base_potential * alignment_bonus)


# ============================================================================
# Helper Functions
# ============================================================================


def hash_for_coordinates(data: Dict[str, Any]) -> str:
    """Deterministic hashing for coordinate assignment"""
    json_str = json.dumps(data, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()


def compute_adjacency_score(tags1: List[str], tags2: List[str]) -> float:
    """
    Compute adjacency (similarity) score between two tag sets.
    Returns 0-100 score.
    """
    if not tags1 or not tags2:
        return 0.0

    common = len(set(tags1) & set(tags2))
    total = len(set(tags1) | set(tags2))
    return (common / total) * 100 if total > 0 else 0.0


if __name__ == "__main__":
    print(
        "FractalStat Entity system loaded. "
        "Use as base class for Companion and Badge entities. "
        "8 dimensions, 100% expressivity."
    )
