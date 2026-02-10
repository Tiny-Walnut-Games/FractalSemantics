"""
Dynamic Enum System for FractalSemantics

Provides extensible enums with immutable core entries and registration system.
Core entries are locked and cannot be modified or removed through API calls.
Additional entries can be registered dynamically.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass, field
import threading
import logging

# Type checking imports removed since we use standard Enum classes
logger = logging.getLogger(__name__)


class DynamicEnumError(Exception):
    """Base exception for dynamic enum operations."""
    pass


class ImmutableCoreError(DynamicEnumError):
    """Raised when attempting to modify immutable core entries."""
    pass


class DuplicateRegistrationError(DynamicEnumError):
    """Raised when attempting to register a duplicate entry."""
    pass


class InvalidEntryError(DynamicEnumError):
    """Raised when attempting to register an invalid entry."""
    pass


@dataclass(frozen=True)
class EnumEntry:
    """
    Represents a single entry in a dynamic enum.

    Attributes:
        name: The enum entry name (must be uppercase, underscore-separated)
        value: The enum entry value (string representation)
        description: Optional human-readable description
        is_core: Whether this is an immutable core entry
        metadata: Additional metadata for the entry
    """
    name: str
    value: str
    description: Optional[str] = None
    is_core: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate entry after initialization."""
        if not self.name.replace('_', '').isupper():
            raise InvalidEntryError(f"Entry name must be uppercase with underscores: {self.name}")
        if not self.value.strip():
            raise InvalidEntryError(f"Entry value cannot be empty: {self.value}")


class DynamicEnumRegistry:
    """
    Registry for dynamic enums with thread-safe operations.

    Provides:
    - Immutable core entries (first N entries, configurable)
    - Dynamic registration of additional entries
    - Thread-safe operations
    - Validation and deduplication
    """

    def __init__(self, name: str, core_entries: List[EnumEntry], immutable_count: Optional[int] = None):
        """
        Initialize registry with core entries.

        Args:
            name: Name of the enum registry
            core_entries: List of core entries (these become immutable)
            immutable_count: Number of entries from start that are immutable.
                           If None, uses len(core_entries)
        """
        self.name = name
        self._lock = threading.RLock()
        self._entries: Dict[str, EnumEntry] = {}
        self._immutable_count = immutable_count if immutable_count is not None else len(core_entries)

        # Register core entries
        for entry in core_entries:
            self._register_entry(entry, is_initialization=True)

        # Mark first immutable_count entries as core
        self._mark_core_entries()

        logger.info(f"Initialized DynamicEnumRegistry '{name}' with {len(core_entries)} core entries "
                   f"({self._immutable_count} immutable)")

    def _register_entry(self, entry: EnumEntry, is_initialization: bool = False) -> None:
        """Register a single entry (internal method)."""
        if entry.name in self._entries:
            existing = self._entries[entry.name]
            if existing.value != entry.value:
                raise DuplicateRegistrationError(
                    f"Entry '{entry.name}' already exists with different value: "
                    f"'{existing.value}' vs '{entry.value}'"
                )
            # Same entry, just update metadata if not core
            if not existing.is_core:
                self._entries[entry.name] = entry
            return

        self._entries[entry.name] = entry

        if not is_initialization:
            logger.info(f"Registered {entry.name} = '{entry.value}' in {self.name}")

    def _mark_core_entries(self) -> None:
        """Mark the first immutable_count entries as core."""
        sorted_entries = sorted(self._entries.items(), key=lambda x: x[1].name)

        for i, (name, entry) in enumerate(sorted_entries):
            if i < self._immutable_count:
                # Create new entry with is_core=True
                core_entry = EnumEntry(
                    name=entry.name,
                    value=entry.value,
                    description=entry.description,
                    is_core=True,
                    metadata=entry.metadata
                )
                self._entries[name] = core_entry

    def register_entry(self, entry: EnumEntry) -> None:
        """
        Register a new entry in the enum.

        Args:
            entry: The entry to register

        Raises:
            ImmutableCoreError: If attempting to modify a core entry
            DuplicateRegistrationError: If entry already exists with different value
            InvalidEntryError: If entry is invalid
        """
        with self._lock:
            # Check if this would conflict with a core entry
            if entry.name in self._entries and self._entries[entry.name].is_core:
                raise ImmutableCoreError(
                    f"Cannot modify core entry '{entry.name}' in {self.name}"
                )

            self._register_entry(entry)

    def register_entries(self, entries: List[EnumEntry]) -> None:
        """
        Register multiple entries atomically.

        If any entry fails validation, none are registered.
        """
        with self._lock:
            # Validate all entries first
            for entry in entries:
                if entry.name in self._entries and self._entries[entry.name].is_core:
                    raise ImmutableCoreError(
                        f"Cannot modify core entry '{entry.name}' in {self.name}"
                    )
                if entry.name in self._entries:
                    existing = self._entries[entry.name]
                    if existing.value != entry.value:
                        raise DuplicateRegistrationError(
                            f"Entry '{entry.name}' already exists with different value: "
                            f"'{existing.value}' vs '{entry.value}'"
                        )

            # Register all entries
            for entry in entries:
                self._register_entry(entry)

    def get_entry(self, name: str) -> Optional[EnumEntry]:
        """Get an entry by name."""
        with self._lock:
            return self._entries.get(name)

    def get_entries(self) -> List[EnumEntry]:
        """Get all entries sorted by name."""
        with self._lock:
            return sorted(self._entries.values(), key=lambda e: e.name)

    def get_core_entries(self) -> List[EnumEntry]:
        """Get only core (immutable) entries."""
        with self._lock:
            return [entry for entry in self._entries.values() if entry.is_core]

    def get_registered_entries(self) -> List[EnumEntry]:
        """Get only dynamically registered entries."""
        with self._lock:
            return [entry for entry in self._entries.values() if not entry.is_core]

    def is_core_entry(self, name: str) -> bool:
        """Check if an entry is a core (immutable) entry."""
        with self._lock:
            entry = self._entries.get(name)
            return entry.is_core if entry else False

    def contains(self, name: str) -> bool:
        """Check if registry contains an entry with the given name."""
        with self._lock:
            return name in self._entries

    def get_count(self) -> int:
        """Get total number of entries."""
        with self._lock:
            return len(self._entries)

    def get_immutable_count(self) -> int:
        """Get number of immutable core entries."""
        return self._immutable_count


# ============================================================================
# FractalSemantics Enums
# ============================================================================

# Core entries for Realm enum (immutable)
REALM_CORE_ENTRIES = [
    EnumEntry("COMPANION", "companion", "Pets, familiars, companions", is_core=True),
    EnumEntry("BADGE", "badge", "Achievement badges", is_core=True),
    EnumEntry("SPONSOR_RING", "sponsor_ring", "Sponsor tier badges", is_core=True),
    EnumEntry("ACHIEVEMENT", "achievement", "Generic achievements", is_core=True),
    EnumEntry("PATTERN", "pattern", "System patterns", is_core=True),
    EnumEntry("FACULTY", "faculty", "Faculty-exclusive entities", is_core=True),
    EnumEntry("TEMPORAL", "temporal", "Time-based entities", is_core=True),
    EnumEntry("VOID", "void", "Null/empty realm", is_core=True),
]

# Core entries for Horizon enum (immutable)
HORIZON_CORE_ENTRIES = [
    EnumEntry("GENESIS", "genesis", "Entity created, initial state", is_core=True),
    EnumEntry("EMERGENCE", "emergence", "Entity becoming active", is_core=True),
    EnumEntry("PEAK", "peak", "Entity at maximum activity", is_core=True),
    EnumEntry("DECAY", "decay", "Entity waning", is_core=True),
    EnumEntry("CRYSTALLIZATION", "crystallization", "Entity settled/permanent", is_core=True),
    EnumEntry("ARCHIVED", "archived", "Historical record", is_core=True),
]

# Core entries for Polarity enum (immutable)
POLARITY_CORE_ENTRIES = [
    # Companion polarities (elemental)
    EnumEntry("LOGIC", "logic", "Logic-based resonance", is_core=True),
    EnumEntry("CREATIVITY", "creativity", "Creative resonance", is_core=True),
    EnumEntry("ORDER", "order", "Order-based resonance", is_core=True),
    EnumEntry("CHAOS", "chaos", "Chaos-based resonance", is_core=True),
    EnumEntry("BALANCE", "balance", "Balanced resonance", is_core=True),

    # Badge polarities (category)
    EnumEntry("ACHIEVEMENT", "achievement", "Achievement-based", is_core=True),
    EnumEntry("CONTRIBUTION", "contribution", "Contribution-based", is_core=True),
    EnumEntry("COMMUNITY", "community", "Community-based", is_core=True),
    EnumEntry("TECHNICAL", "technical", "Technical achievement", is_core=True),
    EnumEntry("CREATIVE", "creative", "Creative achievement", is_core=True),
    EnumEntry("UNITY", "unity", "Unity-focused", is_core=True),

    # Neutral
    EnumEntry("VOID", "void", "Neutral/void polarity", is_core=True),
]

# Core entries for Alignment enum (immutable)
ALIGNMENT_CORE_ENTRIES = [
    # Classical alignment system
    EnumEntry("LAWFUL_GOOD", "lawful_good", "Principled, helpful", is_core=True),
    EnumEntry("NEUTRAL_GOOD", "neutral_good", "Helpful, flexible", is_core=True),
    EnumEntry("CHAOTIC_GOOD", "chaotic_good", "Helpful, unconstrained", is_core=True),
    EnumEntry("LAWFUL_NEUTRAL", "lawful_neutral", "Principled, pragmatic", is_core=True),
    EnumEntry("TRUE_NEUTRAL", "true_neutral", "Balanced, pragmatic", is_core=True),
    EnumEntry("CHAOTIC_NEUTRAL", "chaotic_neutral", "Flexible, pragmatic", is_core=True),
    EnumEntry("LAWFUL_EVIL", "lawful_evil", "Principled, harmful", is_core=True),
    EnumEntry("NEUTRAL_EVIL", "neutral_evil", "Self-serving", is_core=True),
    EnumEntry("CHAOTIC_EVIL", "chaotic_evil", "Harmful, unconstrained", is_core=True),

    # Special classifications
    EnumEntry("HARMONIC", "harmonic", "Naturally coordinated", is_core=True),
    EnumEntry("ENTROPIC", "entropic", "Naturally disruptive", is_core=True),
    EnumEntry("SYMBIOTIC", "symbiotic", "Mutually beneficial connections", is_core=True),
]

# Global registries
REALM_REGISTRY = DynamicEnumRegistry("Realm", REALM_CORE_ENTRIES)
HORIZON_REGISTRY = DynamicEnumRegistry("Horizon", HORIZON_CORE_ENTRIES)
POLARITY_REGISTRY = DynamicEnumRegistry("Polarity", POLARITY_CORE_ENTRIES)
ALIGNMENT_REGISTRY = DynamicEnumRegistry("Alignment", ALIGNMENT_CORE_ENTRIES)


# Create the enums using a simpler approach that mypy can understand
class Realm(Enum):
    """Realm enum for FractalSemantics entities."""
    COMPANION = "companion"
    BADGE = "badge"
    SPONSOR_RING = "sponsor_ring"
    ACHIEVEMENT = "achievement"
    PATTERN = "pattern"
    FACULTY = "faculty"
    TEMPORAL = "temporal"
    VOID = "void"


class Horizon(Enum):
    """Horizon enum for FractalSemantics lifecycle stages."""
    GENESIS = "genesis"
    EMERGENCE = "emergence"
    PEAK = "peak"
    DECAY = "decay"
    CRYSTALLIZATION = "crystallization"
    ARCHIVED = "archived"


class Polarity(Enum):
    """Polarity enum for FractalSemantics resonance types."""
    LOGIC = "logic"
    CREATIVITY = "creativity"
    ORDER = "order"
    CHAOS = "chaos"
    BALANCE = "balance"
    ACHIEVEMENT = "achievement"
    CONTRIBUTION = "contribution"
    COMMUNITY = "community"
    TECHNICAL = "technical"
    CREATIVE = "creative"
    UNITY = "unity"
    VOID = "void"


class Alignment(Enum):
    """Alignment enum for FractalSemantics social dynamics."""
    LAWFUL_GOOD = "lawful_good"
    NEUTRAL_GOOD = "neutral_good"
    CHAOTIC_GOOD = "chaotic_good"
    LAWFUL_NEUTRAL = "lawful_neutral"
    TRUE_NEUTRAL = "true_neutral"
    CHAOTIC_NEUTRAL = "chaotic_neutral"
    LAWFUL_EVIL = "lawful_evil"
    NEUTRAL_EVIL = "neutral_evil"
    CHAOTIC_EVIL = "chaotic_evil"
    HARMONIC = "harmonic"
    ENTROPIC = "entropic"
    SYMBIOTIC = "symbiotic"


# ============================================================================
# Utility Functions
# ============================================================================

def register_realm_entry(entry: EnumEntry) -> None:
    """Register a new realm entry."""
    REALM_REGISTRY.register_entry(entry)
    # Note: Static enums cannot be dynamically extended at runtime
    # This function is kept for API compatibility


def register_horizon_entry(entry: EnumEntry) -> None:
    """Register a new horizon entry."""
    HORIZON_REGISTRY.register_entry(entry)
    # Note: Static enums cannot be dynamically extended at runtime
    # This function is kept for API compatibility


def register_polarity_entry(entry: EnumEntry) -> None:
    """Register a new polarity entry."""
    POLARITY_REGISTRY.register_entry(entry)
    # Note: Static enums cannot be dynamically extended at runtime
    # This function is kept for API compatibility


def register_alignment_entry(entry: EnumEntry) -> None:
    """Register a new alignment entry."""
    ALIGNMENT_REGISTRY.register_entry(entry)
    # Note: Static enums cannot be dynamically extended at runtime
    # This function is kept for API compatibility


def get_realm_entry(name: str) -> Optional[EnumEntry]:
    """Get realm entry metadata."""
    return REALM_REGISTRY.get_entry(name)


def get_horizon_entry(name: str) -> Optional[EnumEntry]:
    """Get horizon entry metadata."""
    return HORIZON_REGISTRY.get_entry(name)


def get_polarity_entry(name: str) -> Optional[EnumEntry]:
    """Get polarity entry metadata."""
    return POLARITY_REGISTRY.get_entry(name)


def get_alignment_entry(name: str) -> Optional[EnumEntry]:
    """Get alignment entry metadata."""
    return ALIGNMENT_REGISTRY.get_entry(name)


def is_core_realm(name: str) -> bool:
    """Check if realm entry is core."""
    return REALM_REGISTRY.is_core_entry(name)


def is_core_horizon(name: str) -> bool:
    """Check if horizon entry is core."""
    return HORIZON_REGISTRY.is_core_entry(name)


def is_core_polarity(name: str) -> bool:
    """Check if polarity entry is core."""
    return POLARITY_REGISTRY.is_core_entry(name)


def is_core_alignment(name: str) -> bool:
    """Check if alignment entry is core."""
    return ALIGNMENT_REGISTRY.is_core_entry(name)


# ============================================================================
# Backward Compatibility
# ============================================================================

# For backward compatibility, create enum aliases that match the original structure
# These will be populated from the registries

def _create_backward_compatible_enum(name: str, registry: DynamicEnumRegistry) -> Enum:
    """Create a backward-compatible enum from registry."""
    entries = registry.get_entries()
    members = {entry.name: entry.value for entry in entries}
    return Enum(name, members)


# Create backward-compatible enums
RealmCompat = _create_backward_compatible_enum("Realm", REALM_REGISTRY)
HorizonCompat = _create_backward_compatible_enum("Horizon", HORIZON_REGISTRY)
PolarityCompat = _create_backward_compatible_enum("Polarity", POLARITY_REGISTRY)
AlignmentCompat = _create_backward_compatible_enum("Alignment", ALIGNMENT_REGISTRY)


# The enum classes above can be used directly as types
