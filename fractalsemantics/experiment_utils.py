"""
Shared utilities for FractalSemantics validation experiments.

This module contains utility functions used by multiple experiments,
such as random bit-chain generation and other shared testing functionality.
"""

import secrets
from typing import List
from datetime import datetime, timezone



# Use cryptographically secure random number generator
secure_random = secrets.SystemRandom()





def compute_shannon_entropy(coordinates: List[str]) -> float:
    """
    Compute Shannon entropy of a list of coordinate representations.

    Shannon entropy H(X) = -Σ p(x) * log2(p(x))
    where p(x) is the probability of observing coordinate value x.

    Higher entropy indicates more information content and better
    discrimination between different entities.

    Args:
        coordinates: List of coordinate string representations

    Returns:
        Shannon entropy in bits
    """
    from collections import Counter
    import numpy as np

    if not coordinates:
        return 0.0

    # Count frequency of each unique coordinate
    counts = Counter(coordinates)
    total = len(coordinates)

    # Calculate Shannon entropy
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        if probability > 0:
            entropy -= probability * np.log2(probability)

    return entropy


def compute_adjacency_score(tags1: List[str], tags2: List[str]) -> float:
    """
    Compute adjacency score between two sets of tags.

    Returns the fraction of overlapping tags between two lists,
    normalized to [0, 1].

    Used to measure semantic closeness between entities based on
    shared attributes/tags.

    Args:
        tags1: First list of tags
        tags2: Second list of tags

    Returns:
        Adjacency score in [0, 1] range
    """
    if not tags1 or not tags2:
        return 0.0

    set1 = set(tags1)
    set2 = set(tags2)
    intersection = set1 & set2

    # Use Jaccard similarity (intersection / union)
    union = set1 | set2
    if not union:
        return 0.0

    return len(intersection) / len(union)


def normalize_float(value, decimal_places=8):
    """
    Normalize a float to a consistent string representation.

    Args:
        value: Float value to normalize
        decimal_places: Number of decimal places to include

    Returns:
        Normalized string representation of the float
    """
    if isinstance(value, str):
        try:
            value = float(value)
        except (ValueError, TypeError):
            return str(value)

    if not isinstance(value, (int, float)):
        return str(value)

    # Handle special float values
    if value is None or (isinstance(value, float) and (
        value != value or  # nan
        value == float('inf') or
        value == -float('inf')
    )):
        return str(value)

    # Round to specified decimal places and remove trailing zeros
    formatted = f"{value:.{decimal_places}f}".rstrip('0').rstrip('.')
    return formatted


def normalize_timestamp(dt=None):
    """
    Normalize a timestamp to ISO8601 format.

    Args:
        dt: datetime object, or None for current time

    Returns:
        ISO8601 formatted timestamp string
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    elif isinstance(dt, str):
        # If it's already a string, return as-is (assume it's already normalized)
        return dt
    elif not hasattr(dt, 'strftime'):
        # Not a datetime object, convert to string
        return str(dt)

    # Convert to UTC if timezone-aware, otherwise assume UTC
    if hasattr(dt, 'utctimetuple'):
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc)

    return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'


def sort_json_keys(obj):
    """
    Recursively sort dictionary keys for deterministic JSON serialization.

    Args:
        obj: Dictionary, list, or primitive to sort

    Returns:
        Object with sorted dictionary keys
    """
    if isinstance(obj, dict):
        return {key: sort_json_keys(obj[key]) for key in sorted(obj.keys())}
    elif isinstance(obj, list):
        return [sort_json_keys(item) for item in obj]
    else:
        return obj


# Enums for backward compatibility
class DataClass(str):
    """Enum replacement for data classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    SENSITIVE = "sensitive"
    CONFIDENTIAL = "confidential"
    PII = "pii"
    RESTRICTED = "restricted"


class Capability(str):
    """Enum replacement for capability levels."""
    COMPRESSED = "compressed"
    PARTIAL = "partial"
    FULL = "full"
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
