"""
EXP-07: LUCA Bootstrap Test - Entities Module

This module defines the core data structures and entities used in the LUCA
bootstrap test. These entities capture the minimal representation needed for
system reconstruction and the comprehensive testing framework for validating
the bootstrap process.

The entities are designed to capture:
- Test bit-chain entities with full lineage and coordinate information
- LUCA state representations for compression and expansion
- Bootstrap result tracking and validation metrics
- Fractal property testing and continuity validation

Core Concept:
The LUCA (Last Universal Common Ancestor) represents the minimal, irreducible
state from which the entire FractalStat system can be reconstructed. This
module provides the entities needed to test this fundamental property.

Author: FractalSemantics
Date: 2025-12-07
"""

import json
import uuid
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone


@dataclass
class TestBitChain:
    """
    Minimal test bit-chain for LUCA bootstrap testing.
    
    This dataclass represents a simplified bit-chain entity used specifically
    for testing the LUCA bootstrap process. It contains all the essential
    information needed to validate compression and reconstruction.
    
    Key Properties:
        - Self-contained with all necessary coordinate information
        - Lineage tracking from LUCA (lineage 0)
        - FractalStat-like addressing for validation
        - Metadata preservation for information integrity testing
    
    Attributes:
        bit_chain_id: Unique identifier for the entity
        content: Test content data
        lineage: Distance from LUCA (0 = LUCA, 1+ = descendants)
        realm: Entity realm classification
        horizon: Entity horizon stage
        polarity: Entity polarity type
        dimensionality: Fractal depth/complexity level
        timestamp: Creation timestamp
        metadata: Additional entity metadata
    """
    
    bit_chain_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    lineage: int = 0  # Distance from LUCA
    realm: str = "pattern"
    horizon: str = "genesis"
    polarity: str = "logic"
    dimensionality: int = 1

    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Returns:
            Dictionary with all entity attributes for serialization
        """
        return {
            "bit_chain_id": self.bit_chain_id,
            "content": self.content,
            "lineage": self.lineage,
            "realm": self.realm,
            "horizon": self.horizon,
            "polarity": self.polarity,
            "dimensionality": self.dimensionality,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """
        Convert to JSON string representation.
        
        Returns:
            JSON string with all entity data
        """
        return json.dumps(self.to_dict())

    def get_fractalstat_address(self) -> str:
        """
        Generate FractalStat-like address for validation.
        
        Returns:
            String representation of FractalStat address format
            
        Address Format:
            FractalStat-{realm[0]}-{lineage:03d}-50-{horizon[0]}-50-{polarity[0]}-{dimensionality}
            
        Example:
            "FractalStat-P-001-50-E-50-L-1"
        """
        return (
            f"FractalStat-{self.realm[0].upper()}-{self.lineage:03d}-"
            f"50-{self.horizon[0].upper()}-50-{self.polarity[0].upper()}-{
                self.dimensionality
            }"
        )

    def get_content_hash(self) -> str:
        """
        Get SHA256 hash of entity content for integrity validation.
        
        Returns:
            Hexadecimal hash string of entity content
        """
        return hashlib.sha256(self.content.encode()).hexdigest()

    def get_entity_signature(self) -> str:
        """
        Get signature hash of all critical entity attributes.
        
        Returns:
            Hexadecimal hash of critical entity data for comparison
        """
        critical_data = {
            "bit_chain_id": self.bit_chain_id,
            "lineage": self.lineage,
            "realm": self.realm,
            "horizon": self.horizon,
            "polarity": self.polarity,
            "dimensionality": self.dimensionality,
        }
        return hashlib.sha256(json.dumps(critical_data, sort_keys=True).encode()).hexdigest()

    def is_luca(self) -> bool:
        """
        Check if this entity represents LUCA (lineage 0).
        
        Returns:
            Boolean indicating if entity is LUCA
        """
        return self.lineage == 0

    def get_lineage_depth(self) -> int:
        """
        Get lineage depth from LUCA.
        
        Returns:
            Integer representing distance from LUCA
        """
        return self.lineage

    def __str__(self) -> str:
        """
        String representation of test bit-chain.
        
        Returns:
            Human-readable summary of entity characteristics
        """
        return (
            f"TestBitChain(id={self.bit_chain_id[:12]}..., "
            f"lineage={self.lineage}, realm={self.realm}, "
            f"address={self.get_fractalstat_address()})"
        )


@dataclass
class LUCAState:
    """
    LUCA (Last Universal Common Ancestor) state representation.
    
    This dataclass captures the minimal, compressed state from which the entire
    system can be reconstructed. The LUCA state contains only essential
    information needed for bootstrap while preserving all critical relationships.
    
    Compression Strategy:
        - Store only essential addressing information
        - Preserve lineage relationships
        - Maintain entity signatures for integrity
        - Minimize metadata while preserving traceability
    
    Attributes:
        luca_version: Version of LUCA encoding format
        entity_count: Number of entities in compressed state
        encodings: List of compressed entity representations
        total_original_size: Total size before compression
        total_compressed_size: Total size after compression
        compression_ratio: Compression efficiency ratio
        luca_timestamp: Timestamp of LUCA state creation
        luca_hash: Hash of entire LUCA state for integrity
    """
    
    luca_version: str = "1.0"
    entity_count: int = 0
    encodings: List[Dict[str, Any]] = field(default_factory=list)
    total_original_size: int = 0
    total_compressed_size: int = 0
    compression_ratio: float = 1.0
    luca_timestamp: str = ""
    luca_hash: str = ""

    def __post_init__(self):
        """Initialize timestamp if not provided."""
        if not self.luca_timestamp:
            self.luca_timestamp = datetime.now(timezone.utc).isoformat()

    def get_compression_efficiency(self) -> Dict[str, Any]:
        """
        Get detailed compression efficiency metrics.
        
        Returns:
            Dictionary with compression statistics and efficiency analysis
        """
        return {
            "compression_ratio": self.compression_ratio,
            "space_saved_bytes": self.total_original_size - self.total_compressed_size,
            "space_saved_percentage": (
                (self.total_original_size - self.total_compressed_size) / self.total_original_size * 100
                if self.total_original_size > 0 else 0.0
            ),
            "bytes_per_entity_original": self.total_original_size / self.entity_count if self.entity_count > 0 else 0,
            "bytes_per_entity_compressed": self.total_compressed_size / self.entity_count if self.entity_count > 0 else 0,
            "compression_factor": self.total_original_size / self.total_compressed_size if self.total_compressed_size > 0 else 1.0
        }

    def validate_integrity(self) -> bool:
        """
        Validate LUCA state integrity using stored hash.
        
        Returns:
            Boolean indicating if LUCA state is intact
        """
        current_hash = hashlib.sha256(
            json.dumps(self.encodings, sort_keys=True).encode()
        ).hexdigest()
        return current_hash == self.luca_hash

    def get_entity_signatures(self) -> List[str]:
        """
        Get all entity signatures from LUCA state.
        
        Returns:
            List of entity signature hashes for validation
        """
        return [encoding.get("hash", "") for encoding in self.encodings]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Returns:
            Dictionary with all LUCA state attributes for serialization
        """
        return {
            "luca_version": self.luca_version,
            "entity_count": self.entity_count,
            "encodings": self.encodings,
            "total_original_size": self.total_original_size,
            "total_compressed_size": self.total_compressed_size,
            "compression_ratio": self.compression_ratio,
            "luca_timestamp": self.luca_timestamp,
            "luca_hash": self.luca_hash,
        }

    def __str__(self) -> str:
        """
        String representation of LUCA state.
        
        Returns:
            Human-readable summary of LUCA state characteristics
        """
        return (
            f"LUCAState(version={self.luca_version}, entities={self.entity_count}, "
            f"compression={self.compression_ratio:.2f}x, "
            f"integrity={'OK' if self.validate_integrity() else 'CORRUPTED'})"
        )


@dataclass
class LUCAEncoding:
    """
    Individual entity encoding in LUCA state.
    
    This dataclass represents the compressed form of a single entity within
    the LUCA state. It contains minimal but sufficient information for
    complete reconstruction.
    
    Encoding Components:
        - Entity identifier and hash for integrity
        - Lineage information for hierarchy
        - Signature characters for realm/horizon/polarity
        - Size and metadata information for reconstruction
    
    Attributes:
        id: Entity identifier
        hash: SHA256 hash of original entity
        lineage: Lineage distance from LUCA
        realm_sig: Single character realm signature
        horizon_sig: Single character horizon signature
        polarity_sig: Single character polarity signature
        dimensionality: Fractal depth level
        content_size: Original content size
        metadata_keys: Keys of preserved metadata
    """
    
    id: str
    hash: str
    lineage: int
    realm_sig: str
    horizon_sig: str
    polarity_sig: str
    dimensionality: int
    content_size: int
    metadata_keys: List[str]

    def expand_to_entity(self) -> TestBitChain:
        """
        Expand LUCA encoding back to full TestBitChain entity.
        
        Returns:
            Reconstructed TestBitChain entity from LUCA encoding
        """
        return TestBitChain(
            bit_chain_id=self.id,
            content=f"[BOOTSTRAPPED] {self.content_size} bytes",
            lineage=self.lineage,
            realm=self._expand_signature(self.realm_sig),
            horizon=self._expand_signature(self.horizon_sig),
            polarity=self._expand_signature(self.polarity_sig),
            dimensionality=self.dimensionality,
            metadata={key: None for key in self.metadata_keys},
        )

    def _expand_signature(self, sig: str) -> str:
        """
        Expand single-character signature to full value.
        
        Args:
            sig: Single character signature
            
        Returns:
            Full string value corresponding to signature
        """
        signature_map = {
            "p": "pattern",
            "d": "data", 
            "n": "narrative",
            "e": "emergence",
            "k": "peak",
            "c": "crystallization",
            "l": "logic",
            "r": "creativity",
        }
        return signature_map.get(sig, "unknown")

    def get_reconstruction_integrity(self, original_entity: TestBitChain) -> Dict[str, bool]:
        """
        Check reconstruction integrity against original entity.
        
        Args:
            original_entity: Original entity to compare against
            
        Returns:
            Dictionary with integrity check results for each attribute
        """
        reconstructed = self.expand_to_entity()
        
        return {
            "id_match": reconstructed.bit_chain_id == original_entity.bit_chain_id,
            "lineage_match": reconstructed.lineage == original_entity.lineage,
            "realm_match": reconstructed.realm == original_entity.realm,
            "horizon_match": reconstructed.horizon == original_entity.horizon,
            "polarity_match": reconstructed.polarity == original_entity.polarity,
            "dimensionality_match": reconstructed.dimensionality == original_entity.dimensionality,
            "hash_match": reconstructed.get_entity_signature() == original_entity.get_entity_signature()
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Returns:
            Dictionary with all encoding attributes for serialization
        """
        return {
            "id": self.id,
            "hash": self.hash,
            "lineage": self.lineage,
            "realm_sig": self.realm_sig,
            "horizon_sig": self.horizon_sig,
            "polarity_sig": self.polarity_sig,
            "dimensionality": self.dimensionality,
            "content_size": self.content_size,
            "metadata_keys": self.metadata_keys,
        }

    def __str__(self) -> str:
        """
        String representation of LUCA encoding.
        
        Returns:
            Human-readable summary of encoding characteristics
        """
        return (
            f"LUCAEncoding(id={self.id[:12]}..., lineage={self.lineage}, "
            f"realm={self.realm_sig}, horizon={self.horizon_sig}, "
            f"polarity={self.polarity_sig}, dimensionality={self.dimensionality})"
        )


@dataclass
class BootstrapValidation:
    """
    Results of bootstrap validation testing.
    
    This dataclass captures the comprehensive validation results from testing
    the LUCA bootstrap process, including entity recovery rates, information
    preservation, and system integrity metrics.
    
    Validation Categories:
        - Entity recovery rates (perfect reconstruction required)
        - Information preservation (no data loss)
        - Lineage continuity (hierarchy maintained)
        - Fractal property preservation (self-similarity)
        - Multiple cycle stability (no degradation)
    
    Attributes:
        entity_recovery_rate: Percentage of entities successfully recovered
        lineage_recovery_rate: Percentage of lineages correctly preserved
        realm_recovery_rate: Percentage of realms correctly preserved
        dimensionality_recovery_rate: Percentage of dimensionalities preserved
        information_loss_detected: Whether any information was lost
        bootstrap_cycles_completed: Number of successful bootstrap cycles
        bootstrap_failures: Number of bootstrap failures
        lineage_continuity: Whether lineage hierarchy was maintained
        fractal_properties_preserved: Whether fractal characteristics were maintained
        reconstruction_errors: List of any reconstruction errors encountered
    """
    
    entity_recovery_rate: float = 0.0
    lineage_recovery_rate: float = 0.0
    realm_recovery_rate: float = 0.0
    dimensionality_recovery_rate: float = 0.0
    information_loss_detected: bool = False
    bootstrap_cycles_completed: int = 0
    bootstrap_failures: int = 0
    lineage_continuity: bool = False
    fractal_properties_preserved: bool = False
    reconstruction_errors: List[str] = field(default_factory=list)

    def get_overall_success(self) -> bool:
        """
        Determine overall bootstrap validation success.
        
        Returns:
            Boolean indicating if all validation criteria were met
        """
        perfect_recovery = (
            self.entity_recovery_rate >= 1.0
            and self.lineage_recovery_rate >= 1.0
            and self.realm_recovery_rate >= 1.0
            and self.dimensionality_recovery_rate >= 1.0
        )
        
        no_degradation = self.bootstrap_failures == 0 and self.lineage_continuity
        fractal_integrity = self.fractal_properties_preserved
        
        return perfect_recovery and no_degradation and fractal_integrity

    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive validation summary.
        
        Returns:
            Dictionary with detailed validation results and success indicators
        """
        return {
            "overall_success": self.get_overall_success(),
            "recovery_rates": {
                "entity": round(self.entity_recovery_rate, 4),
                "lineage": round(self.lineage_recovery_rate, 4),
                "realm": round(self.realm_recovery_rate, 4),
                "dimensionality": round(self.dimensionality_recovery_rate, 4),
            },
            "integrity_checks": {
                "information_loss": self.information_loss_detected,
                "lineage_continuity": self.lineage_continuity,
                "fractal_properties": self.fractal_properties_preserved,
            },
            "bootstrap_stability": {
                "cycles_completed": self.bootstrap_cycles_completed,
                "failures": self.bootstrap_failures,
                "success_rate": (
                    (self.bootstrap_cycles_completed - self.bootstrap_failures) / self.bootstrap_cycles_completed
                    if self.bootstrap_cycles_completed > 0 else 1.0
                )
            },
            "reconstruction_errors": self.reconstruction_errors,
            "critical_failures": len(self.reconstruction_errors) > 0
        }

    def add_reconstruction_error(self, error: str):
        """
        Add a reconstruction error to the validation record.
        
        Args:
            error: Description of the reconstruction error
        """
        self.reconstruction_errors.append(error)
        self.information_loss_detected = True

    def __str__(self) -> str:
        """
        String representation of bootstrap validation.
        
        Returns:
            Human-readable summary of validation results
        """
        return (
            f"BootstrapValidation(success={self.get_overall_success()}, "
            f"entity_recovery={self.entity_recovery_rate:.1%}, "
            f"cycles={self.bootstrap_cycles_completed}, "
            f"errors={len(self.reconstruction_errors)})"
        )


@dataclass
class FractalPropertyTest:
    """
    Results of fractal property testing.
    
    This dataclass captures the validation results for fractal properties
    of the system, including self-similarity, scale invariance, recursive
    structure, and LUCA traceability.
    
    Fractal Properties Tested:
        - Self-similarity: Entities have consistent structure across scales
        - Scale invariance: System behavior consistent at different levels
        - Recursive structure: Hierarchical organization maintained
        - LUCA traceability: All entities traceable to LUCA
        - Information entropy: Proper information distribution
    
    Attributes:
        self_similarity: Whether entities show self-similar structure
        scale_invariance: Whether system is scale-invariant
        recursive_structure: Whether recursive patterns are preserved
        luca_traceability: Whether all entities trace to LUCA
        information_entropy: Whether information entropy is appropriate
        structural_consistency: Whether entity structures are consistent
        lineage_depth: Number of distinct lineage levels
        entropy_ratio: Ratio of structural to content information
    """
    
    self_similarity: bool = False
    scale_invariance: bool = False
    recursive_structure: bool = False
    luca_traceability: bool = False
    information_entropy: bool = False
    structural_consistency: bool = False
    lineage_depth: int = 0
    entropy_ratio: float = 0.0

    def get_fractal_score(self) -> float:
        """
        Calculate overall fractal property score.
        
        Returns:
            Float between 0.0 and 1.0 representing fractal property completeness
        """
        properties = [
            self.self_similarity,
            self.scale_invariance,
            self.recursive_structure,
            self.luca_traceability,
            self.information_entropy,
            self.structural_consistency
        ]
        return sum(properties) / len(properties) if properties else 0.0

    def is_fractal_system(self) -> bool:
        """
        Determine if system demonstrates fractal properties.
        
        Returns:
            Boolean indicating if system meets fractal criteria
        """
        # Require most properties to be true for fractal classification
        fractal_score = self.get_fractal_score()
        return fractal_score >= 0.8  # 80% of properties must be true

    def get_property_details(self) -> Dict[str, Any]:
        """
        Get detailed fractal property analysis.
        
        Returns:
            Dictionary with property values and analysis details
        """
        return {
            "self_similarity": self.self_similarity,
            "scale_invariance": self.scale_invariance,
            "recursive_structure": self.recursive_structure,
            "luca_traceability": self.luca_traceability,
            "information_entropy": self.information_entropy,
            "structural_consistency": self.structural_consistency,
            "lineage_depth": self.lineage_depth,
            "entropy_ratio": self.entropy_ratio,
            "fractal_score": self.get_fractal_score(),
            "is_fractal": self.is_fractal_system()
        }

    def __str__(self) -> str:
        """
        String representation of fractal property test.
        
        Returns:
            Human-readable summary of fractal property results
        """
        return (
            f"FractalPropertyTest(fractal_score={self.get_fractal_score():.2f}, "
            f"lineage_depth={self.lineage_depth}, "
            f"entropy_ratio={self.entropy_ratio:.2f}, "
            f"is_fractal={self.is_fractal_system()})"
        )


@dataclass
class LUCABootstrapResult:
    """
    Complete results for LUCA bootstrap test.
    
    This dataclass captures the comprehensive results of the LUCA bootstrap
    experiment, including all validation metrics, compression statistics,
    and system integrity assessments.
    
    Result Categories:
        - Test metadata and execution information
        - Compression and expansion statistics
        - Entity recovery and information preservation
        - Fractal property validation
        - Bootstrap stability and continuity testing
        - Overall success determination
    
    Attributes:
        experiment: Experiment identifier
        title: Test title
        timestamp: Test execution timestamp
        status: Overall test status (PASS/FAIL)
        results: Detailed results dictionary
    """
    
    experiment: str = "EXP-07"
    title: str = "LUCA Bootstrap Test"
    timestamp: str = ""
    status: str = "PASS"
    results: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize timestamp if not provided."""
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation of complete test results
        """
        return {
            "experiment": self.experiment,
            "title": self.title,
            "timestamp": self.timestamp,
            "status": self.status,
            "results": self.results,
        }

    def get_summary(self) -> Dict[str, Any]:
        """
        Get high-level summary of test results.
        
        Returns:
            Dictionary with key metrics and overall assessment
        """
        return {
            "experiment": self.experiment,
            "status": self.status,
            "timestamp": self.timestamp,
            "compression_ratio": self.results.get("compression", {}).get("ratio", 0.0),
            "entity_recovery_rate": self.results.get("comparison", {}).get("entity_recovery_rate", 0.0),
            "bootstrap_success": self.results.get("comparison", {}).get("entity_recovery_rate", 0.0) >= 1.0,
            "fractal_properties": self.results.get("fractal", {}).get("fractal_score", 0.0),
            "continuity_preserved": self.results.get("continuity", {}).get("lineage_preserved", False),
            "overall_success": self.status == "PASS"
        }

    def is_successful(self) -> bool:
        """
        Determine if test was successful.
        
        Returns:
            Boolean indicating if test passed all criteria
        """
        return self.status == "PASS"

    def __str__(self) -> str:
        """
        String representation of LUCA bootstrap result.
        
        Returns:
            Human-readable summary of test execution and results
        """
        summary = self.get_summary()
        return (
            f"LUCABootstrapResult({self.experiment}: {self.status}, "
            f"compression={summary['compression_ratio']:.2f}x, "
            f"recovery={summary['entity_recovery_rate']:.1%}, "
            f"fractal_score={summary['fractal_properties']:.2f})"
        )