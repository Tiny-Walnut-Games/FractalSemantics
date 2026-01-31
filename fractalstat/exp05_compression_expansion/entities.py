"""
EXP-05: Bit-Chain Compression/Expansion Losslessness Validation - Entities Module

This module defines the core data structures and entities used in the compression
and expansion validation experiment. These entities capture the complete pipeline
from original bit-chains through compression stages to reconstruction attempts.

The entities are designed to capture:
- Compression pipeline stages and their characteristics
- Information preservation metrics across stages
- Reconstruction accuracy and losslessness validation
- Provenance chain integrity tracking
- Compression efficiency and quality metrics

Author: FractalSemantics
Date: 2025-12-07
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone


@dataclass
class CompressionStage:
    """
    Single stage in the compression pipeline.
    
    This dataclass represents one step in the FractalStat compression pipeline,
    capturing the characteristics and information preservation at each stage.
    
    Pipeline Stages:
        1. Original: Full FractalStat coordinates and metadata
        2. Fragments: Serialized bit-chain representation with embeddings
        3. Clusters: Grouped fragments with provenance tracking
        4. Glyphs: Molten form with affect and compressed summary
        5. Mist: Final evaporation with proto-thought representation
    
    Attributes:
        stage_name: Name of the compression stage
        size_bytes: Size of data at this stage in bytes
        record_count: Number of records/entities at this stage
        key_metadata: Critical metadata that survives compression
        luminosity: Activity level/heat preserved through compression
        provenance_intact: Whether provenance chain remains complete
    """
    
    stage_name: str  # "original", "fragments", "cluster", "glyph", "mist"
    size_bytes: int
    record_count: int
    key_metadata: Dict[str, Any]  # What survives at this stage
    luminosity: float  # Activity level / heat
    provenance_intact: bool

    def compression_ratio_from_original(self, original_bytes: int) -> float:
        """
        Calculate compression ratio relative to original size.
        
        Args:
            original_bytes: Size of original data in bytes
            
        Returns:
            Compression ratio (original_size / current_size)
            
        Example:
            >>> stage = CompressionStage("glyph", 1000, 1, {}, 0.5, True)
            >>> ratio = stage.compression_ratio_from_original(10000)
            >>> ratio  # 10:1 compression
            10.0
        """
        return original_bytes / max(self.size_bytes, 1)

    def get_stage_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics for this compression stage.
        
        Returns:
            Dictionary with stage-specific metrics and characteristics
            
        Metrics Include:
            - Compression efficiency
            - Information preservation
            - Luminosity retention
            - Provenance integrity
        """
        return {
            'stage_name': self.stage_name,
            'size_bytes': self.size_bytes,
            'record_count': self.record_count,
            'luminosity': self.luminosity,
            'provenance_intact': self.provenance_intact,
            'metadata_fields': len(self.key_metadata),
            'metadata_size': sum(len(str(v)) for v in self.key_metadata.values()) if self.key_metadata else 0
        }

    def __str__(self) -> str:
        """
        String representation of compression stage.
        
        Returns:
            Human-readable summary of stage characteristics
        """
        return (
            f"{self.stage_name}: {self.size_bytes} bytes, "
            f"{self.record_count} records, luminosity={self.luminosity:.3f}, "
            f"provenance={'intact' if self.provenance_intact else 'broken'}"
        )


@dataclass
class BitChainCompressionPath:
    """
    Complete compression path for a single bit-chain.
    
    This dataclass captures the entire journey of a bit-chain through the
    compression pipeline, from original coordinates to mist form and back
    to reconstruction attempts.
    
    Attributes:
        original_bitchain: The original bit-chain before compression
        original_address: FractalStat address of original bit-chain
        original_fractalstat_dict: Original coordinate dictionary
        original_serialized_size: Size of serialized original data
        original_luminosity: Original luminosity value
        
        stages: List of compression stages traversed
        reconstructed_address: Attempted reconstruction address
        coordinate_match_accuracy: Accuracy of coordinate reconstruction (0.0-1.0)
        can_expand_completely: Whether full expansion is possible
        
        final_compression_ratio: Overall compression achieved
        luminosity_final: Final luminosity after compression
        narrative_preserved: Whether narrative meaning survived
        provenance_chain_complete: Whether provenance chain remained intact
    """
    
    original_bitchain: Any  # BitChain object
    original_address: str
    original_fractalstat_dict: Dict[str, Any]
    original_serialized_size: int
    original_luminosity: float

    # Stages
    stages: List[CompressionStage] = field(default_factory=list)

    # Reconstruction attempt
    reconstructed_address: Optional[str] = None
    coordinate_match_accuracy: float = 0.0  # 0.0 to 1.0
    can_expand_completely: bool = False

    # Metrics
    final_compression_ratio: float = 0.0
    luminosity_final: float = 0.0
    narrative_preserved: bool = False
    provenance_chain_complete: bool = False

    def get_compression_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive analysis of the compression path.
        
        Returns:
            Dictionary with compression efficiency and quality metrics
            
        Analysis Includes:
            - Compression ratio progression through stages
            - Luminosity decay pattern
            - Information preservation metrics
            - Reconstruction quality assessment
        """
        if not self.stages:
            return {
                'compression_path_complete': False,
                'stages_count': 0,
                'compression_progression': [],
                'luminosity_decay': 0.0,
                'information_preservation': 0.0
            }

        # Calculate compression progression
        compression_progression = []
        original_size = self.original_serialized_size
        
        for stage in self.stages:
            ratio = stage.compression_ratio_from_original(original_size)
            compression_progression.append({
                'stage': stage.stage_name,
                'compression_ratio': ratio,
                'size_bytes': stage.size_bytes,
                'luminosity': stage.luminosity
            })

        # Calculate luminosity decay
        final_luminosity = self.stages[-1].luminosity if self.stages else 0.0
        luminosity_decay = (self.original_luminosity - final_luminosity) / max(self.original_luminosity, 0.001)

        return {
            'compression_path_complete': True,
            'stages_count': len(self.stages),
            'compression_progression': compression_progression,
            'luminosity_decay': luminosity_decay,
            'final_compression_ratio': self.final_compression_ratio,
            'coordinate_accuracy': self.coordinate_match_accuracy,
            'reconstruction_possible': self.can_expand_completely,
            'narrative_preserved': self.narrative_preserved,
            'provenance_intact': self.provenance_chain_complete
        }

    def get_reconstruction_quality(self) -> Dict[str, Any]:
        """
        Get detailed reconstruction quality metrics.
        
        Returns:
            Dictionary with reconstruction accuracy and completeness
            
        Quality Metrics:
            - Coordinate field recovery accuracy
            - Address reconstruction success
            - Information completeness assessment
            - Expansion capability validation
        """
        if not self.stages:
            return {
                'reconstruction_attempted': False,
                'coordinate_accuracy': 0.0,
                'address_reconstructed': False,
                'expansion_possible': False,
                'fields_recovered': 0,
                'total_fields': 0
            }

        # Count recovered fields
        total_fields = 8  # realm, lineage, adjacency, horizon, luminosity, polarity, dimensionality, alignment
        fields_recovered = 0

        # Check each coordinate field for recovery
        original_coords = self.original_fractalstat_dict
        if self.stages:
            final_stage = self.stages[-1]
            breadcrumbs = final_stage.key_metadata.get('recovery_breadcrumbs', {})

            # Check field recovery
            if breadcrumbs.get('original_realm') == original_coords.get('realm'):
                fields_recovered += 1
            if breadcrumbs.get('original_lineage') == original_coords.get('lineage'):
                fields_recovered += 1
            if breadcrumbs.get('original_adjacency') == original_coords.get('adjacency', []):
                fields_recovered += 1
            if breadcrumbs.get('original_horizon') == original_coords.get('horizon'):
                fields_recovered += 1
            if breadcrumbs.get('original_polarity') == original_coords.get('polarity'):
                fields_recovered += 1
            if breadcrumbs.get('original_dimensionality') == original_coords.get('dimensionality'):
                fields_recovered += 1
            if breadcrumbs.get('original_alignment') == original_coords.get('alignment'):
                fields_recovered += 1
            if breadcrumbs.get('original_luminosity') is not None:
                fields_recovered += 1

        return {
            'reconstruction_attempted': len(self.stages) > 0,
            'coordinate_accuracy': self.coordinate_match_accuracy,
            'address_reconstructed': self.reconstructed_address is not None,
            'expansion_possible': self.can_expand_completely,
            'fields_recovered': fields_recovered,
            'total_fields': total_fields,
            'field_recovery_rate': fields_recovered / total_fields if total_fields > 0 else 0.0,
            'narrative_preserved': self.narrative_preserved,
            'provenance_intact': self.provenance_chain_complete
        }

    def calculate_stats(self) -> Dict[str, Any]:
        """
        Compute summary statistics for this compression path.
        
        Returns:
            Dictionary with key statistics and metrics for the path
            
        Summary Statistics:
            - Original and final characteristics
            - Compression efficiency
            - Information preservation quality
            - Reconstruction capability
        """
        result = {
            "original_realm": self.original_fractalstat_dict.get("realm"),
            "original_address": self.original_address[:16] + "...",
            "stages_count": len(self.stages),
            "compression_ratio": self.final_compression_ratio,
            "luminosity_decay": self.original_luminosity - self.luminosity_final,
            "coordinate_accuracy": round(self.coordinate_match_accuracy, 4),
            "provenance_intact": self.provenance_chain_complete,
            "narrative_preserved": self.narrative_preserved,
            "can_expand": self.can_expand_completely,
        }

        if self.stages:
            final_stage = self.stages[-1]
            result["final_stage"] = final_stage.stage_name

        return result

    def __str__(self) -> str:
        """
        String representation of compression path.
        
        Returns:
            Human-readable summary of the compression journey
        """
        return (
            f"Compression Path: {self.original_bitchain.id[:12]}... "
            f"→ {self.final_compression_ratio:.2f}x compression, "
            f"{self.coordinate_match_accuracy:.1%} coordinate accuracy, "
            f"expandable: {'YES' if self.can_expand_completely else 'NO'}"
        )


@dataclass
class CompressionExperimentResults:
    """
    Complete results from EXP-05 compression/expansion validation.
    
    This dataclass captures the comprehensive results of testing compression
    and expansion across multiple bit-chains, including aggregate statistics
    and losslessness validation.
    
    Attributes:
        start_time: ISO timestamp when the experiment started
        end_time: ISO timestamp when the experiment completed
        total_duration_seconds: Total time for all compression tests
        
        num_bitchains_tested: Number of bit-chains processed
        compression_paths: Individual compression paths for each bit-chain
        
        avg_compression_ratio: Average compression achieved across all paths
        avg_luminosity_decay_ratio: Average luminosity decay
        avg_coordinate_accuracy: Average coordinate reconstruction accuracy
        percent_provenance_intact: Percentage of paths with intact provenance
        percent_narrative_preserved: Percentage of paths with preserved narrative
        percent_expandable: Percentage of paths that can be fully expanded
        
        is_lossless: Overall losslessness determination
        major_findings: Key findings and observations from the experiment
    """
    
    start_time: str
    end_time: str
    total_duration_seconds: float
    num_bitchains_tested: int

    # Per-bitchain paths
    compression_paths: List[BitChainCompressionPath]

    # Aggregate statistics
    avg_compression_ratio: float
    avg_luminosity_decay_ratio: float
    avg_coordinate_accuracy: float
    percent_provenance_intact: float
    percent_narrative_preserved: float
    percent_expandable: float

    # Overall validation
    is_lossless: bool
    major_findings: List[str] = field(default_factory=list)

    def get_losslessness_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive losslessness analysis.
        
        Returns:
            Dictionary with detailed losslessness validation metrics
            
        Analysis Includes:
            - Overall losslessness determination
            - Success criteria validation
            - Quality thresholds assessment
            - Critical failure points identification
        """
        return {
            'is_lossless': self.is_lossless,
            'success_criteria': {
                'provenance_integrity': self.percent_provenance_intact >= 100.0,
                'narrative_preservation': self.percent_narrative_preserved >= 90.0,
                'coordinate_accuracy': self.avg_coordinate_accuracy >= 0.4,
                'compression_efficiency': self.avg_compression_ratio >= 2.0
            },
            'quality_metrics': {
                'avg_compression_ratio': self.avg_compression_ratio,
                'avg_luminosity_decay': self.avg_luminosity_decay_ratio,
                'avg_coordinate_accuracy': self.avg_coordinate_accuracy,
                'provenance_intact_percentage': self.percent_provenance_intact,
                'narrative_preserved_percentage': self.percent_narrative_preserved,
                'expandable_percentage': self.percent_expandable
            },
            'major_findings': self.major_findings,
            'experiment_summary': {
                'bitchains_tested': self.num_bitchains_tested,
                'total_duration': self.total_duration_seconds,
                'paths_analyzed': len(self.compression_paths)
            }
        }

    def get_compression_efficiency_summary(self) -> Dict[str, Any]:
        """
        Get compression efficiency analysis across all paths.
        
        Returns:
            Dictionary with compression performance metrics
            
        Efficiency Metrics:
            - Compression ratio distribution
            - Size reduction effectiveness
            - Performance consistency
            - Optimal compression characteristics
        """
        if not self.compression_paths:
            return {
                'compression_analysis_complete': False,
                'avg_compression_ratio': 0.0,
                'compression_distribution': {},
                'size_reduction_effectiveness': 0.0
            }

        compression_ratios = [p.final_compression_ratio for p in self.compression_paths]
        
        return {
            'compression_analysis_complete': True,
            'avg_compression_ratio': self.avg_compression_ratio,
            'compression_distribution': {
                'min_ratio': min(compression_ratios),
                'max_ratio': max(compression_ratios),
                'median_ratio': sorted(compression_ratios)[len(compression_ratios) // 2],
                'std_deviation': (sum((r - self.avg_compression_ratio) ** 2 for r in compression_ratios) / len(compression_ratios)) ** 0.5 if compression_ratios else 0.0
            },
            'size_reduction_effectiveness': sum(p.original_serialized_size for p in self.compression_paths) / max(sum(p.stages[-1].size_bytes for p in self.compression_paths if p.stages), 1),
            'compression_consistency': 'HIGH' if max(compression_ratios) - min(compression_ratios) < self.avg_compression_ratio * 0.5 else 'MODERATE' if max(compression_ratios) - min(compression_ratios) < self.avg_compression_ratio else 'LOW'
        }

    def get_information_preservation_summary(self) -> Dict[str, Any]:
        """
        Get information preservation analysis across all paths.
        
        Returns:
            Dictionary with information preservation quality metrics
            
        Preservation Metrics:
            - Coordinate accuracy distribution
            - Provenance chain integrity
            - Narrative meaning preservation
            - Reconstruction capability assessment
        """
        if not self.compression_paths:
            return {
                'preservation_analysis_complete': False,
                'avg_coordinate_accuracy': 0.0,
                'provenance_integrity': 0.0,
                'narrative_preservation': 0.0,
                'reconstruction_capability': 0.0
            }

        coordinate_accuracies = [p.coordinate_match_accuracy for p in self.compression_paths]
        provenance_intact_count = sum(1 for p in self.compression_paths if p.provenance_chain_complete)
        narrative_preserved_count = sum(1 for p in self.compression_paths if p.narrative_preserved)
        expandable_count = sum(1 for p in self.compression_paths if p.can_expand_completely)

        return {
            'preservation_analysis_complete': True,
            'avg_coordinate_accuracy': self.avg_coordinate_accuracy,
            'coordinate_accuracy_distribution': {
                'min_accuracy': min(coordinate_accuracies),
                'max_accuracy': max(coordinate_accuracies),
                'median_accuracy': sorted(coordinate_accuracies)[len(coordinate_accuracies) // 2],
                'high_accuracy_paths': sum(1 for acc in coordinate_accuracies if acc >= 0.8),
                'low_accuracy_paths': sum(1 for acc in coordinate_accuracies if acc < 0.2)
            },
            'provenance_integrity': self.percent_provenance_intact,
            'narrative_preservation': self.percent_narrative_preserved,
            'reconstruction_capability': self.percent_expandable,
            'information_loss_points': {
                'provenance_loss_count': len(self.compression_paths) - provenance_intact_count,
                'narrative_loss_count': len(self.compression_paths) - narrative_preserved_count,
                'expansion_failure_count': len(self.compression_paths) - expandable_count
            }
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to JSON-serializable dictionary.
        
        Returns:
            Dictionary representation suitable for JSON export
        """
        return {
            "experiment": "EXP-05",
            "test_type": "Compression/Expansion Losslessness",
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_duration_seconds": round(self.total_duration_seconds, 3),
            "bitchains_tested": self.num_bitchains_tested,
            "aggregate_metrics": {
                "avg_compression_ratio": round(self.avg_compression_ratio, 3),
                "avg_luminosity_decay_ratio": round(self.avg_luminosity_decay_ratio, 4),
                "avg_coordinate_accuracy": round(self.avg_coordinate_accuracy, 4),
                "percent_provenance_intact": round(self.percent_provenance_intact, 1),
                "percent_narrative_preserved": round(
                    self.percent_narrative_preserved, 1
                ),
                "percent_expandable": round(self.percent_expandable, 1),
            },
            "compression_quality": {
                "is_lossless": self.is_lossless,
                "major_findings": self.major_findings,
            },
            "sample_paths": [
                p.calculate_stats()
                for p in self.compression_paths[
                    : min(5, len(self.compression_paths))
                ]  # Show first 5
            ],
            "all_valid": (
                all(
                    p.provenance_chain_complete and p.narrative_preserved
                    for p in self.compression_paths
                )
                if self.compression_paths
                else False
            ),
            "losslessness_analysis": self.get_losslessness_analysis(),
            "compression_efficiency": self.get_compression_efficiency_summary(),
            "information_preservation": self.get_information_preservation_summary()
        }

    def __str__(self) -> str:
        """
        String representation of compression experiment results.
        
        Returns:
            Human-readable summary of the complete experiment results
        """
        return (
            f"Compression Experiment: {self.num_bitchains_tested} bit-chains, "
            f"avg compression {self.avg_compression_ratio:.2f}x, "
            f"lossless: {'YES' if self.is_lossless else 'NO'}, "
            f"duration: {self.total_duration_seconds:.1f}s"
        )