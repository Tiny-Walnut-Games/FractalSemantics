"""
EXP-06: Entanglement Detection - Entities Module

This module defines the core data structures and entities used in the entanglement
detection experiment. These entities capture the mathematical framework for detecting
semantic entanglement between bit-chains through coordinate similarity analysis.

The entities are designed to capture:
- Entanglement score calculations with component breakdowns
- Validation metrics for algorithm performance assessment
- Realm adjacency relationships for cross-realm compatibility
- Mathematical similarity measures for coordinate comparison

Core Mathematical Framework:
E(B1, B2) = 0.5·P + 0.15·R + 0.2·A + 0.1·L + 0.05·ell

Where:
  P = Polarity Resonance (cosine similarity)
  R = Realm Affinity (categorical adjacency)
  A = Adjacency Overlap (Jaccard similarity)
  L = Luminosity Proximity (density distance)
  ell = Lineage Affinity (exponential decay)

Author: FractalSemantics
Date: 2025-12-07
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Any, Optional
from datetime import datetime, timezone


# ============================================================================
# REALM ADJACENCY GRAPH
# ============================================================================

# Realm adjacency graph (symmetric) - defines cross-realm compatibility
REALM_ADJACENCY: Dict[str, Set[str]] = {
    "data": {"data", "narrative", "system", "event", "pattern"},
    "narrative": {
        "data",
        "narrative",
        "system",
        "faculty",
        "event",
        "pattern",
    },
    "system": {"data", "system", "faculty", "pattern"},
    "faculty": {"narrative", "system", "faculty", "event"},
    "event": {"data", "narrative", "faculty", "event", "pattern"},
    "pattern": {"data", "narrative", "system", "event", "pattern", "void"},
    "void": {"pattern", "void"},
}


@dataclass
class EntanglementScore:
    """
    Results of entanglement score computation with detailed component breakdown.
    
    This dataclass captures the complete entanglement analysis between two bit-chains,
    including the total score and individual component contributions that make up
    the mathematical framework.
    
    Mathematical Framework:
        E(B1, B2) = 0.5·P + 0.15·R + 0.2·A + 0.1·L + 0.05·ell
    
    Component Weights:
        - Polarity Resonance (P): 0.5 - Primary entanglement signal
        - Realm Affinity (R): 0.15 - Cross-realm compatibility
        - Adjacency Overlap (A): 0.2 - "Guilt by association" signal
        - Luminosity Proximity (L): 0.1 - Compression state matching
        - Lineage Affinity (ell): 0.05 - Generational relationships
    
    Attributes:
        bitchain1_id: ID of first bit-chain
        bitchain2_id: ID of second bit-chain
        total_score: Overall entanglement score (0.0 to 1.0)
        polarity_resonance: Polarity similarity component (0.0 to 1.0)
        realm_affinity: Realm compatibility component (0.0 to 1.0)
        adjacency_overlap: Neighbor set similarity component (0.0 to 1.0)
        luminosity_proximity: Compression state similarity component (0.0 to 1.0)
        lineage_affinity: Generational relationship component (0.0 to 1.0)
    """
    
    bitchain1_id: str
    bitchain2_id: str
    total_score: float
    polarity_resonance: float
    realm_affinity: float
    adjacency_overlap: float
    luminosity_proximity: float
    lineage_affinity: float

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization and analysis.
        
        Returns:
            Dictionary representation with rounded precision for readability
            
        Dictionary Structure:
            {
                "bitchain1_id": str,
                "bitchain2_id": str,
                "total_score": float,
                "components": {
                    "polarity_resonance": float,
                    "realm_affinity": float,
                    "adjacency_overlap": float,
                    "luminosity_proximity": float,
                    "lineage_affinity": float
                }
            }
        """
        return {
            "bitchain1_id": self.bitchain1_id,
            "bitchain2_id": self.bitchain2_id,
            "total_score": round(self.total_score, 8),
            "components": {
                "polarity_resonance": round(self.polarity_resonance, 8),
                "realm_affinity": round(self.realm_affinity, 8),
                "adjacency_overlap": round(self.adjacency_overlap, 8),
                "luminosity_proximity": round(self.luminosity_proximity, 8),
                "lineage_affinity": round(self.lineage_affinity, 8),
            },
        }

    def get_component_analysis(self) -> Dict[str, Any]:
        """
        Get detailed analysis of component contributions.
        
        Returns:
            Dictionary with component weights, values, and weighted contributions
            
        Analysis Includes:
            - Raw component values
            - Weighted contributions to total score
            - Component rankings by contribution
            - Signal strength categorization
        """
        # Component weights from mathematical framework
        weights = {
            "polarity_resonance": 0.5,
            "realm_affinity": 0.15,
            "adjacency_overlap": 0.2,
            "luminosity_proximity": 0.1,
            "lineage_affinity": 0.05
        }
        
        # Component values
        values = {
            "polarity_resonance": self.polarity_resonance,
            "realm_affinity": self.realm_affinity,
            "adjacency_overlap": self.adjacency_overlap,
            "luminosity_proximity": self.luminosity_proximity,
            "lineage_affinity": self.lineage_affinity
        }
        
        # Weighted contributions
        contributions = {
            name: value * weight 
            for name, value in values.items() 
            for weight in [weights[name]]
        }
        
        # Sort by contribution strength
        sorted_contributions = sorted(
            contributions.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return {
            "component_weights": weights,
            "component_values": values,
            "weighted_contributions": contributions,
            "contribution_ranking": sorted_contributions,
            "dominant_signal": sorted_contributions[0][0] if sorted_contributions else None,
            "signal_strength": "STRONG" if self.total_score >= 0.8 else "MODERATE" if self.total_score >= 0.6 else "WEAK"
        }

    def is_entangled(self, threshold: float = 0.85) -> bool:
        """
        Determine if bit-chains are entangled based on threshold.
        
        Args:
            threshold: Entanglement detection threshold (default: 0.85)
            
        Returns:
            Boolean indicating if entanglement threshold is met
            
        Threshold Interpretation:
            - ≥ 0.85: Strong entanglement (high confidence)
            - 0.70-0.84: Moderate entanglement (medium confidence)
            - 0.50-0.69: Weak entanglement (low confidence)
            - < 0.50: No significant entanglement
        """
        return self.total_score >= threshold

    def get_entanglement_type(self) -> str:
        """
        Classify the type of entanglement based on component analysis.
        
        Returns:
            String describing the entanglement type
            
        Entanglement Types:
            - "IDENTICAL": Perfect match across all components
            - "POLARITY_DOMINANT": Strong polarity resonance
            - "ADJACENCY_DOMINANT": Strong neighbor overlap
            - "CROSS_REALM": Different realms with high compatibility
            - "GENERATIONAL": Strong lineage relationship
            - "WEAK": Low overall similarity
        """
        if self.total_score >= 0.95:
            return "IDENTICAL"
        elif self.polarity_resonance >= 0.8:
            return "POLARITY_DOMINANT"
        elif self.adjacency_overlap >= 0.7:
            return "ADJACENCY_DOMINANT"
        elif self.realm_affinity == 0.7 and self.total_score >= 0.6:
            return "CROSS_REALM"
        elif self.lineage_affinity >= 0.8:
            return "GENERATIONAL"
        else:
            return "WEAK"

    def __str__(self) -> str:
        """
        String representation of entanglement score.
        
        Returns:
            Human-readable summary of entanglement analysis
        """
        return (
            f"Entanglement({self.bitchain1_id[:12]}... ↔ {self.bitchain2_id[:12]}...): "
            f"Score={self.total_score:.3f}, "
            f"Type={self.get_entanglement_type()}, "
            f"Signal={self.get_component_analysis()['dominant_signal']}"
        )


@dataclass
class ValidationResult:
    """
    Results of validation experiment for entanglement detection algorithm.
    
    This dataclass captures the comprehensive validation metrics for assessing
    the performance of the entanglement detection algorithm against ground truth
    relationships.
    
    Validation Metrics:
        - Precision: True positives / (True positives + False positives)
        - Recall: True positives / (True positives + False negatives)
        - F1 Score: Harmonic mean of precision and recall
        - Accuracy: Overall correctness across all predictions
        - Runtime: Performance efficiency measurement
    
    Success Criteria:
        - Precision ≥ 70% (algorithm makes accurate predictions)
        - Recall ≥ 60% (algorithm finds most true relationships)
        - F1 Score ≥ 65% (balanced performance)
        - Runtime efficiency for large-scale analysis
    
    Attributes:
        threshold: Detection threshold used for validation
        true_positives: Correctly identified entangled pairs
        false_positives: Incorrectly identified as entangled
        false_negatives: Missed true entanglements
        true_negatives: Correctly identified as non-entangled
        precision: Precision metric (0.0 to 1.0)
        recall: Recall metric (0.0 to 1.0)
        f1_score: F1 score metric (0.0 to 1.0)
        accuracy: Accuracy metric (0.0 to 1.0)
        runtime_seconds: Time taken for validation (seconds)
    """
    
    threshold: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    runtime_seconds: float

    @property
    def passed(self) -> bool:
        """
        Check if validation passed performance targets.
        
        Returns:
            Boolean indicating if algorithm meets success criteria
            
        Success Criteria:
            - Precision ≥ 70% (algorithm makes accurate predictions)
            - Recall ≥ 60% (algorithm finds most true relationships)
            
        Note: These are initial validation targets. Production systems may
        require higher thresholds (e.g., Precision ≥ 90%, Recall ≥ 85%).
        """
        return self.precision >= 0.70 and self.recall >= 0.60

    @property
    def balanced_performance(self) -> bool:
        """
        Check if algorithm has balanced precision and recall.
        
        Returns:
            Boolean indicating if precision and recall are well-balanced
            
        Balance Criteria:
            - Difference between precision and recall ≤ 0.20
            - Both metrics ≥ 0.65
        """
        return (
            abs(self.precision - self.recall) <= 0.20
            and self.precision >= 0.65
            and self.recall >= 0.65
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization and reporting.
        
        Returns:
            Dictionary representation with comprehensive validation results
            
        Dictionary Structure:
            {
                "threshold": float,
                "confusion_matrix": {
                    "true_positives": int,
                    "false_positives": int,
                    "false_negatives": int,
                    "true_negatives": int
                },
                "metrics": {
                    "precision": float,
                    "recall": float,
                    "f1_score": float,
                    "accuracy": float
                },
                "runtime_seconds": float,
                "passed": bool,
                "balanced_performance": bool
            }
        """
        return {
            "threshold": self.threshold,
            "confusion_matrix": {
                "true_positives": self.true_positives,
                "false_positives": self.false_positives,
                "false_negatives": self.false_negatives,
                "true_negatives": self.true_negatives,
            },
            "metrics": {
                "precision": round(self.precision, 4),
                "recall": round(self.recall, 4),
                "f1_score": round(self.f1_score, 4),
                "accuracy": round(self.accuracy, 4),
            },
            "runtime_seconds": round(self.runtime_seconds, 4),
            "passed": self.passed,
            "balanced_performance": self.balanced_performance,
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary with interpretation.
        
        Returns:
            Dictionary with performance analysis and recommendations
            
        Summary Includes:
            - Performance categorization (Excellent, Good, Fair, Poor)
            - Strengths and weaknesses analysis
            - Recommendations for threshold adjustment
            - Confidence level assessment
        """
        # Performance categorization
        if self.precision >= 0.90 and self.recall >= 0.85:
            performance_level = "EXCELLENT"
        elif self.precision >= 0.80 and self.recall >= 0.75:
            performance_level = "GOOD"
        elif self.precision >= 0.70 and self.recall >= 0.60:
            performance_level = "FAIR"
        else:
            performance_level = "POOR"

        # Strengths analysis
        strengths = []
        if self.precision >= 0.85:
            strengths.append("High precision (few false positives)")
        if self.recall >= 0.80:
            strengths.append("High recall (few false negatives)")
        if self.balanced_performance:
            strengths.append("Balanced precision and recall")
        if self.f1_score >= 0.80:
            strengths.append("High F1 score (overall good performance)")

        # Weaknesses analysis
        weaknesses = []
        if self.precision < 0.70:
            weaknesses.append("Low precision (many false positives)")
        if self.recall < 0.60:
            weaknesses.append("Low recall (many false negatives)")
        if not self.balanced_performance:
            weaknesses.append("Imbalanced precision and recall")
        if self.f1_score < 0.65:
            weaknesses.append("Low F1 score (poor overall performance)")

        # Recommendations
        recommendations = []
        if self.precision < 0.70:
            recommendations.append("Increase threshold to reduce false positives")
        elif self.recall < 0.60:
            recommendations.append("Decrease threshold to reduce false negatives")
        if not self.balanced_performance:
            recommendations.append("Adjust threshold for better balance between precision and recall")

        return {
            "performance_level": performance_level,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "recommendations": recommendations,
            "confidence_level": "HIGH" if self.passed and self.balanced_performance else "MEDIUM" if self.passed else "LOW",
            "threshold_optimization_needed": len(recommendations) > 0
        }

    def __str__(self) -> str:
        """
        String representation of validation results.
        
        Returns:
            Human-readable summary of validation performance
        """
        return (
            f"Validation(threshold={self.threshold}): "
            f"Precision={self.precision:.3f}, Recall={self.recall:.3f}, "
            f"F1={self.f1_score:.3f}, Passed={'YES' if self.passed else 'NO'}"
        )


@dataclass
class EntanglementAnalysis:
    """
    Comprehensive analysis of entanglement detection results.
    
    This dataclass provides a complete analysis of entanglement detection
    across a dataset, including statistical summaries, distribution analysis,
    and performance metrics.
    
    Analysis Components:
        - Score distribution analysis
        - Entanglement type classification
        - Cross-realm relationship analysis
        - Performance efficiency metrics
        - Quality assessment of detected relationships
    
    Attributes:
        total_pairs_analyzed: Total number of bit-chain pairs analyzed
        entangled_pairs_found: Number of pairs identified as entangled
        average_score: Mean entanglement score across all pairs
        score_distribution: Statistical distribution of scores
        entanglement_types: Classification of entanglement types found
        cross_realm_connections: Analysis of cross-realm entanglements
        performance_metrics: Efficiency and quality metrics
    """
    
    total_pairs_analyzed: int
    entangled_pairs_found: int
    average_score: float
    score_distribution: Dict[str, float]
    entanglement_types: Dict[str, int]
    cross_realm_connections: int
    performance_metrics: Dict[str, float]

    def get_analysis_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive analysis summary.
        
        Returns:
            Dictionary with complete analysis results and insights
            
        Summary Includes:
            - Overall entanglement density
            - Quality assessment of detected relationships
            - Cross-realm compatibility analysis
            - Performance efficiency evaluation
            - Recommendations for algorithm tuning
        """
        entanglement_density = (
            self.entangled_pairs_found / self.total_pairs_analyzed
            if self.total_pairs_analyzed > 0
            else 0.0
        )
        
        # Quality assessment
        quality_level = "HIGH" if self.average_score >= 0.8 else "MEDIUM" if self.average_score >= 0.6 else "LOW"
        
        # Cross-realm analysis
        cross_realm_ratio = (
            self.cross_realm_connections / self.entangled_pairs_found
            if self.entangled_pairs_found > 0
            else 0.0
        )
        
        return {
            "entanglement_density": round(entanglement_density, 4),
            "average_entanglement_score": round(self.average_score, 4),
            "quality_assessment": quality_level,
            "cross_realm_ratio": round(cross_realm_ratio, 4),
            "total_analyzed": self.total_pairs_analyzed,
            "entangled_found": self.entangled_pairs_found,
            "score_distribution": self.score_distribution,
            "entanglement_types": self.entanglement_types,
            "performance_metrics": self.performance_metrics
        }

    def __str__(self) -> str:
        """
        String representation of entanglement analysis.
        
        Returns:
            Human-readable summary of analysis results
        """
        return (
            f"Analysis(total={self.total_pairs_analyzed}, entangled={self.entangled_pairs_found}, "
            f"avg_score={self.average_score:.3f}, cross_realm={self.cross_realm_connections})"
        )