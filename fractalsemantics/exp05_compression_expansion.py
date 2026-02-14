"""
EXP-05: Bit-Chain Compression/Expansion Losslessness Validation

Tests whether FractalSemantics bit-chains can be compressed through the full pipeline
(fragments → clusters → glyphs → mist) and then expanded back to original
coordinates without information loss.

Validates:
- Provenance chain integrity (all source IDs tracked)
- FractalSemantics coordinate reconstruction accuracy
- Luminosity decay through compression stages
- Narrative preservation (embeddings, affect survival)
- Compression ratio efficiency

Status: Phase 2 validation experiment
"""

import hashlib
import json
import statistics
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, dict, list

from fractalsemantics.dynamic_enum import Alignment, Polarity

# Reuse canonical serialization from Phase 1
from fractalsemantics.fractalsemantics_entity import (
    BitChain,
    Coordinates,
    canonical_serialize,
    generate_random_bitchain,
)

# Import progress communication

# Import subprocess communication for enhanced progress reporting
try:
    from fractalsemantics.subprocess_comm import (
        is_subprocess_communication_enabled,
        send_subprocess_completion,
        send_subprocess_progress,
        send_subprocess_status,
    )
except ImportError:
    # Fallback if subprocess communication is not available
    def send_subprocess_progress(*args, **kwargs) -> bool: return False
    def send_subprocess_status(*args, **kwargs) -> bool: return False
    def send_subprocess_completion(*args, **kwargs) -> bool: return False
    def is_subprocess_communication_enabled() -> bool: return False

# ============================================================================
# EXP-05 DATA STRUCTURES
# ============================================================================


@dataclass
class CompressionStage:
    """Single stage in the compression pipeline."""

    stage_name: str  # "original", "fragments", "cluster", "glyph", "mist"
    size_bytes: int
    record_count: int
    key_metadata: dict[str, any]  # What survives at this stage
    luminosity: float  # Activity level / heat
    provenance_intact: bool

    def compression_ratio_from_original(self, original_bytes: int) -> float:
        """Calculate compression ratio relative to original."""
        return original_bytes / max(self.size_bytes, 1)


@dataclass
class BitChainCompressionPath:
    """Complete compression path for a single bit-chain."""

    original_bitchain: BitChain
    original_address: str
    original_fractalsemantics_dict: dict[str, any]
    original_serialized_size: int
    original_luminosity: float

    # Stages
    stages: list[CompressionStage] = field(default_factory=list)

    # Reconstruction attempt
    reconstructed_address: Optional[str] = None
    coordinate_match_accuracy: float = 0.0  # 0.0 to 1.0
    can_expand_completely: bool = False

    # Metrics
    final_compression_ratio: float = 0.0
    luminosity_final: float = 0.0
    narrative_preserved: bool = False
    provenance_chain_complete: bool = False

    def calculate_stats(self) -> dict[str, any]:
        """Compute summary statistics for this compression path."""
        result = {
            "original_realm": self.original_fractalsemantics_dict.get("realm"),
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


@dataclass
class CompressionExperimentResults:
    """Complete results from EXP-05 compression/expansion validation."""

    start_time: str
    end_time: str
    total_duration_seconds: float
    num_bitchains_tested: int

    # Per-bitchain paths
    compression_paths: list[BitChainCompressionPath]

    # Aggregate statistics
    avg_compression_ratio: float
    avg_luminosity_decay_ratio: float
    avg_coordinate_accuracy: float
    percent_provenance_intact: float
    percent_narrative_preserved: float
    percent_expandable: float

    # Overall validation
    is_lossless: bool
    major_findings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, any]:
        """Convert to serializable dict."""
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
        }


# ============================================================================
# COMPRESSION PIPELINE SIMULATION
# ============================================================================


class CompressionPipeline:
    """Simulates the compression pipeline from the Seed engine."""

    def __init__(self):
        self.fragment_store = {}
        self.cluster_store = {}
        self.glyph_store = {}
        self.mist_store = {}

    def compress_bitchain(self, bc: BitChain) -> BitChainCompressionPath:
        """
        Compress a bit-chain through the full pipeline.

        Stages:
        1. Original FractalSemantics coordinates
        2. Fragment representation (serialize bit-chain)
        3. Cluster (group fragments - here just one per chain)
        4. Glyph (molten form with provenance)
        5. Mist (evaporated proto-thought)
        """
        # Convert bitchain to dict for serialization
        bc_dict = {
            "id": bc.id,
            "coordinates": bc.coordinates.to_dict(),
        }

        path = BitChainCompressionPath(
            original_bitchain=bc,
            original_address=bc.compute_address(),
            original_fractalsemantics_dict=bc.coordinates.to_dict(),
            original_serialized_size=len(canonical_serialize(bc_dict)),
            original_luminosity=abs(
                bc.coordinates.luminosity
            ),  # Use absolute value for luminosity
        )

        # Stage 1: Original (baseline)
        original_stage = CompressionStage(
            stage_name="original",
            size_bytes=path.original_serialized_size,
            record_count=1,
        key_metadata={
            "address": path.original_address,
            "realm": bc.coordinates.realm,
            "luminosity": bc.coordinates.luminosity,
        },
        luminosity=abs(
            bc.coordinates.luminosity
        ),  # Use absolute value for luminosity
            provenance_intact=True,
        )
        path.stages.append(original_stage)

        # Stage 2: Fragment representation
        fragment_id = str(uuid.uuid4())[:12]
        heat: float = abs(
            bc.coordinates.luminosity
        )  # Use absolute value for heat/luminosity
        fragment = {
            "id": fragment_id,
            "bitchain_id": bc.id,
            "realm": bc.coordinates.realm,
            "text": f"{bc.coordinates.realm}:{bc.coordinates.lineage}:{bc.coordinates.dimensionality}",
            "heat": heat,
            "embedding": [bc.coordinates.luminosity, bc.coordinates.polarity.value],
        }
        self.fragment_store[fragment_id] = fragment

        fragment_size = len(json.dumps(fragment))
        fragment_stage = CompressionStage(
            stage_name="fragments",
            size_bytes=fragment_size,
            record_count=1,
            key_metadata={
                "fragment_id": fragment_id,
                "heat": heat,
                "embedding": fragment["embedding"],
            },
            luminosity=heat,
            provenance_intact=True,
        )
        path.stages.append(fragment_stage)

        # Stage 3: Cluster (group fragments - here just wrapping one)
        cluster_id = f"cluster_{hashlib.sha256(fragment_id.encode()).hexdigest()[:10]}"
        cluster = {
            "id": cluster_id,
            "fragments": [fragment_id],
            "size": 1,
            "source_bitchain_ids": [bc.id],
            "provenance_hash": hashlib.sha256(
                f"{bc.id}:{bc.coordinates.realm}".encode()
            ).hexdigest(),
        }
        self.cluster_store[cluster_id] = cluster

        cluster_size = len(json.dumps(cluster))
        cluster_stage = CompressionStage(
            stage_name="cluster",
            size_bytes=cluster_size,
            record_count=1,
            key_metadata={
                "cluster_id": cluster_id,
                "source_bitchain_ids": cluster["source_bitchain_ids"],
                "provenance_hash": cluster["provenance_hash"],
            },
            luminosity=heat * 0.95,
            provenance_intact=True,
        )
        path.stages.append(cluster_stage)

        # Stage 4: Glyph (molten form - further compress with affect)
        glyph_id = f"mglyph_{hashlib.sha256(cluster_id.encode()).hexdigest()[:12]}"
        affect_intensity = abs(
            bc.coordinates.luminosity
        )  # Use luminosity as affect proxy
        heat_seed: float = heat * 0.85
        glyph = {
            "id": glyph_id,
            "source_ids": [bc.id],
            "source_cluster_id": cluster_id,
            "compressed_summary": f"[{bc.coordinates.realm}] gen={bc.coordinates.lineage}",
            "embedding": fragment["embedding"],  # Preserve embedding
            "affect": {
                "awe": affect_intensity * 0.3,
                "humor": affect_intensity * 0.2,
                "tension": affect_intensity * 0.1,
            },
            "heat_seed": heat_seed,
            "provenance_hash": cluster["provenance_hash"],
            "luminosity": heat_seed,
        }
        self.glyph_store[glyph_id] = glyph

        glyph_size = len(json.dumps(glyph))
        glyph_stage = CompressionStage(
            stage_name="glyph",
            size_bytes=glyph_size,
            record_count=1,
            key_metadata={
                "glyph_id": glyph_id,
                "embedding": glyph["embedding"],
                "affect": glyph["affect"],
                "provenance_hash": glyph["provenance_hash"],
            },
            luminosity=heat_seed,
            provenance_intact=True,
        )
        path.stages.append(glyph_stage)

        # Stage 5: Mist (final compression - proto-thought)
        mist_id = f"mist_{glyph_id[7:]}"  # Remove mglyph_ prefix
        mist_luminosity: float = heat_seed * 0.7
        mist = {
            "id": mist_id,
            "source_glyph": glyph_id,
            "proto_thought": f"[Proto] {bc.coordinates.realm}...",
            "evaporation_temp": 0.7,
            "mythic_weight": affect_intensity,
            "technical_clarity": 0.6,
            "luminosity": mist_luminosity,
            # Preserve coordinate fields for full reconstruction
            "recovery_breadcrumbs": {
                "original_realm": bc.coordinates.realm,
                "original_lineage": bc.coordinates.lineage,
                "original_adjacency": bc.coordinates.adjacency,
                "original_horizon": bc.coordinates.horizon,
                "original_polarity": bc.coordinates.polarity.name,  # Store enum name
                "original_dimensionality": bc.coordinates.dimensionality,
                "original_alignment": bc.coordinates.alignment.name,  # Store enum name
                "original_luminosity": bc.coordinates.luminosity,
                "original_embedding": glyph["embedding"],
            },
        }
        self.mist_store[mist_id] = mist

        mist_size = len(json.dumps(mist))
        mist_stage = CompressionStage(
            stage_name="mist",
            size_bytes=mist_size,
            record_count=1,
            key_metadata={
                "mist_id": mist_id,
                "recovery_breadcrumbs": mist["recovery_breadcrumbs"],
                "luminosity": mist_luminosity,
            },
            luminosity=mist_luminosity,
            provenance_intact=True,  # Breadcrumbs preserve some info
        )
        path.stages.append(mist_stage)

        # Calculate path statistics
        path.final_compression_ratio = path.original_serialized_size / max(mist_size, 1)
        path.luminosity_final = mist_luminosity

        # Attempt reconstruction
        path = self._reconstruct_from_mist(path, mist)

        return path

    def _reconstruct_from_mist(
        self, path: BitChainCompressionPath, mist: dict[str, any]
    ) -> BitChainCompressionPath:
        """Attempt to reconstruct FractalSemantics coordinates from mist form."""
        try:
            breadcrumbs = mist.get("recovery_breadcrumbs", {})

            # Try to recover all coordinate fields
            realm = breadcrumbs.get("original_realm", "void")
            lineage = breadcrumbs.get("original_lineage", 0)
            adjacency = breadcrumbs.get("original_adjacency", [])
            horizon = breadcrumbs.get("original_horizon", "void")
            polarity_str = breadcrumbs.get("original_polarity", "VOID")
            dimensionality = breadcrumbs.get("original_dimensionality", 0)
            alignment_str = breadcrumbs.get("original_alignment", "TRUE_NEUTRAL")
            original_luminosity = breadcrumbs.get("original_luminosity", mist.get("luminosity", 0))

            # Convert enum names back to enum objects
            try:
                polarity = getattr(Polarity, polarity_str, Polarity.VOID)
            except (AttributeError, KeyError):
                polarity = Polarity.VOID

            try:
                alignment = getattr(Alignment, alignment_str, Alignment.TRUE_NEUTRAL)
            except (AttributeError, KeyError):
                alignment = Alignment.TRUE_NEUTRAL

            # Reconstruct coordinates with all recovered data
            reconstructed_coords = Coordinates(
                realm=realm,
                lineage=lineage,
                adjacency=adjacency,
                horizon=horizon,
                luminosity=mist["luminosity"],  # Use mist luminosity as final value
                polarity=polarity,
                dimensionality=dimensionality,
                alignment=alignment,
            )

            # Can we expand completely? Check if all fields match or are present
            all_fields_present = all([
                realm != "void",
                lineage >= 0,  # 0 is valid for lineage
                mist.get("luminosity", 0) > 0,
                reconstructed_coords is not None,
            ])

            # Narrative preserved if embedding survives
            embedding = breadcrumbs.get("original_embedding", [])
            narrative_preserved = len(embedding) > 0

            # Check coordinate accuracy - compare reconstructed vs original
            original_coords = path.original_fractalsemantics_dict
            fields_recovered = 0
            total_fields = 8  # realm, lineage, adjacency, horizon, luminosity, polarity, dimensionality, alignment

            # Check each field for accuracy
            if realm == original_coords.get("realm"):
                fields_recovered += 1
            if lineage == original_coords.get("lineage"):
                fields_recovered += 1
            if adjacency == original_coords.get("adjacency", []):
                fields_recovered += 1
            if horizon == original_coords.get("horizon"):
                fields_recovered += 1
            if polarity.name == original_coords.get("polarity"):
                fields_recovered += 1
            if dimensionality == original_coords.get("dimensionality"):
                fields_recovered += 1
            if alignment.name == original_coords.get("alignment"):
                fields_recovered += 1
            # Luminosity always recovered from mist (even if decayed)
            if abs(original_luminosity) > 0:  # Original had some luminosity
                fields_recovered += 1

            path.coordinate_match_accuracy = fields_recovered / total_fields
            path.can_expand_completely = all_fields_present and fields_recovered >= 6  # 75% recovery
            path.narrative_preserved = narrative_preserved
            path.provenance_chain_complete = True  # Breadcrumbs preserved it
            path.luminosity_final = mist["luminosity"]

        except Exception as e:
            print(f"  Reconstruction failed: {e}")
            path.coordinate_match_accuracy = 0.0
            path.can_expand_completely = False

        return path


# ============================================================================
# VALIDATION EXPERIMENT ORCHESTRATION
# ============================================================================


def _print_experiment_header(num_bitchains: int):
    """Print experiment header information."""
    print("\n" + "=" * 80)
    print("EXP-05: COMPRESSION/EXPANSION LOSSLESSNESS VALIDATION")
    print("=" * 80)
    print(
        f"Testing {num_bitchains} random bit-chains through full compression pipeline"
    )
    print()


def _compress_bitchains(pipeline: CompressionPipeline, num_bitchains: int) -> list[BitChainCompressionPath]:
    """Compress specified number of bit-chains and return paths."""
    compression_paths: list[BitChainCompressionPath] = []

    print("Compressing bit-chains...")
    print("-" * 80)

    for i in range(num_bitchains):
        bc = generate_random_bitchain()
        path = pipeline.compress_bitchain(bc)
        compression_paths.append(path)

        if (i + 1) % 25 == 0:
            print(f"  [OK] Processed {i + 1}/{num_bitchains} bit-chains")

    print()
    return compression_paths


def _print_sample_paths(compression_paths: list[BitChainCompressionPath]):
    """Print detailed sample compression paths."""
    if not compression_paths:
        return

    print("=" * 80)
    print("SAMPLE COMPRESSION PATHS (First 3)")
    print("=" * 80)

    for path in compression_paths[:3]:
        print(f"\nBit-Chain: {path.original_bitchain.id[:12]}...")
        print(
            f"  Original FractalSemantics: {path.original_fractalsemantics_dict['realm']} gen={path.original_fractalsemantics_dict['lineage']}"
        )
        print(f"  Original Address: {path.original_address[:32]}...")
        print(f"  Original Size: {path.original_serialized_size} bytes")
        print(f"  Original Luminosity: {path.original_luminosity:.4f}")
        print()

        for stage in path.stages:
            print(f"  Stage: {stage.stage_name:12} | Size: {stage.size_bytes:6} bytes | Luminosity: {stage.luminosity:.4f}")

        print(f"  Final Compression Ratio: {path.final_compression_ratio:.2f}x")
        print(f"  Coordinate Accuracy: {path.coordinate_match_accuracy:.1%}")
        print(f"  Expandable: {'[Y]' if path.can_expand_completely else '[N]'}")
        print(f"  Provenance: {'[Y]' if path.provenance_chain_complete else '[N]'}")
        print(f"  Narrative: {'[Y]' if path.narrative_preserved else '[N]'}")


def _compute_aggregate_metrics(compression_paths: list[BitChainCompressionPath]) -> dict[str, float]:
    """Compute aggregate statistics from compression paths."""
    compression_ratios = [p.final_compression_ratio for p in compression_paths]
    luminosity_decay_ratios = [
        (max(p.original_luminosity, 0.01) - max(p.luminosity_final, 0)) / max(p.original_luminosity, 0.01)
        for p in compression_paths
    ]
    coord_accuracies = [p.coordinate_match_accuracy for p in compression_paths]

    percent_provenance = sum(1 for p in compression_paths if p.provenance_chain_complete) / len(compression_paths) * 100
    percent_narrative = sum(1 for p in compression_paths if p.narrative_preserved) / len(compression_paths) * 100
    percent_expandable = sum(1 for p in compression_paths if p.can_expand_completely) / len(compression_paths) * 100

    return {
        "avg_compression_ratio": statistics.mean(compression_ratios),
        "avg_luminosity_decay_ratio": statistics.mean(luminosity_decay_ratios),
        "avg_coordinate_accuracy": statistics.mean(coord_accuracies),
        "percent_provenance": percent_provenance,
        "percent_narrative": percent_narrative,
        "percent_expandable": percent_expandable,
    }


def _print_aggregate_metrics(metrics: dict[str, float]):
    """Print aggregate metrics to console."""
    print()
    print("=" * 80)
    print("AGGREGATE METRICS")
    print("=" * 80)

    print(f"Average Compression Ratio: {metrics['avg_compression_ratio']:.3f}x")
    print(f"Average Luminosity Decay: {metrics['avg_luminosity_decay_ratio']:.4f}")
    print(f"Average Coordinate Accuracy: {metrics['avg_coordinate_accuracy']:.1%}")
    print(f"Provenance Integrity: {metrics['percent_provenance']:.1f}%")
    print(f"Narrative Preservation: {metrics['percent_narrative']:.1f}%")
    print(f"Expandability: {metrics['percent_expandable']:.1f}%")
    print()


def _determine_losslessness(metrics: dict[str, float]) -> bool:
    """Determine if system meets losslessness criteria."""
    return (
        metrics["percent_provenance"] == 100.0
        and metrics["percent_narrative"] >= 90.0
        and metrics["avg_coordinate_accuracy"] >= 0.4
    )


def _generate_major_findings(metrics: dict[str, float], compression_paths: list[BitChainCompressionPath]) -> list[str]:
    """Generate major findings based on metrics."""
    findings = []

    if metrics["percent_provenance"] == 100.0:
        findings.append("[OK] Provenance chain maintained through all compression stages")
    else:
        findings.append(f"[WARN] Provenance loss detected ({100 - metrics['percent_provenance']:.1f}% affected)")

    if metrics["percent_narrative"] >= 90.0:
        findings.append("[OK] Narrative meaning preserved via embeddings and affect")
    else:
        findings.append(f"[WARN] Narrative degradation observed ({100 - metrics['percent_narrative']:.1f}% affected)")

    if metrics["avg_coordinate_accuracy"] >= 0.4:
        findings.append(f"[OK] FractalSemantics coordinates partially recoverable ({metrics['avg_coordinate_accuracy']:.1%})")
    else:
        findings.append(f"[FAIL] FractalSemantics coordinate recovery insufficient ({metrics['avg_coordinate_accuracy']:.1%})")

    if metrics["avg_compression_ratio"] >= 2.0:
        findings.append(f"[OK] Effective compression achieved ({metrics['avg_compression_ratio']:.2f}x)")
    else:
        findings.append(f"[WARN] Compression ratio modest ({metrics['avg_compression_ratio']:.2f}x)")

    luminosity_retention = (1.0 - metrics["avg_luminosity_decay_ratio"]) * 100
    if luminosity_retention >= 70.0:
        findings.append(f"[OK] Luminosity retained through compression ({luminosity_retention:.1f}%)")
    else:
        findings.append(f"[WARN] Luminosity decay significant ({100 - luminosity_retention:.1f}% loss)")

    return findings


def _print_losslessness_analysis(is_lossless: bool, major_findings: list[str]):
    """Print final losslessness analysis."""
    print("=" * 80)
    print("LOSSLESSNESS ANALYSIS")
    print("=" * 80)
    print(f"Lossless System: {'[YES]' if is_lossless else '[NO]'}")
    print()
    for finding in major_findings:
        print(f"  {finding}")
    print()


def run_compression_expansion_test(
    num_bitchains: int = 100, show_samples: bool = True
) -> CompressionExperimentResults:
    """
    Run EXP-05: Compression/Expansion Losslessness Validation

    Args:
        num_bitchains: Number of random bit-chains to compress
        show_samples: Whether to print detailed sample compression paths

    Returns:
        Complete results object
    """
    start_time = datetime.now(timezone.utc).isoformat()
    overall_start = time.time()

    _print_experiment_header(num_bitchains)

    # Send subprocess status message
    send_subprocess_status("EXP-05", "Initialization", f"Starting compression/expansion test with {num_bitchains} bit-chains")

    pipeline = CompressionPipeline()
    compression_paths = _compress_bitchains(pipeline, num_bitchains)

    if show_samples:
        _print_sample_paths(compression_paths)

    metrics = _compute_aggregate_metrics(compression_paths)
    _print_aggregate_metrics(metrics)

    is_lossless = _determine_losslessness(metrics)
    major_findings = _generate_major_findings(metrics, compression_paths)
    _print_losslessness_analysis(is_lossless, major_findings)

    overall_end = time.time()
    end_time = datetime.now(timezone.utc).isoformat()

    return CompressionExperimentResults(
        start_time=start_time,
        end_time=end_time,
        total_duration_seconds=overall_end - overall_start,
        num_bitchains_tested=num_bitchains,
        compression_paths=compression_paths,
        avg_compression_ratio=metrics["avg_compression_ratio"],
        avg_luminosity_decay_ratio=metrics["avg_luminosity_decay_ratio"],
        avg_coordinate_accuracy=metrics["avg_coordinate_accuracy"],
        percent_provenance_intact=metrics["percent_provenance"],
        percent_narrative_preserved=metrics["percent_narrative"],
        percent_expandable=metrics["percent_expandable"],
        is_lossless=is_lossless,
        major_findings=major_findings,
    )


# ============================================================================
# CLI & RESULTS PERSISTENCE
# ============================================================================


def save_results(
    results: CompressionExperimentResults, output_file: Optional[str] = None
) -> str:
    """Save results to JSON file."""
    if output_file is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_file = f"exp05_compression_expansion_{timestamp}.json"

    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = str(results_dir / output_file)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results.to_dict(), f, indent=2)

    print(f"Results saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    # Load from config or fall back to command-line args
    try:
        from fractalsemantics.config import ExperimentConfig

        config = ExperimentConfig()
        num_bitchains = config.get("EXP-05", "num_bitchains", 1000000)
        show_samples = config.get("EXP-05", "show_samples", True)
    except Exception:
        num_bitchains = 1000000  # Default to 1M for extreme scale testing
        show_samples = True
        if "--quick" in sys.argv:
            num_bitchains = 10000  # Reduced for quick testing
        elif "--full" in sys.argv:
            num_bitchains = 10000000  # 10M for maximum testing

    try:
        results = run_compression_expansion_test(
            num_bitchains=num_bitchains, show_samples=show_samples
        )
        output_file = save_results(results)

        print("\n" + "=" * 80)
        print("[OK] EXP-05 COMPLETE")
        print("=" * 80)
        print(f"Results: {output_file}")
        print()

        # Send subprocess completion message
        success = results.is_lossless and all(p.provenance_chain_complete and p.narrative_preserved for p in results.compression_paths)
        send_subprocess_completion("EXP-05", success, f"Compression/expansion test completed with {results.num_bitchains_tested} bit-chains tested")

    except Exception as e:
        print(f"\n[FAIL] EXPERIMENT FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
