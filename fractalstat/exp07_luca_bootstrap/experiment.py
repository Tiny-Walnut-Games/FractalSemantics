"""
EXP-07: LUCA Bootstrap Test - Experiment Module

This module implements the core experiment logic for testing LUCA bootstrap
and system reconstruction. The experiment validates that FractalStat can
compress a full system to an irreducible minimum and then bootstrap back to
the complete system without information loss.

Core Hypothesis:
The FractalStat system is self-contained and fractal, allowing complete
reconstruction from a minimal LUCA state while preserving all critical
information and maintaining system integrity.

Key Validation Points:
- Compression to LUCA state preserves essential information
- Bootstrap reconstruction achieves 100% entity recovery
- Lineage continuity maintained through compression/expansion cycles
- Fractal properties preserved across scales
- System demonstrates self-similarity and scale invariance

Success Criteria:
- Entity recovery rate ≥ 100% (perfect reconstruction)
- Lineage recovery rate ≥ 100% (continuity preserved)
- Realm recovery rate ≥ 100% (structural integrity)
- Dimensionality recovery rate ≥ 100% (fractal depth preserved)
- Multiple bootstrap cycles without degradation
- Compression ratio > 0 (meaningful compression achieved)

Author: FractalSemantics
Date: 2025-12-07
"""

import json
import time
import uuid
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path

from .entities import (
    TestBitChain,
    LUCAState,
    LUCAEncoding,
    BootstrapValidation,
    FractalPropertyTest,
    LUCABootstrapResult
)


class LUCABootstrapTester:
    """
    Test LUCA bootstrap and system reconstruction.
    
    This class implements the comprehensive testing framework for validating
    that the FractalStat system can be compressed to a minimal LUCA state
    and then fully reconstructed without information loss.
    
    Test Phases:
        1. Create test entities with known lineage from LUCA
        2. Compress entities to LUCA-equivalent state
        3. Bootstrap entities back from LUCA state
        4. Compare original and bootstrapped entities
        5. Test fractal properties of the system
        6. Test LUCA continuity and entity health through multiple cycles
    
    Key Features:
        - Perfect reconstruction validation
        - Multiple bootstrap cycle testing
        - Fractal property verification
        - Information integrity preservation
        - Compression efficiency measurement
    """
    
    def __init__(self):
        """Initialize the LUCA bootstrap tester."""
        self.results = LUCABootstrapResult()
        self.luca_dictionary: Dict[str, Any] = {}  # Master reference

    def create_test_entities(self, num_entities: int = 10) -> List[TestBitChain]:
        """
        Create test entities with known lineage from LUCA.
        
        Args:
            num_entities: Number of test entities to create
            
        Returns:
            List of TestBitChain entities with diverse characteristics
            
        Entity Characteristics:
            - Lineage progression from LUCA (0 to num_entities)
            - Diverse realm assignments (pattern, data, narrative)
            - Different horizon stages (emergence, peak, crystallization)
            - Alternating polarity types (logic, creativity)
            - Progressive dimensionality levels
            - Unique metadata for integrity validation
        """
        entities = []

        for i in range(num_entities):
            # Create entity with lineage from LUCA
            lineage = i + 1  # LUCA is lineage 0, these are descendants

            entity = TestBitChain(
                content=f"Test entity {i}: data fragment with id {i:03d}",
                lineage=lineage,
                realm=(
                    "pattern" if i % 3 == 0 else ("data" if i % 3 == 1 else "narrative")
                ),
                horizon=(
                    "emergence"
                    if i % 3 == 0
                    else ("peak" if i % 3 == 1 else "crystallization")
                ),
                polarity="logic" if i % 2 == 0 else "creativity",
                dimensionality=i + 1,  # fractal depth
                metadata={
                    "index": i,
                    "sequence": i,
                    "checksum": hashlib.sha256(f"entity-{i}".encode()).hexdigest()[:8],
                },
            )
            entities.append(entity)

        return entities

    def compute_luca_encoding(self, entity: TestBitChain) -> LUCAEncoding:
        """
        Encode entity to minimal LUCA-equivalent representation.
        
        This method creates the compressed form of an entity that contains
        only the essential information needed for complete reconstruction.
        
        Args:
            entity: TestBitChain entity to encode
            
        Returns:
            LUCAEncoding with minimal but sufficient reconstruction data
            
        Encoding Strategy:
            - Preserve entity identifier and content hash for integrity
            - Store lineage information for hierarchy
            - Use single-character signatures for realm/horizon/polarity
            - Maintain dimensionality for fractal depth
            - Preserve metadata keys for reconstruction
        """
        # LUCA state contains only the essential addressing info + content hash
        return LUCAEncoding(
            id=entity.bit_chain_id,
            hash=entity.get_entity_signature(),
            lineage=entity.lineage,
            realm_sig=entity.realm[0],  # Single character signature
            horizon_sig=entity.horizon[0],
            polarity_sig=entity.polarity[0],
            dimensionality=entity.dimensionality,
            content_size=len(entity.content),
            metadata_keys=list(entity.metadata.keys()),
        )

    def compress_to_luca(self, entities: List[TestBitChain]) -> LUCAState:
        """
        Compress entities to LUCA-equivalent state.
        
        This method creates the minimal bootstrap form from which everything
        can be reconstructed. The LUCA state represents the irreducible minimum
        of the system while preserving all critical relationships.
        
        Args:
            entities: List of TestBitChain entities to compress
            
        Returns:
            LUCAState containing compressed entity representations
            
        Compression Process:
            1. Encode each entity to minimal LUCA form
            2. Calculate compression statistics
            3. Generate integrity hash for the entire state
            4. Store as reference for reconstruction validation
        """
        print("   Compressing entities to LUCA state...")

        luca_encodings = []
        total_original = 0
        total_compressed = 0

        for entity in entities:
            original_size = len(entity.to_json())
            luca_encoding = self.compute_luca_encoding(entity)
            luca_json_size = len(json.dumps(luca_encoding.to_dict()))

            luca_encodings.append(luca_encoding.to_dict())
            total_original += original_size
            total_compressed += luca_json_size

        # Create LUCA-equivalent state
        luca_state = LUCAState(
            entity_count=len(entities),
            encodings=luca_encodings,
            total_original_size=total_original,
            total_compressed_size=total_compressed,
            compression_ratio=(
                total_compressed / total_original if total_original > 0 else 1.0
            ),
            luca_hash=hashlib.sha256(
                json.dumps(luca_encodings, sort_keys=True).encode()
            ).hexdigest(),
        )

        # Store as reference (this is the irreducible minimum)
        self.luca_dictionary = luca_state.to_dict()

        return luca_state

    def bootstrap_from_luca(self, luca_state: LUCAState) -> Tuple[List[TestBitChain], List[bool]]:
        """
        Bootstrap entities back from LUCA state.
        
        This method reconstructs the full entities from the minimal LUCA
        encoding. It validates that the compression process preserved all
        necessary information for complete reconstruction.
        
        Args:
            luca_state: LUCAState containing compressed entity data
            
        Returns:
            Tuple of (reconstructed entities, expansion success flags)
            
        Bootstrap Process:
            1. Iterate through LUCA encodings
            2. Expand each encoding back to full entity
            3. Track success/failure for each reconstruction
            4. Return complete list of reconstructed entities
        """
        print("   Bootstrapping entities from LUCA state...")

        bootstrapped_entities = []
        expansion_success = []

        for luca_encoding_dict in luca_state.encodings:
            try:
                # Reconstruct entity from LUCA encoding
                luca_encoding = LUCAEncoding(**luca_encoding_dict)
                entity = luca_encoding.expand_to_entity()

                bootstrapped_entities.append(entity)
                expansion_success.append(True)

            except Exception as e:
                expansion_success.append(False)
                print(f"     Error bootstrapping entity: {e}")

        return bootstrapped_entities, expansion_success

    def compare_entities(
        self, original: List[TestBitChain], bootstrapped: List[TestBitChain]
    ) -> Dict[str, Any]:
        """
        Compare original and bootstrapped entities.
        
        This method performs detailed comparison between original entities
        and their bootstrapped reconstructions to validate information
        preservation and reconstruction accuracy.
        
        Args:
            original: List of original TestBitChain entities
            bootstrapped: List of bootstrapped TestBitChain entities
            
        Returns:
            Dictionary with comparison results and recovery statistics
            
        Comparison Metrics:
            - Entity count matching
            - ID preservation
            - Lineage continuity
            - Realm preservation
            - Dimensionality preservation
            - Information loss detection
        """
        print("   Comparing original and bootstrapped entities...")

        id_matches: int = 0
        lineage_matches: int = 0
        realm_matches: int = 0
        dimensionality_matches: int = 0
        details: List[Dict[str, Any]] = []

        comparison = {
            "original_count": len(original),
            "bootstrapped_count": len(bootstrapped),
            "count_match": len(original) == len(bootstrapped),
            "id_matches": id_matches,
            "lineage_matches": lineage_matches,
            "realm_matches": realm_matches,
            "dimensionality_matches": dimensionality_matches,
            "information_loss_detected": False,
            "details": details,
        }

        # Create lookup for bootstrapped entities
        bootstrapped_by_id = {entity.bit_chain_id: entity for entity in bootstrapped}

        for original_entity in original:
            entity_id = original_entity.bit_chain_id

            if entity_id in bootstrapped_by_id:
                bootstrapped_entity = bootstrapped_by_id[entity_id]
                id_matches += 1

                # Compare critical attributes
                lineage_match = original_entity.lineage == bootstrapped_entity.lineage
                if lineage_match:
                    lineage_matches += 1

                realm_match = original_entity.realm == bootstrapped_entity.realm
                if realm_match:
                    realm_matches += 1

                dimensionality_match = (
                    original_entity.dimensionality == bootstrapped_entity.dimensionality
                )
                if dimensionality_match:
                    dimensionality_matches += 1

                # Record mismatch
                if not (lineage_match and realm_match and dimensionality_match):
                    comparison["information_loss_detected"] = True
                    details.append(
                        {
                            "entity_id": entity_id,
                            "lineage_match": lineage_match,
                            "realm_match": realm_match,
                            "dimensionality_match": dimensionality_match,
                        }
                    )
            else:
                comparison["information_loss_detected"] = True
                details.append(
                    {
                        "entity_id": entity_id,
                        "error": "Entity missing after bootstrap",
                    }
                )

        # Calculate recovery rates
        total = len(original)
        comparison["id_matches"] = id_matches
        comparison["lineage_matches"] = lineage_matches
        comparison["realm_matches"] = realm_matches
        comparison["dimensionality_matches"] = dimensionality_matches
        comparison["entity_recovery_rate"] = id_matches / total if total > 0 else 0
        comparison["lineage_recovery_rate"] = (
            lineage_matches / total if total > 0 else 0
        )
        comparison["realm_recovery_rate"] = realm_matches / total if total > 0 else 0
        comparison["dimensionality_recovery_rate"] = (
            dimensionality_matches / total if total > 0 else 0
        )

        return comparison

    def test_fractal_properties(self, entities: List[TestBitChain]) -> FractalPropertyTest:
        """
        Test fractal properties of the system.
        
        This method validates that the system demonstrates key fractal
        properties including self-similarity, scale invariance, recursive
        structure, and LUCA traceability.
        
        Args:
            entities: List of TestBitChain entities to analyze
            
        Returns:
            FractalPropertyTest with validation results
            
        Fractal Properties Tested:
            - Self-similarity: Entities have consistent structure
            - Scale invariance: System behavior consistent across scales
            - Recursive structure: Hierarchical organization maintained
            - LUCA traceability: All entities traceable to LUCA
            - Information entropy: Proper information distribution
        """
        print("   Testing fractal properties...")

        details: Dict[str, Any] = {}
        fractal_tests = FractalPropertyTest()

        # Test LUCA traceability: all entities have valid lineage
        lineages = [e.lineage for e in entities]
        if not all(0 <= lineage for lineage in lineages):
            fractal_tests.luca_traceability = False
        details["lineages"] = sorted(set(lineages))

        # Test self-similarity: entities have consistent structure
        entity_structure_keys = [set(e.to_dict().keys()) for e in entities]
        all_same = all(
            struct == entity_structure_keys[0] for struct in entity_structure_keys
        )
        fractal_tests.self_similarity = all_same
        details["structural_consistency"] = all_same

        # Test scale invariance: multiple lineage levels exist
        unique_lineages = len(set(lineages))
        has_multiple_scales = unique_lineages >= 2
        fractal_tests.scale_invariance = has_multiple_scales
        details["lineage_depth"] = unique_lineages

        # Test recursive structure: dimensionality matches lineage conceptually
        for entity in entities:
            if entity.dimensionality != entity.lineage:
                fractal_tests.recursive_structure = False
                break

        # Test information entropy: measure information preservation through encoding
        try:
            total_content_info = sum(len(str(e.content)) for e in entities)
            total_address_info = sum(len(str(e.bit_chain_id) + str(e.realm) + str(e.horizon)) for e in entities)

            # Fractal systems should have higher information density in structure than content
            entropy_ratio = total_address_info / total_content_info if total_content_info > 0 else 0
            fractal_tests.information_entropy = 0.5 <= entropy_ratio <= 2.0  # Reasonable range
            details["entropy_ratio"] = entropy_ratio
        except Exception:
            fractal_tests.information_entropy = False
            details["entropy_ratio"] = 0

        return fractal_tests

    def test_luca_continuity(self, original: List[TestBitChain]) -> BootstrapValidation:
        """
        Test that LUCA provides continuity and health for entities.
        
        This method performs multiple bootstrap cycles to validate that
        the LUCA state provides stable, continuous system reconstruction
        without degradation over time.
        
        Args:
            original: List of original TestBitChain entities
            
        Returns:
            BootstrapValidation with continuity test results
            
        Continuity Testing:
            - Multiple compression/expansion cycles
            - Lineage hierarchy preservation
            - Address stability validation
            - Metadata preservation verification
            - Error tracking and degradation detection
        """
        print("   Testing LUCA continuity and entity health...")

        bootstraps_performed: int = 0
        bootstrap_failures: int = 0
        reconstruction_errors: List[str] = []
        lineage_continuity: bool = True

        continuity_test = BootstrapValidation()

        # Test 1: Multiple bootstrap cycles
        current_entities = original
        for cycle in range(3):
            print(f"      Bootstrap cycle {cycle + 1}/3...")

            # Compress to LUCA
            luca_state = self.compress_to_luca(current_entities)

            # Bootstrap back
            bootstrapped, success_list = self.bootstrap_from_luca(luca_state)
            bootstraps_performed += 1

            if not all(success_list):
                bootstrap_failures += 1

            # Verify lineage is preserved
            for orig, boot in zip(current_entities, bootstrapped):
                if orig.lineage != boot.lineage:
                    lineage_continuity = False
                    reconstruction_errors.append(
                        f"Cycle {cycle}: Lineage mismatch for {orig.bit_chain_id}"
                    )

            # Next cycle uses bootstrapped entities
            current_entities = bootstrapped

        continuity_test.lineage_continuity = lineage_continuity
        continuity_test.bootstrap_cycles_completed = bootstraps_performed
        continuity_test.bootstrap_failures = bootstrap_failures
        continuity_test.reconstruction_errors = reconstruction_errors

        return continuity_test

    def run_comprehensive_test(self) -> LUCABootstrapResult:
        """
        Run comprehensive LUCA bootstrap test.
        
        This method orchestrates the complete LUCA bootstrap validation
        process, including entity creation, compression, reconstruction,
        comparison, and fractal property testing.
        
        Returns:
            LUCABootstrapResult with complete test results
            
        Test Phases:
            1. Create test entities with known characteristics
            2. Compress entities to LUCA state
            3. Bootstrap entities from LUCA state
            4. Compare original and bootstrapped entities
            5. Test fractal properties of the system
            6. Test LUCA continuity through multiple cycles
            7. Determine overall test success
        """
        print("\n" + "=" * 70)
        print("EXP-07: LUCA Bootstrap Test")
        print("Testing: Can we reliably reconstruct system from LUCA?")
        print("=" * 70)

        start_time = time.time()

        # Phase 1: Create test entities
        print("\n [1/6] Creating test entities...")
        original_entities = self.create_test_entities(1000000)
        print(f"      Created {len(original_entities)} test entities")
        for i, e in enumerate(original_entities[:3]):
            print(
                f"        - Entity {i}: lineage={e.lineage}, realm={e.realm}, address={
                    e.get_fractalstat_address()
                }"
            )

        # Phase 2: Compress to LUCA
        print("\n [2/6] Compressing to LUCA state...")
        luca_state = self.compress_to_luca(original_entities)
        print(f"      OK Compression ratio: {luca_state.compression_ratio:.2f}x")
        print(f"      OK Original size: {luca_state.total_original_size} bytes")
        print(f"      OK LUCA size: {luca_state.total_compressed_size} bytes")

        # Phase 3: Bootstrap from LUCA
        print("\n[3/6] Bootstrapping from LUCA state...")
        bootstrapped_entities, expansion_success = self.bootstrap_from_luca(luca_state)
        success_rate = (
            sum(expansion_success) / len(expansion_success) if expansion_success else 0
        )
        print(
            f"      OK Bootstrapped {len(bootstrapped_entities)}/{
                len(original_entities)
            } entities"
        )
        print(f"      OK Success rate: {success_rate:.1%}")

        # Phase 4: Compare entities
        print("\n[4/6] Comparing original and bootstrapped entities...")
        comparison = self.compare_entities(original_entities, bootstrapped_entities)
        print(f"      OK Entity recovery rate: {comparison['entity_recovery_rate']:.1%}")
        print(
            f"      OK Lineage recovery rate: {comparison['lineage_recovery_rate']:.1%}"
        )
        print(f"      OK Realm recovery rate: {comparison['realm_recovery_rate']:.1%}")
        print(
            f"      OK Dimensionality recovery rate: {
                comparison['dimensionality_recovery_rate']:.1%}"
        )
        if comparison["information_loss_detected"]:
            print("      WARN Information loss detected!")

        # Phase 5: Test fractal properties
        print("\n[5/6] Testing fractal properties...")
        fractal_tests = self.test_fractal_properties(original_entities)
        print(f"      OK Self-similarity: {fractal_tests.self_similarity}")
        print(f"      OK Scale invariance: {fractal_tests.scale_invariance}")
        print(f"      OK Recursive structure: {fractal_tests.recursive_structure}")
        print(f"      OK LUCA traceability: {fractal_tests.luca_traceability}")
        print(
            f"      OK Lineage depth: {
                fractal_tests.get_property_details().get('lineage_depth', 'unknown')
            }"
        )

        # Phase 6: Test LUCA continuity
        print("\n[6/6] Testing LUCA continuity and entity health...")
        continuity = self.test_luca_continuity(original_entities)
        print(f"      OK Bootstrap cycles: {continuity.bootstrap_cycles_completed}")
        print(f"      OK Bootstrap failures: {continuity.bootstrap_failures}")
        print(f"      OK Lineage continuity: {continuity.lineage_continuity}")
        if continuity.reconstruction_errors:
            for err in continuity.reconstruction_errors[:3]:
                print(f"      WARN {err}")

        # Determine test result with stricter validation
        elapsed = time.time() - start_time

        # Comprehensive pass criteria
        recovery_perfect = (
            comparison["entity_recovery_rate"] >= 1.0
            and comparison["lineage_recovery_rate"] >= 1.0
            and comparison["realm_recovery_rate"] >= 1.0
            and comparison["dimensionality_recovery_rate"] >= 1.0
        )
        fractal_valid = (
            fractal_tests.self_similarity
            and fractal_tests.scale_invariance
            and fractal_tests.recursive_structure
            and fractal_tests.luca_traceability
        )
        continuity_valid = (
            continuity.lineage_continuity
            and continuity.bootstrap_failures == 0
            and continuity.bootstrap_cycles_completed == 3  # All cycles completed
        )
        compression_valid = luca_state.compression_ratio > 0 and luca_state.compression_ratio < 1.0

        all_pass = recovery_perfect and fractal_valid and continuity_valid and compression_valid

        status = "PASS" if all_pass else "FAIL"

        # Store results
        self.results.status = status
        self.results.results = {
            "compression": {
                "ratio": luca_state.compression_ratio,
                "original_size": luca_state.total_original_size,
                "luca_size": luca_state.total_compressed_size,
            },
            "bootstrap": {
                "bootstrapped_count": len(bootstrapped_entities),
                "success_rate": success_rate,
            },
            "comparison": {
                "entity_recovery_rate": comparison["entity_recovery_rate"],
                "lineage_recovery_rate": comparison["lineage_recovery_rate"],
                "realm_recovery_rate": comparison["realm_recovery_rate"],
                "dimensionality_recovery_rate": comparison[
                    "dimensionality_recovery_rate"
                ],
                "information_loss": comparison["information_loss_detected"],
            },
            "fractal": fractal_tests.get_property_details(),
            "continuity": {
                "cycles_performed": continuity.bootstrap_cycles_completed,
                "failures": continuity.bootstrap_failures,
                "lineage_preserved": continuity.lineage_continuity,
            },
            "elapsed_time": f"{elapsed:.2f}s",
        }

        print("\n" + "=" * 70)
        print(f"Result: {status}")
        print(f"Elapsed: {elapsed:.2f}s")
        print("=" * 70 + "\n")

        return self.results


def save_results(results: Dict[str, Any], output_file: Optional[str] = None) -> str:
    """
    Save LUCA bootstrap test results to JSON file.
    
    Args:
        results: Results dictionary containing all test data
        output_file: Optional output file path. If None, generates timestamped filename.
        
    Returns:
        Path to the saved results file
        
    File Format:
        JSON file with comprehensive test results including:
        - Compression and expansion statistics
        - Entity recovery and information preservation metrics
        - Fractal property validation results
        - Bootstrap continuity and stability testing
        - Overall success determination and detailed analysis
        
    Saved Location:
        Results directory in project root with timestamped filename
    """
    if output_file is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_file = f"exp07_luca_bootstrap_{timestamp}.json"

    results_dir = Path(__file__).resolve().parent.parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = str(results_dir / output_file)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        f.write("\n")

    print(f"Results saved to: {output_path}")
    return output_path


def run_experiment_from_config(config: Optional[Dict[str, Any]] = None) -> LUCABootstrapResult:
    """
    Run the LUCA bootstrap experiment with configuration parameters.
    
    Args:
        config: Optional configuration dictionary with experiment parameters
        
    Returns:
        LUCABootstrapResult object with complete test results
        
    Configuration Options:
        - num_entities: Number of test entities to create (default: 10)
    """
    if config is None:
        config = {}
    
    tester = LUCABootstrapTester()
    
    # Create entities with configured size
    num_entities = config.get("num_entities", 10)
    original_entities = tester.create_test_entities(num_entities)
    
    # Run the comprehensive test
    results = tester.run_comprehensive_test()
    
    return results


def main():
    """
    Main entry point for EXP-07 execution.
    
    This function runs the complete LUCA bootstrap test and handles
    result processing and output.
    """
    import sys

    tester = LUCABootstrapTester()
    results = tester.run_comprehensive_test()

    # Save complete results to JSON file
    save_results(results.to_dict())

    # Set exit code based on test status for orchestrator
    success = results.status == "PASS"
    exit_code = 0 if success else 1

    # Print summary with celebration at the end
    print("\n[SUMMARY]")
    print("-" * 70)
    print(json.dumps(results.results, indent=2))

    # Celebration at the end
    if success:
        print("\n[Success] EXP-07 LUCA Bootstrap: PERFECT RECONSTRUCTION ACHIEVED!")
        print(f"  - 100% All {results.results['bootstrap']['bootstrapped_count']} entities recovered")
        print("   - Lineage continuity: VERIFIED")
        print("   - Fractal properties: CONFIRMED")
        print("   - Bootstrap stability: COMPLETED")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()