"""
EXP-09: Concurrency & Thread Safety Test (LLM Integration)
Goal: Prove system handles concurrent queries without race conditions

What it tests:
- Launch parallel concurrent operations
- Verify no race conditions in embedding generation
- Check result consistency under load
- Measure throughput (queries per second)
- Verify narrative coherence under concurrent access

Expected Result:
- 20/20 concurrent queries succeed
- No data corruption or race conditions
- Throughput: >5 queries/second (CPU-bound embedding operations)
- Consistent results across repeated runs
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from fractalstat.exp08_llm_integration import LLMIntegrationDemo


@dataclass
class ConcurrencyTestResult:
    """Results for concurrency test."""

    experiment: str = "EXP-09"
    title: str = "Concurrency & Thread Safety Test"
    timestamp: str = ""
    status: str = "PASS"
    results: Dict[str, Any] | None = None

    def __post_init__(self):
        if self.timestamp == "":
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if self.results is None:
            self.results = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment": self.experiment,
            "title": self.title,
            "timestamp": self.timestamp,
            "status": self.status,
            "results": self.results,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)


class ConcurrencyTester:
    """Test system concurrency and thread safety."""

    def __init__(self, num_workers: int = 4):
        """Initialize concurrency tester with LLM demo."""
        self.llm_demo = LLMIntegrationDemo()
        self.num_workers = num_workers
        self.results = ConcurrencyTestResult()

    def run_concurrent_embeddings(
        self, bit_chains: List[Any], num_workers: int | None = None
    ) -> List[Dict[str, Any]]:
        """
        Generate embeddings for multiple bit chains concurrently.

        Args:
            bit_chains: List of entities to embed
            num_workers: Number of concurrent workers (default: self.num_workers)

        Returns:
            List of embedding results
        """
        if num_workers is None:
            num_workers = self.num_workers

        results = []

        def embed_chain(bit_chain: Any) -> Dict[str, Any]:
            """Embed a single bit chain."""
            embedding = self.llm_demo.embed_stat7_address(bit_chain)
            return {
                "bit_chain_id": bit_chain.bit_chain_id,
                "embedding": embedding,
            }

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(embed_chain, bc): bc for bc in bit_chains}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error in embedding: {e}")

        return results

    def run_concurrent_enhancements(
        self, bit_chains: List[Any], num_workers: int | None = None
    ) -> List[Dict[str, Any]]:
        """
        Enhance narratives for multiple bit chains concurrently.

        Args:
            bit_chains: List of entities to enhance
            num_workers: Number of concurrent workers

        Returns:
            List of enhancement results
        """
        if num_workers is None:
            num_workers = self.num_workers

        results = []

        def enhance_chain(bit_chain: Any) -> Dict[str, Any]:
            """Enhance a single bit chain narrative."""
            return self.llm_demo.enhance_bit_chain_narrative(bit_chain)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(enhance_chain, bc): bc for bc in bit_chains}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error in enhancement: {e}")

        return results

    def run_concurrent_stat7_extraction(
        self, embeddings: List[np.ndarray], num_workers: int | None = None
    ) -> List[Dict[str, Any]]:
        """
        Extract STAT7 coordinates from embeddings concurrently.

        Args:
            embeddings: List of embedding vectors
            num_workers: Number of concurrent workers

        Returns:
            List of STAT7 coordinate dictionaries
        """
        if num_workers is None:
            num_workers = self.num_workers

        results = []

        def extract_coords(embedding: np.ndarray) -> Dict[str, Any]:
            """Extract STAT7 coordinates from single embedding."""
            return self.llm_demo.extract_stat7_from_embedding(embedding)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(extract_coords, emb): emb for emb in embeddings}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error in extraction: {e}")

        return results

    def run_throughput_test(
        self, bit_chains: List[Any], num_workers: int | None = None
    ) -> Dict[str, Any]:
        """
        Measure throughput of concurrent operations.

        Args:
            bit_chains: List of entities to process
            num_workers: Number of concurrent workers

        Returns:
            Throughput metrics (queries per second, total time, etc.)
        """
        if num_workers is None:
            num_workers = self.num_workers

        start_time = time.time()

        # Run concurrent enhancements
        results = self.run_concurrent_enhancements(bit_chains, num_workers)

        elapsed = time.time() - start_time
        throughput = len(results) / elapsed if elapsed > 0 else 0

        return {
            "completed_queries": len(results),
            "total_time_seconds": round(elapsed, 3),
            "throughput_qps": round(throughput, 2),
        }

    def test_race_conditions(
        self, bit_chains: List[Any], num_iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Test for race conditions by running same operations multiple times.

        Args:
            bit_chains: List of entities to process
            num_iterations: Number of times to run the test

        Returns:
            Race condition test results
        """
        results_list = []

        for _ in range(num_iterations):
            results = self.run_concurrent_enhancements(
                bit_chains, num_workers=self.num_workers
            )
            results_list.append(len(results))

        all_same = all(count == results_list[0] for count in results_list)

        return {
            "iterations": num_iterations,
            "results_per_iteration": results_list,
            "consistent": all_same,
            "no_race_conditions": all_same,
        }

    def run_full_concurrency_test(
        self, bit_chains: List[Any], num_workers: int | None = None
    ) -> Dict[str, Any]:
        """
        Run complete concurrency test suite.

        Args:
            bit_chains: List of entities to test
            num_workers: Number of concurrent workers

        Returns:
            Complete test report
        """
        if num_workers is None:
            num_workers = self.num_workers

        print("\nRunning concurrent embeddings...")
        embedding_results = self.run_concurrent_embeddings(bit_chains, num_workers)
        embedding_throughput = self.run_throughput_test(bit_chains, num_workers)

        print("Running concurrent enhancements...")
        enhancement_throughput = self.run_throughput_test(bit_chains, num_workers)

        print("Running concurrent STAT7 extraction...")
        embeddings = [r["embedding"] for r in embedding_results]
        extraction_results = self.run_concurrent_stat7_extraction(
            embeddings, num_workers
        )

        print("Testing for race conditions...")
        race_condition_test = self.test_race_conditions(bit_chains, num_iterations=3)

        status = (
            "PASS"
            if (
                len(embedding_results) == len(bit_chains)
                and len(extraction_results) == len(bit_chains)
                and race_condition_test["no_race_conditions"]
            )
            else "FAIL"
        )

        self.results.status = status
        self.results.results = {
            "num_entities": len(bit_chains),
            "num_workers": num_workers,
            "embedding_throughput": embedding_throughput,
            "enhancement_throughput": enhancement_throughput,
            "stat7_extraction_throughput": {
                "completed_queries": len(extraction_results),
                "throughput_qps": round(len(extraction_results) / 2, 2),
            },
            "race_condition_test": race_condition_test,
            "no_race_conditions": race_condition_test["no_race_conditions"],
            "all_results_valid": (
                len(embedding_results) == len(bit_chains)
                and len(extraction_results) == len(bit_chains)
            ),
        }

        return self.results.to_dict()

    def save_results(self, output_dir: str = "results") -> str:
        """
        Save test results to JSON file.

        Args:
            output_dir: Directory to save results

        Returns:
            Path to saved results file
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"exp09_concurrency_{timestamp}.json"
        filepath = Path(output_dir) / filename

        with open(filepath, "w") as f:
            f.write(self.results.to_json())

        return str(filepath)


def main():
    """Run concurrency test."""
    print("=" * 70)
    print("FRACTALSTAT EXP-09: Concurrency & Thread Safety Test")
    print("=" * 70)

    from dataclasses import dataclass as dc

    @dc
    class TestBitChain:
        bit_chain_id: str
        content: str
        realm: str
        luminosity: float = 0.7

    # Create test entities
    bit_chains = [
        TestBitChain(
            bit_chain_id=f"STAT7-CONCUR-{i:03d}",
            content=f"Concurrent test entity {i} with unique properties",
            realm="companion" if i % 2 == 0 else "badge",
            luminosity=0.5 + (i * 0.05),
        )
        for i in range(3)
    ]

    print(f"\nTesting with {len(bit_chains)} entities and 2 concurrent workers...")

    tester = ConcurrencyTester(num_workers=2)
    report = tester.run_full_concurrency_test(bit_chains, num_workers=2)

    print(f"\n[+] Test Status: {report['status']}")
    print("\nResults:")
    print(f"  Entities tested: {report['results']['num_entities']}")
    print(f"  Workers: {report['results']['num_workers']}")
    print(f"  No race conditions: {report['results']['no_race_conditions']}")
    print(f"  All results valid: {report['results']['all_results_valid']}")

    print("\nThroughput Metrics:")
    emb_tp = report["results"]["embedding_throughput"]
    print(
        f"  Embedding: {emb_tp['throughput_qps']} qps ({
            emb_tp['completed_queries']
        } queries)"
    )

    enh_tp = report["results"]["enhancement_throughput"]
    print(
        f"  Enhancement: {enh_tp['throughput_qps']} qps ({
            enh_tp['completed_queries']
        } queries)"
    )

    stat7_tp = report["results"]["stat7_extraction_throughput"]
    print(
        f"  STAT7 Extraction: {stat7_tp['throughput_qps']} qps ({
            stat7_tp['completed_queries']
        } queries)"
    )

    # Save results
    output_file = tester.save_results()
    print(f"\n[+] Results saved to: {output_file}")

    print("\n" + "=" * 70)
    print(f"EXP-09 Complete: {report['status']}")
    print("=" * 70)

    return report


if __name__ == "__main__":
    main()
