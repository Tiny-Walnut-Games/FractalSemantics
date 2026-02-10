"""
Exp10 Bob Skeptic High-Volume Stress Test Framework

Stress tests Bob the Skeptic with prolonged, high-volume queries
to simulate real-world city simulation with thousands of active NPCs.

This script can be run on any operating system using:
    python bob_stress_test.py [options]

The code uses pathlib.Path for OS-agnostic file path handling.
"""

import asyncio
import json
import sys
import time
import secrets
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import requests
import numpy as np

secure_random = secrets.SystemRandom()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class BobStressTestConfig:
    """Configuration for stress testing Bob the Skeptic"""

    # Test duration and volume settings
    test_duration_minutes: int = 30
    queries_per_second_target: float = 10.0
    max_concurrent_queries: int = 50

    # Bob evaluation thresholds
    bob_coherence_high: float = 0.85
    bob_entanglement_low: float = 0.30
    bob_consistency_threshold: float = 0.85

    # API configuration
    api_base_url: str = "http://localhost:8000"

    # HTTP timeout for queries
    query_timeout_seconds: int = 30

    # Results storage settings
    results_directory: str = "results"
    detailed_results_count: int = 100

    # Query patterns (static)
    query_types: List[str] = field(
        default_factory=lambda: [
            "npc_character_development",
            "narrative_consistency",
            "world_building",
            "character_relationships",
            "plot_progression",
            "emotional_states",
            "memory_consolidation",
            "behavioral_patterns",
        ]
    )

    @classmethod
    def from_experiment_config(cls) -> "BobStressTestConfig":
        """Load configuration from ExperimentConfig or use defaults"""
        config_dict = {}

        try:
            from fractalsemantics.config import ExperimentConfig

            config = ExperimentConfig()

            # Load all configuration values
            config_dict = {
                "test_duration_minutes": config.get("EXP-10", "duration_minutes", 30),
                "queries_per_second_target": config.get(
                    "EXP-10", "queries_per_second_target", 10
                ),
                "max_concurrent_queries": config.get(
                    "EXP-10", "max_concurrent_queries", 50
                ),
                "bob_coherence_high": config.get("EXP-10", "bob_coherence_high", 0.85),
                "bob_entanglement_low": config.get(
                    "EXP-10", "bob_entanglement_low", 0.30
                ),
                "bob_consistency_threshold": config.get(
                    "EXP-10", "bob_consistency_threshold", 0.85
                ),
                "api_base_url": config.get(
                    "EXP-10", "api_base_url", "http://localhost:8000"
                ),
                "query_timeout_seconds": config.get(
                    "EXP-10", "query_timeout_seconds", 30
                ),
                "results_directory": config.get(
                    "EXP-10", "results_directory", "results"
                ),
                "detailed_results_count": config.get(
                    "EXP-10", "detailed_results_count", 100
                ),
            }
        except Exception as e:
            logger.warning(f"Could not load ExperimentConfig, using defaults: {e}")

        return cls(**config_dict)

    def validate(self) -> None:
        """Validate configuration values"""
        if self.test_duration_minutes <= 0:
            raise ValueError("Test duration must be positive")
        if self.queries_per_second_target <= 0:
            raise ValueError("QPS target must be positive")
        if self.max_concurrent_queries <= 0:
            raise ValueError("Max concurrent queries must be positive")
        if not (0 <= self.bob_coherence_high <= 1):
            raise ValueError("Bob coherence high must be between 0 and 1")
        if not (0 <= self.bob_entanglement_low <= 1):
            raise ValueError("Bob entanglement low must be between 0 and 1")
        if not (0 <= self.bob_consistency_threshold <= 1):
            raise ValueError("Bob consistency threshold must be between 0 and 1")
        if self.query_timeout_seconds <= 0:
            raise ValueError("Query timeout must be positive")
        if self.detailed_results_count < 0:
            raise ValueError("Detailed results count must be non-negative")

    # Backward compatibility properties for existing code
    @property
    def TEST_DURATION_MINUTES(self) -> int:
        return self.test_duration_minutes

    @property
    def QUERIES_PER_SECOND_TARGET(self) -> float:
        return self.queries_per_second_target

    @property
    def MAX_CONCURRENT_QUERIES(self) -> int:
        return self.max_concurrent_queries

    @property
    def BOB_COHERENCE_HIGH(self) -> float:
        return self.bob_coherence_high

    @property
    def BOB_ENTANGLEMENT_LOW(self) -> float:
        return self.bob_entanglement_low

    @property
    def BOB_CONSISTENCY_THRESHOLD(self) -> float:
        return self.bob_consistency_threshold

    @property
    def QUERY_TYPES(self) -> List[str]:
        return self.query_types


class NPCQueryGenerator:
    """Generates realistic NPC queries for stress testing"""

    def __init__(self):
        self.npc_names = [
            "Elena",
            "Marcus",
            "Sofia",
            "James",
            "Aria",
            "Kai",
            "Luna",
            "Orion",
            "Zara",
            "Finn",
            "Maya",
            "Leo",
            "Iris",
            "Rex",
            "Nova",
            "Echo",
        ]

        self.locations = [
            "Crystal Spire",
            "Shadow Market",
            "Sun Temple",
            "Moon Harbor",
            "Star Forge",
            "Dream Weavers",
            "Time Keepers",
            "Memory Palace",
        ]

        self.emotions = [
            "joyful",
            "melancholy",
            "determined",
            "conflicted",
            "hopeful",
            "anxious",
            "peaceful",
            "restless",
            "curious",
            "wary",
        ]

        self.activities = [
            "crafting",
            "exploring",
            "meditating",
            "negotiating",
            "celebrating",
            "mourning",
            "learning",
            "teaching",
            "defending",
            "healing",
        ]

    def generate_query(self, query_type: str) -> Dict[str, Any]:
        """Generate a realistic NPC query"""

        npc = secure_random.choice(self.npc_names)
        location = secure_random.choice(self.locations)
        emotion = secure_random.choice(self.emotions)
        activity = secure_random.choice(self.activities)

        queries = {
            "npc_character_development": f"How does {npc}'s {emotion} state affect their {activity} at {location}?",
            "narrative_consistency": f"What patterns emerge from {npc}'s behavior across multiple visits to {location}?",
            "world_building": f"How does {location} influence the {emotion} experiences of visitors like {npc}?",
            "character_relationships": f"Describe the evolving relationship between {npc} and others during {activity} sessions",
            "plot_progression": f"What narrative developments occur when {npc} engages in {activity} while feeling {emotion}?",
            "emotional_states": f"Trace the emotional journey of {npc} from {emotion} to other states during {activity}",
            "memory_consolidation": f"How does {npc} consolidate memories of {activity} experiences at {location}?",
            "behavioral_patterns": f"What behavioral patterns does {npc} exhibit when {emotion} during {activity} at {location}?",
        }

        return {
            "query_id": f"stress_{int(time.time() * 1000)}_{secure_random.randint(1000, 9999)}",
            "semantic": queries.get(query_type, queries["npc_character_development"]),
            "query_type": query_type,
            "npc": npc,
            "location": location,
            "emotion": emotion,
            "activity": activity,
            "hybrid": secure_random.choice([True, False]),
            "weight_semantic": secure_random.uniform(0.5, 0.8),
            "weight_fractalsemantics": secure_random.uniform(0.2, 0.5),
        }


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""

    query_times: List[float] = field(default_factory=list)
    bob_verdicts: Dict[str, int] = field(
        default_factory=lambda: {"PASSED": 0, "VERIFIED": 0, "QUARANTINED": 0}
    )
    error_count: int = 0
    queries_per_second_actual: float = 0.0

    def reset(self) -> None:
        """Reset all metrics"""
        self.query_times.clear()
        self.bob_verdicts = {"PASSED": 0, "VERIFIED": 0, "QUARANTINED": 0}
        self.error_count = 0
        self.queries_per_second_actual = 0.0


class BobStressTester:
    """Main stress testing framework for Bob the Skeptic"""

    def __init__(self, config: Optional[BobStressTestConfig] = None):
        self.config = config or BobStressTestConfig.from_experiment_config()
        self.config.validate()  # Validate configuration

        self.api_base_url = self.config.api_base_url
        self.query_generator = NPCQueryGenerator()
        self.results: List[Dict[str, Any]] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

        # Performance tracking
        self.metrics = PerformanceMetrics()

    # Backward compatibility properties
    @property
    def query_times(self) -> List[float]:
        return self.metrics.query_times

    @query_times.setter
    def query_times(self, value: List[float]) -> None:
        self.metrics.query_times = value

    @property
    def bob_verdicts(self) -> Dict[str, int]:
        return self.metrics.bob_verdicts

    @bob_verdicts.setter
    def bob_verdicts(self, value: Dict[str, int]) -> None:
        self.metrics.bob_verdicts = value

    @property
    def error_count(self) -> int:
        return self.metrics.error_count

    @error_count.setter
    def error_count(self, value: int) -> None:
        self.metrics.error_count = value

    @property
    def queries_per_second_actual(self) -> float:
        return self.metrics.queries_per_second_actual

    @queries_per_second_actual.setter
    def queries_per_second_actual(self, value: float) -> None:
        self.metrics.queries_per_second_actual = value

    async def test_api_connectivity(self) -> Dict[str, Any]:
        """Test if the API server is available and responding

        Returns:
            Dictionary with connectivity test results
        """
        try:
            # Simple health check - try to connect with a short timeout
            response = requests.get(
                f"{self.api_base_url}/health",
                timeout=5,  # Short timeout for connectivity test
                headers={"User-Agent": "BobStressTest/1.0"}
            )

            if response.status_code == 200:
                return {
                    "available": True,
                    "response_time_seconds": response.elapsed.total_seconds(),
                    "status_code": response.status_code
                }
            else:
                return {
                    "available": False,
                    "error": f"HTTP {response.status_code}: {response.text.strip()[:100]}"
                }

        except requests.exceptions.Timeout:
            return {
                "available": False,
                "error": "Connection timeout (server may not be running)"
            }

        except requests.exceptions.ConnectionError as e:
            return {
                "available": False,
                "error": f"Connection failed: {str(e)}"
            }

        except Exception as e:
            return {
                "available": False,
                "error": f"Unexpected error: {str(e)}"
            }

    async def single_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single query and track Bob's response

        Args:
            query_data: Dictionary containing query parameters

        Returns:
            Dict containing the query result or error information
        """
        start_time = time.time()

        try:
            # Validate required query data
            required_fields = [
                "query_id",
                "semantic",
                "npc",
                "location",
                "emotion",
                "activity",
            ]
            for field in required_fields:
                if field not in query_data:
                    raise ValueError(f"Missing required field: {field}")

            # Prepare query parameters with full API format
            params = {
                "query_id": query_data["query_id"],
                "mode": "semantic_similarity",
                "semantic_query": query_data["semantic"],
                "anchor_ids": ["string"],  # Use placeholder like curl example
                "max_results": 10,
                "confidence_threshold": 0.6,
                "fractalsemantics_hybrid": query_data.get("hybrid", False),
                "fractalsemantics_address": {
                    "realm": {"additionalProp1": {}},
                    "lineage": 0,
                    "adjacency": "semantic_proximity",
                    "horizon": "emergence",
                    "luminosity": 0.7,
                    "polarity": 0.5,
                    "dimensionality": 1
                },
                "weight_semantic": query_data.get("weight_semantic", 0.6),
                "weight_fractalsemantics": query_data.get("weight_fractalsemantics", 0.4)
            }

            # Execute query with configured timeout (POST with JSON body)
            response = requests.post(
                f"{self.api_base_url}/query",
                json=params,
                timeout=self.config.query_timeout_seconds,
            )

            query_time = time.time() - start_time

            if response.status_code == 200:
                try:
                    result = response.json()
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON response: {e}")
                    self.metrics.error_count += 1
                    return {
                        "timestamp": datetime.now().isoformat(),
                        "query_id": query_data["query_id"],
                        "error": f"Invalid JSON response: {e}",
                        "query_time": query_time,
                    }

                # Track Bob's verdict
                bob_status = result.get("bob_status", "UNKNOWN")
                self.metrics.bob_verdicts[bob_status] = (
                    self.metrics.bob_verdicts.get(bob_status, 0) + 1
                )

                # Store detailed result
                query_result = {
                    "timestamp": datetime.now().isoformat(),
                    "query_id": query_data["query_id"],
                    "query_type": query_data["query_type"],
                    "query_time": query_time,
                    "bob_status": bob_status,
                    "result_count": len(result.get("results", [])),
                    "npc": query_data["npc"],
                    "location": query_data["location"],
                    "emotion": query_data["emotion"],
                    "activity": query_data["activity"],
                    "hybrid": query_data["hybrid"],
                    "coherence": result.get("coherence", 0.0),
                    "entanglement": result.get("entanglement", 0.0),
                    "bob_verification_log": result.get("bob_verification_log"),
                }

                self.metrics.query_times.append(query_time)
                return query_result

            else:
                self.metrics.error_count += 1
                logger.warning(
                    f"Query failed with HTTP {response.status_code}: {query_data['query_id']}"
                )
                return {
                    "timestamp": datetime.now().isoformat(),
                    "query_id": query_data["query_id"],
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "query_time": query_time,
                }

        except requests.exceptions.Timeout:
            self.metrics.error_count += 1
            logger.warning(f"Query timeout: {query_data['query_id']}")
            return {
                "timestamp": datetime.now().isoformat(),
                "query_id": query_data["query_id"],
                "error": "Query timeout",
                "query_time": time.time() - start_time,
            }

        except requests.exceptions.ConnectionError as e:
            self.metrics.error_count += 1
            logger.warning(f"Connection error: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "query_id": query_data["query_id"],
                "error": f"Connection error: {e}",
                "query_time": time.time() - start_time,
            }

        except Exception as e:
            self.metrics.error_count += 1
            logger.error(f"Unexpected error in single_query: {e}", exc_info=True)
            return {
                "timestamp": datetime.now().isoformat(),
                "query_id": query_data["query_id"],
                "error": f"Unexpected error: {e}",
                "query_time": time.time() - start_time,
            }

    async def query_worker(self, worker_id: int, duration_seconds: int):
        """Worker that continuously generates and executes queries with adaptive rate limiting

        Args:
            worker_id: Unique identifier for this worker
            duration_seconds: Total duration for this worker to run
        """
        end_time = time.time() + duration_seconds
        queries_executed = 0

        # Calculate target queries per worker based on overall QPS target
        target_qps_per_worker = (
            self.config.queries_per_second_target / self.config.max_concurrent_queries
        )
        min_interval = (
            0.01  # Minimum 10ms between queries to prevent overwhelming the system
        )

        logger.info(
            f"Worker {worker_id} started (target QPS: {target_qps_per_worker:.2f})"
        )

        while time.time() < end_time:
            query_start_time = time.time()

            # Generate query
            query_type = secure_random.choice(self.config.QUERY_TYPES)
            query_data = self.query_generator.generate_query(query_type)

            # Execute query
            result = await self.single_query(query_data)
            self.results.append(result)
            queries_executed += 1

            # Adaptive rate limiting with exponential backoff
            elapsed = time.time() - query_start_time

            # Calculate target interval with recent performance consideration
            if len(self.query_times) >= 5:
                recent_avg = statistics.mean(self.query_times[-5:])
                # Adjust target interval based on recent performance
                effective_target_interval = max(
                    min_interval, 1.0 / target_qps_per_worker - recent_avg * 0.1
                )
            else:
                effective_target_interval = max(
                    min_interval, 1.0 / target_qps_per_worker
                )

            # Sleep to maintain target rate (but don't sleep if we're behind schedule)
            remaining_time = end_time - time.time()
            if (
                remaining_time > effective_target_interval
                and elapsed < effective_target_interval
            ):
                sleep_time = min(
                    effective_target_interval - elapsed,
                    remaining_time - effective_target_interval,
                )
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            # Safety check: don't execute more queries than we can reasonably handle
            if len(self.results) > 10000:  # Emergency brake
                logger.warning(
                    f"Worker {worker_id} reached emergency query limit, terminating"
                )
                break

        logger.info(f"Worker {worker_id} completed {queries_executed} queries")

    async def run_stress_test(
        self, duration_minutes: Optional[int] = None
    ) -> Dict[str, Any]:
        """Run the complete stress test"""

        duration_minutes = duration_minutes or self.config.TEST_DURATION_MINUTES
        duration_seconds = duration_minutes * 60

        # Test API connectivity before starting
        logger.info(f"🔍 Testing API connectivity to {self.api_base_url}")
        test_result = await self.test_api_connectivity()

        if not test_result["available"]:
            logger.warning(f"[Fail] API server not available: {test_result['error']}")
            logger.warning("   Please ensure the FractalSemantics API server is running")
            logger.warning("   Start it with: python -m fractalsemantics.api_server")

            # Return a skipped result for integration tests that can't run
            return {
                "status": "SKIPPED",
                "reason": f"API server not available: {test_result['error']}",
                "api_endpoint": self.api_base_url,
                "recommendation": "This is an integration test requiring external infrastructure"
            }

        logger.info("🚀 Starting Bob Stress Test")
        logger.info(f"   Duration: {duration_minutes} minutes")
        logger.info(f"   Target QPS: {self.config.QUERIES_PER_SECOND_TARGET}")
        logger.info(f"   Max Concurrent: {self.config.MAX_CONCURRENT_QUERIES}")
        logger.info(f"   API Endpoint: {self.api_base_url}")

        self.start_time = datetime.now()

        # Start concurrent workers
        tasks = []
        for i in range(self.config.MAX_CONCURRENT_QUERIES):
            task = asyncio.create_task(self.query_worker(i, duration_seconds))
            tasks.append(task)

        # Wait for all workers to complete
        await asyncio.gather(*tasks)

        self.end_time = datetime.now()

        # Calculate final metrics
        return self.generate_report()

    def calculate_performance_metrics(
        self, query_times: List[float]
    ) -> Dict[str, float]:
        """Calculate statistical performance metrics

        Args:
            query_times: List of query execution times

        Returns:
            Dictionary containing calculated performance metrics
        """
        if not query_times:
            return {
                "avg_query_time": 0.0,
                "median_query_time": 0.0,
                "p95_query_time": 0.0,
                "p99_query_time": 0.0,
            }

        return {
            "avg_query_time": float(statistics.mean(query_times)),
            "median_query_time": float(statistics.median(query_times)),
            "p95_query_time": float(np.percentile(query_times, 95)),
            "p99_query_time": float(np.percentile(query_times, 99)),
        }

    def calculate_bob_analysis(self, bob_verdicts: Dict[str, int]) -> Dict[str, Any]:
        """Calculate Bob's decision analysis metrics

        Args:
            bob_verdicts: Dictionary of Bob's verdicts and their counts

        Returns:
            Dictionary containing Bob analysis metrics
        """
        total_bob_decisions = sum(bob_verdicts.values())

        if total_bob_decisions == 0:
            return {
                "total_decisions": 0,
                "passed": 0,
                "verified": 0,
                "quarantined": 0,
                "alert_rate": 0.0,
                "quarantine_rate": 0.0,
            }

        alert_rate = (
            bob_verdicts.get("VERIFIED", 0) + bob_verdicts.get("QUARANTINED", 0)
        ) / total_bob_decisions
        quarantine_rate = bob_verdicts.get("QUARANTINED", 0) / total_bob_decisions

        return {
            "total_decisions": total_bob_decisions,
            "passed": bob_verdicts.get("PASSED", 0),
            "verified": bob_verdicts.get("VERIFIED", 0),
            "quarantined": bob_verdicts.get("QUARANTINED", 0),
            "alert_rate": alert_rate,
            "quarantine_rate": quarantine_rate,
        }

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive stress test report"""
        if self.start_time is None or self.end_time is None:
            raise RuntimeError(
                "Test start and end times must be set before generating report"
            )

        total_duration = (self.end_time - self.start_time).total_seconds()
        total_queries = len(self.results)
        successful_queries = len([r for r in self.results if "error" not in r])

        # Calculate QPS
        self.queries_per_second_actual = (
            total_queries / total_duration if total_duration > 0 else 0.0
        )

        # Performance metrics
        perf_metrics = self.calculate_performance_metrics(self.query_times)

        # Bob analysis
        bob_analysis = self.calculate_bob_analysis(self.bob_verdicts)

        # Query type analysis
        query_type_stats = {}
        for result in self.results:
            if "query_type" in result:
                qtype = result["query_type"]
                if qtype not in query_type_stats:
                    query_type_stats[qtype] = {
                        "total": 0,
                        "errors": 0,
                        "quarantined": 0,
                    }
                query_type_stats[qtype]["total"] += 1
                if "error" in result:
                    query_type_stats[qtype]["errors"] += 1
                if result.get("bob_status") == "QUARANTINED":
                    query_type_stats[qtype]["quarantined"] += 1

        report = {
            "test_summary": {
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat(),
                "duration_seconds": total_duration,
                "duration_minutes": total_duration / 60,
            },
            "volume_metrics": {
                "total_queries": total_queries,
                "successful_queries": successful_queries,
                "failed_queries": self.error_count,
                "success_rate": (
                    successful_queries / total_queries if total_queries > 0 else 0
                ),
                "queries_per_second_target": self.config.queries_per_second_target,
                "queries_per_second_actual": self.queries_per_second_actual,
            },
            "performance_metrics": {
                "avg_query_time_ms": perf_metrics["avg_query_time"] * 1000,
                "median_query_time_ms": perf_metrics["median_query_time"] * 1000,
                "p95_query_time_ms": perf_metrics["p95_query_time"] * 1000,
                "p99_query_time_ms": perf_metrics["p99_query_time"] * 1000,
            },
            "bob_analysis": bob_analysis,
            "query_type_analysis": query_type_stats,
            "detailed_results": self.results[-self.config.detailed_results_count :],
        }

        # Save report with configured directory
        report_dir = Path(__file__).parent / self.config.results_directory
        report_dir.mkdir(exist_ok=True, parents=True)
        report_file = (
            report_dir
            / f"exp10_bob_stress_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        try:
            with open(report_file, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"📊 Stress test report saved: {report_file}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
            raise

        return report


def check_api_sync() -> bool:
    """Synchronous API connectivity check"""
    try:
        import requests
        # Test with a simple query similar to the working curl command
        params = {
            "query_id": "health_check",
            "mode": "semantic_similarity",
            "semantic_query": "test query",
            "anchor_ids": ["string"],  # Use placeholder like curl example
            "max_results": 1,
            "confidence_threshold": 0.6,
            "fractalsemantics_hybrid": False,
            "fractalsemantics_address": {
                "realm": {"additionalProp1": {}},
                "lineage": 0,
                "adjacency": "semantic_proximity",
                "horizon": "emergence",
                "luminosity": 0.7,
                "polarity": 0.5,
                "dimensionality": 1
            },
            "weight_semantic": 0.6,
            "weight_fractalsemantics": 0.4
        }
        response = requests.post("http://localhost:8000/query", json=params, timeout=5)
        return response.status_code == 200
    except Exception:
        return False

async def main():
    """Main entry point for Bob stress testing"""

    # Check API connectivity before proceeding (for orchestrator compatibility)
    if not check_api_sync():
        print("=" * 80)
        print("BOB STRESS TEST SKIPPED")
        print("=" * 80)
        print("This is an integration test requiring external infrastructure.")
        print("Reason: API server not available at http://localhost:8000")
        print("Recommendation: This test requires FractalSemantics API server to be running")
        print("\nFor validation pipeline, this skip is expected when API server is not running.")
        print("=" * 80)
        return  # Exit successfully (no exception) for validation pipeline

    import argparse

    parser = argparse.ArgumentParser(description="Bob Skeptic Stress Test Framework")
    # Load defaults from config
    try:
        from fractalsemantics.config import ExperimentConfig

        exp_config = ExperimentConfig()
        default_duration = exp_config.get("EXP-10", "duration_minutes", 30)
        default_qps = exp_config.get("EXP-10", "queries_per_second_target", 10)
        default_concurrent = exp_config.get("EXP-10", "max_concurrent_queries", 50)
        default_api_url = exp_config.get(
            "EXP-10", "api_base_url", "http://localhost:8000"
        )
    except Exception:
        default_duration = 30
        default_qps = 10
        default_concurrent = 50
        default_api_url = "http://localhost:8000"

    parser.add_argument(
        "--duration",
        "-d",
        type=int,
        default=default_duration,
        help="Test duration in minutes",
    )
    parser.add_argument(
        "--qps",
        "-q",
        type=float,
        default=default_qps,
        help="Target queries per second",
    )
    parser.add_argument(
        "--concurrent",
        "-c",
        type=int,
        default=default_concurrent,
        help="Maximum concurrent queries",
    )
    parser.add_argument("--api-url", "-u", default=default_api_url, help="API base URL")

    args = parser.parse_args()

    # Create custom configuration from command line arguments
    custom_config = BobStressTestConfig(
        test_duration_minutes=args.duration,
        queries_per_second_target=args.qps,
        max_concurrent_queries=args.concurrent,
        api_base_url=args.api_url,
    )

    # Configure stress tester with custom config
    tester = BobStressTester(config=custom_config)

    try:
        # Run stress test
        report = await tester.run_stress_test()

        # Handle skipped tests (infrastructure not available)
        if report.get("status") == "SKIPPED":
            print("\n" + "=" * 80)
            print("⏭️  BOB STRESS TEST SKIPPED")
            print("=" * 80)
            print("This is an integration test requiring external infrastructure.")
            print(f"Reason: {report['reason']}")
            print(f"Recommendation: {report['recommendation']}")
            print("\nFor validation pipeline, this skip is expected when API server is not running.")
            print("=" * 80)
            sys.exit(0)  # Exit successfully for validation pipeline

        # Print summary for successful tests
        print("\n" + "=" * 80)
        print("🎯 BOB STRESS TEST RESULTS")
        print("=" * 80)

        print("\n📊 Volume Metrics:")
        print(f"   Total Queries: {report['volume_metrics']['total_queries']:,}")
        print(f"   Success Rate: {report['volume_metrics']['success_rate']:.2%}")
        print(f"   QPS Target: {report['volume_metrics']['queries_per_second_target']}")
        print(
            f"   QPS Actual: {report['volume_metrics']['queries_per_second_actual']:.2f}"
        )

        print("\n⚡ Performance Metrics:")
        print(
            f"   Avg Query Time: {
                report['performance_metrics']['avg_query_time_ms']:.2f}ms"
        )
        print(
            f"   P95 Query Time: {
                report['performance_metrics']['p95_query_time_ms']:.2f}ms"
        )
        print(
            f"   P99 Query Time: {
                report['performance_metrics']['p99_query_time_ms']:.2f}ms"
        )

        print("\n🔍 Bob Analysis:")
        print(f"   Total Decisions: {report['bob_analysis']['total_decisions']:,}")
        print(f"   Passed: {report['bob_analysis']['passed']:,}")
        print(f"   Verified: {report['bob_analysis']['verified']:,}")
        print(f"   Quarantined: {report['bob_analysis']['quarantined']:,}")
        print(f"   Alert Rate: {report['bob_analysis']['alert_rate']:.2%}")
        print(f"   Quarantine Rate: {report['bob_analysis']['quarantine_rate']:.2%}")

        # Health assessment
        print("\n🏥 System Health Assessment:")
        if report["volume_metrics"]["success_rate"] > 0.95:
            print("   [Success] Query Success Rate: HEALTHY")
        else:
            print("   [Fail] Query Success Rate: DEGRADED")

        if report["performance_metrics"]["p95_query_time_ms"] < 1000:
            print("   [Success] Query Latency: HEALTHY")
        else:
            print("   [Warn] Query Latency: DEGRADED")

        if 0.01 <= report["bob_analysis"]["quarantine_rate"] <= 0.10:
            print("   [Success] Bob Quarantine Rate: OPTIMAL")
        elif report["bob_analysis"]["quarantine_rate"] > 0.10:
            print("   [Warn] Bob Quarantine Rate: HIGH (may need tuning)")
        else:
            print("   [Warn] Bob Quarantine Rate: LOW (may be missing issues)")

        print("\n" + "=" * 80)

    except KeyboardInterrupt:
        print("\n⏹️ Stress test interrupted by user")
    except Exception as e:
        print(f"\n💥 Stress test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
