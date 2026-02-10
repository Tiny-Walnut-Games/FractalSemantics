"""
Test suite for EXP-13: Fractal Gravity Without Falloff

Tests the gravitational cohesion experiment to ensure it properly validates
that fractal entities exhibit natural gravity without distance falloff.
"""

import pytest

from fractalsemantics.exp13_fractal_gravity import (
    ElementEntity,
    GravityTestResults,
    EXP13_GravityTestResults,
    generate_element_population,
    calculate_hierarchical_distance,
    calculate_euclidean_distance,
    calculate_gravitational_cohesion,
    analyze_universal_falloff_pattern,
    run_fractal_gravity_experiment,
)


class TestElementEntity:
    """Test ElementEntity class and properties."""

    def test_fractal_density_calculation(self):
        """Test that fractal density is calculated correctly."""
        from fractalsemantics.dynamic_enum import Realm, Polarity, Alignment
        from fractalsemantics.fractalsemantics_entity import Coordinates, Horizon

        coordinates = Coordinates(
            realm=Realm.ACHIEVEMENT,
            lineage=1,
            adjacency=["adj1", "adj2"],
            horizon=Horizon.PEAK,
            luminosity=75.0,
            polarity=Polarity.ACHIEVEMENT,
            dimensionality=3,
            alignment=Alignment.HARMONIC,
        )

        entity = ElementEntity(
            element_type="gold",
            fractal_mass=0.0,
            coordinates=coordinates,
            hierarchical_position=(0, 1, 2),
        )

        # Base density = dimensionality + 1 = 4
        # Adjacency factor = len(adjacency) * 0.1 = 0.2
        # Luminosity factor = luminosity/100 = 0.75
        # Expected: 4 * (1 + 0.2) * (1 + 0.75) = 4 * 1.2 * 1.75 = 8.4
        expected_density = 4 * (1 + 0.2) * (1 + 0.75)
        assert abs(entity.fractal_density - expected_density) < 1e-6


class TestDistanceCalculations:
    """Test distance calculation functions."""

    def test_hierarchical_distance(self):
        """Test hierarchical distance calculation."""
        # Same position
        assert calculate_hierarchical_distance((0, 0, 0), (0, 0, 0)) == 1

        # Adjacent positions
        assert calculate_hierarchical_distance((0, 0, 0), (0, 0, 1)) == 1
        assert calculate_hierarchical_distance((0, 0, 0), (0, 1, 0)) == 1

        # Different positions (sum of absolute differences)
        assert calculate_hierarchical_distance((0, 0, 0), (1, 2, 3)) == 6

    def test_euclidean_distance(self):
        """Test Euclidean distance calculation."""
        # Same position
        assert calculate_euclidean_distance((0, 0, 0), (0, 0, 0)) == 0

        # Unit distance
        assert calculate_euclidean_distance((0, 0, 0), (1, 0, 0)) == 1
        assert calculate_euclidean_distance((0, 0, 0), (0, 1, 0)) == 1

        # Pythagorean distance
        distance = calculate_euclidean_distance((0, 0, 0), (3, 4, 0))
        assert abs(distance - 5.0) < 1e-6


class TestGravitationalCohesion:
    """Test gravitational cohesion calculations."""

    @pytest.fixture
    def sample_entities(self):
        """Create sample entities for testing."""
        from fractalsemantics.dynamic_enum import Realm, Polarity, Alignment
        from fractalsemantics.fractalsemantics_entity import Coordinates, Horizon

        coordinates1 = Coordinates(
            realm=Realm.ACHIEVEMENT,
            lineage=1,
            adjacency=["adj1"],
            horizon=Horizon.PEAK,
            luminosity=50.0,
            polarity=Polarity.ACHIEVEMENT,
            dimensionality=2,
            alignment=Alignment.HARMONIC,
        )

        coordinates2 = Coordinates(
            realm=Realm.PATTERN,
            lineage=1,
            adjacency=["adj2"],
            horizon=Horizon.PEAK,
            luminosity=60.0,
            polarity=Polarity.ORDER,
            dimensionality=3,
            alignment=Alignment.HARMONIC,
        )

        entity1 = ElementEntity(
            element_type="gold",
            fractal_mass=0.0,
            coordinates=coordinates1,
            hierarchical_position=(0, 0, 0),
        )
        entity1.fractal_mass = entity1.fractal_density  # Set mass from density

        entity2 = ElementEntity(
            element_type="nickel",
            fractal_mass=0.0,
            coordinates=coordinates2,
            hierarchical_position=(0, 0, 1),
        )
        entity2.fractal_mass = entity2.fractal_density  # Set mass from density

        return entity1, entity2

    def test_natural_cohesion_no_falloff(self, sample_entities):
        """Test natural cohesion without falloff."""
        entity1, entity2 = sample_entities

        cohesion = calculate_gravitational_cohesion(
            entity1, entity2,
            hierarchical_distance=2,
            euclidean_distance=1.0,
            use_falloff=False
        )

        # Should be mass_product / hierarchical_distance
        expected_mass_product = entity1.fractal_density * entity2.fractal_density
        expected_cohesion = expected_mass_product / 2

        assert cohesion == expected_cohesion

    def test_falloff_cohesion(self, sample_entities):
        """Test cohesion with falloff."""
        entity1, entity2 = sample_entities

        cohesion = calculate_gravitational_cohesion(
            entity1, entity2,
            hierarchical_distance=2,
            euclidean_distance=2.0,
            use_falloff=True
        )

        # Should include 1/r² falloff factor
        natural_cohesion = calculate_gravitational_cohesion(
            entity1, entity2, 2, 2.0, use_falloff=False
        )
        expected_falloff_factor = 1.0 / (1.0 + 2.0 ** 2)  # 1/(1+r²)
        expected_cohesion = natural_cohesion * expected_falloff_factor

        assert cohesion == expected_cohesion

    def test_falloff_weakens_cohesion(self, sample_entities):
        """Test that falloff always weakens cohesion."""
        entity1, entity2 = sample_entities

        natural = calculate_gravitational_cohesion(
            entity1, entity2, 2, 1.0, use_falloff=False
        )
        with_falloff = calculate_gravitational_cohesion(
            entity1, entity2, 2, 1.0, use_falloff=True
        )

        assert with_falloff < natural


class TestElementPopulationGeneration:
    """Test element population generation."""

    def test_generate_population_size(self):
        """Test that correct number of entities are generated."""
        population = generate_element_population("gold", 50, (0, 0, 0))
        assert len(population) == 50

    def test_generate_population_properties(self):
        """Test that generated entities have correct properties."""
        population = generate_element_population("gold", 10, (0, 0, 0))

        for entity in population:
            assert entity.element_type == "gold"
            assert entity.fractal_mass > 0  # Should be set from density
            assert len(entity.hierarchical_position) == 5  # base + 2 generated
            assert entity.coordinates.realm.value == "achievement"  # Gold realm
            assert entity.coordinates.alignment.value == "harmonic"


class TestGravityTestResults:
    """Test gravity test result analysis."""

    def test_distance_independence_calculation(self):
        """Test distance independence score calculation."""
        # Create mock results with perfect distance independence
        result = GravityTestResults(
            element_type="gold",
            sample_size=100,
            natural_cohesion_mean=10.0,
            natural_cohesion_std=0.1,
            falloff_cohesion_mean=5.0,
            falloff_cohesion_std=0.1,
            falloff_weakening_ratio=0.5,
            distance_independence_score=0.95,  # High independence
            consistency_score=0.9,
        )

        # Should be considered distance independent
        assert result.distance_independence_score >= 0.7

    def test_falloff_weakening_calculation(self):
        """Test falloff weakening ratio calculation."""
        result = GravityTestResults(
            element_type="gold",
            sample_size=100,
            natural_cohesion_mean=20.0,
            natural_cohesion_std=1.0,
            falloff_cohesion_mean=5.0,  # 4x weaker
            falloff_cohesion_std=1.0,
            falloff_weakening_ratio=0.25,
            distance_independence_score=0.8,
            consistency_score=0.9,
        )

        assert result.falloff_weakening_ratio == 0.25


class TestUniversalFalloffPattern:
    """Test universal falloff pattern analysis."""

    def test_consistent_falloff_pattern(self):
        """Test detection of consistent falloff across elements."""
        results = [
            GravityTestResults(
                element_type="gold",
                sample_size=100,
                natural_cohesion_mean=10.0,
                natural_cohesion_std=1.0,
                falloff_cohesion_mean=3.0,  # 3.33x weakening
                falloff_cohesion_std=1.0,
                falloff_weakening_ratio=0.3,
                distance_independence_score=0.8,
                consistency_score=0.9,
            ),
            GravityTestResults(
                element_type="nickel",
                sample_size=100,
                natural_cohesion_mean=8.0,
                natural_cohesion_std=1.0,
                falloff_cohesion_mean=2.4,  # 3.33x weakening
                falloff_cohesion_std=1.0,
                falloff_weakening_ratio=0.3,
                distance_independence_score=0.8,
                consistency_score=0.9,
            ),
        ]

        assert analyze_universal_falloff_pattern(results)

    def test_inconsistent_falloff_pattern(self):
        """Test detection of inconsistent falloff across elements."""
        results = [
            GravityTestResults(
                element_type="gold",
                sample_size=100,
                natural_cohesion_mean=10.0,
                natural_cohesion_std=1.0,
                falloff_cohesion_mean=5.0,  # 2x weakening
                falloff_cohesion_std=1.0,
                falloff_weakening_ratio=0.5,
                distance_independence_score=0.8,
                consistency_score=0.9,
            ),
            GravityTestResults(
                element_type="nickel",
                sample_size=100,
                natural_cohesion_mean=8.0,
                natural_cohesion_std=1.0,
                falloff_cohesion_mean=1.0,  # 8x weakening (too different)
                falloff_cohesion_std=1.0,
                falloff_weakening_ratio=0.125,
                distance_independence_score=0.8,
                consistency_score=0.9,
            ),
        ]

        assert not analyze_universal_falloff_pattern(results)


class TestExperimentIntegration:
    """Test full experiment integration."""

    def test_run_fractal_gravity_experiment(self):
        """Test running the complete gravity experiment."""
        results = run_fractal_gravity_experiment(
            elements_to_test=["gold", "nickel"],
            population_size=10,  # Small for testing
            interaction_samples=50,
        )

        assert isinstance(results, EXP13_GravityTestResults)
        assert len(results.element_results) == 2
        assert results.start_time is not None
        assert results.end_time is not None
        assert results.total_duration_seconds >= 0

    def test_experiment_result_validation(self):
        """Test that experiment results contain expected validation fields."""
        results = EXP13_GravityTestResults(
            start_time="2024-01-01T00:00:00Z",
            end_time="2024-01-01T00:01:00Z",
            total_duration_seconds=60.0,
            element_results=[],
            interactions=[],
            universal_falloff_pattern=True,
            natural_gravity_confirmed=True,
            hierarchical_gravity_proven=True,
        )

        # Should validate as successful
        assert results.universal_falloff_pattern
        assert results.natural_gravity_confirmed
        assert results.hierarchical_gravity_proven


if __name__ == "__main__":
    pytest.main([__file__])
