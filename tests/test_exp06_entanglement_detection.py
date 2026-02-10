"""
Test suite for EXP-06: Entanglement Detection
Tests semantic entanglement scoring between bit-chains.
"""

import pytest


class TestPolarityVector:
    """Test polarity vector computation."""

    def test_compute_polarity_vector_returns_list(self):
        """compute_polarity_vector should return 7-element list."""
        from fractalsemantics.exp06_entanglement_detection import (
            compute_polarity_vector,
        )

        bitchain = {
            "coordinates": {
                "realm": "data",
                "lineage": 5,
                "adjacency": ["a", "b"],
                "horizon": "genesis",
                "resonance": 0.5,
                "velocity": 0.3,
                "density": 0.7,
            }
        }

        vector = compute_polarity_vector(bitchain)
        assert isinstance(vector, list)
        assert len(vector) == 7

    def test_polarity_vector_normalization(self):
        """Polarity vector components should be in valid range (some may be [-1,1])."""
        from fractalsemantics.exp06_entanglement_detection import (
            compute_polarity_vector,
        )

        bitchain = {
            "coordinates": {
                "realm": "narrative",
                "lineage": 50,
                "adjacency": [],
                "horizon": "peak",
                "resonance": 0.5,
                "velocity": -0.5,
                "density": 0.5,
            }
        }

        vector = compute_polarity_vector(bitchain)
        # Some components can be in [-1, 1] range (like velocity), others in [0, 1]
        for component in vector:
            assert -1.0 <= component <= 1.0, f"Component {component} out of allowed range [-1.0, 1.0]"


class TestCosineSimilarity:
    """Test cosine similarity computation."""

    def test_cosine_similarity_identical_vectors(self):
        """Cosine similarity of identical vectors should be 1.0."""
        from fractalsemantics.exp06_entanglement_detection import cosine_similarity

        vec = [1.0, 0.0, 0.0, 0.5, 0.5]
        similarity = cosine_similarity(vec, vec)
        assert abs(similarity - 1.0) < 0.001

    def test_cosine_similarity_orthogonal_vectors(self):
        """Cosine similarity of orthogonal vectors should be 0.0."""
        from fractalsemantics.exp06_entanglement_detection import cosine_similarity

        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        similarity = cosine_similarity(vec1, vec2)
        assert abs(similarity - 0.0) < 0.001

    def test_cosine_similarity_opposite_vectors(self):
        """Cosine similarity of opposite vectors should be -1.0."""
        from fractalsemantics.exp06_entanglement_detection import cosine_similarity

        vec1 = [1.0, 1.0, 1.0]
        vec2 = [-1.0, -1.0, -1.0]
        similarity = cosine_similarity(vec1, vec2)
        assert abs(similarity - (-1.0)) < 0.001

    def test_cosine_similarity_length_mismatch_raises(self):
        """Cosine similarity should raise on vector length mismatch."""
        from fractalsemantics.exp06_entanglement_detection import cosine_similarity

        with pytest.raises(ValueError):
            cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])


class TestPolarityResonance:
    """Test polarity resonance scoring."""

    def test_polarity_resonance_returns_float(self):
        """polarity_resonance should return a float score."""
        from fractalsemantics.exp06_entanglement_detection import polarity_resonance

        bc1 = {
            "coordinates": {
                "realm": "data",
                "lineage": 1,
                "adjacency": [],
                "horizon": "genesis",
                "resonance": 0.5,
                "velocity": 0.5,
                "density": 0.5,
            }
        }

        bc2 = {
            "coordinates": {
                "realm": "data",
                "lineage": 2,
                "adjacency": [],
                "horizon": "emergence",
                "resonance": 0.5,
                "velocity": 0.5,
                "density": 0.5,
            }
        }

        score = polarity_resonance(bc1, bc2)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_polarity_resonance_identical_chains(self):
        """Polarity resonance of identical chains should be high."""
        from fractalsemantics.exp06_entanglement_detection import polarity_resonance

        bc = {
            "coordinates": {
                "realm": "narrative",
                "lineage": 5,
                "adjacency": ["x", "y"],
                "horizon": "peak",
                "resonance": 0.7,
                "velocity": 0.3,
                "density": 0.6,
            }
        }

        score = polarity_resonance(bc, bc)
        assert score > 0.8


class TestRealmAffinity:
    """Test realm affinity scoring."""

    def test_realm_affinity_same_realm(self):
        """Realm affinity of same realm should be 1.0."""
        from fractalsemantics.exp06_entanglement_detection import realm_affinity

        bc1 = {"coordinates": {"realm": "data"}}
        bc2 = {"coordinates": {"realm": "data"}}

        score = realm_affinity(bc1, bc2)
        assert score == 1.0

    def test_realm_affinity_adjacent_realms(self):
        """Realm affinity of adjacent realms should be 0.7."""
        from fractalsemantics.exp06_entanglement_detection import realm_affinity

        bc1 = {"coordinates": {"realm": "data"}}
        bc2 = {"coordinates": {"realm": "narrative"}}

        score = realm_affinity(bc1, bc2)
        assert score == 0.7

    def test_realm_affinity_orthogonal_realms(self):
        """Realm affinity of non-adjacent realms should be 0.0."""
        from fractalsemantics.exp06_entanglement_detection import realm_affinity

        bc1 = {"coordinates": {"realm": "system"}}
        bc2 = {"coordinates": {"realm": "void"}}

        score = realm_affinity(bc1, bc2)
        assert score == 0.0


class TestAdjacencyOverlap:
    """Test adjacency overlap (Jaccard similarity)."""

    def test_jaccard_similarity_identical_sets(self):
        """Jaccard similarity of identical sets should be 1.0."""
        from fractalsemantics.exp06_entanglement_detection import jaccard_similarity

        set1 = {"a", "b", "c"}
        set2 = {"a", "b", "c"}

        score = jaccard_similarity(set1, set2)
        assert score == 1.0

    def test_jaccard_similarity_disjoint_sets(self):
        """Jaccard similarity of disjoint sets should be 0.0."""
        from fractalsemantics.exp06_entanglement_detection import jaccard_similarity

        set1 = {"a", "b"}
        set2 = {"c", "d"}

        score = jaccard_similarity(set1, set2)
        assert score == 0.0

    def test_jaccard_similarity_both_empty(self):
        """Jaccard similarity of empty sets should be 1.0."""
        from fractalsemantics.exp06_entanglement_detection import jaccard_similarity

        score = jaccard_similarity(set(), set())
        assert score == 1.0

    def test_adjacency_overlap_returns_score(self):
        """adjacency_overlap should return normalized score."""
        from fractalsemantics.exp06_entanglement_detection import adjacency_overlap

        bc1 = {"coordinates": {"adjacency": ["n1", "n2", "n3"]}}
        bc2 = {"coordinates": {"adjacency": ["n2", "n3", "n4"]}}

        score = adjacency_overlap(bc1, bc2)
        assert 0.0 <= score <= 1.0


class TestLuminosityProximity:
    """Test luminosity proximity scoring."""

    def test_luminosity_proximity_identical_density(self):
        """Luminosity proximity of same density should be 1.0."""
        from fractalsemantics.exp06_entanglement_detection import (
            luminosity_proximity,
        )

        bc1 = {"coordinates": {"density": 0.5}}
        bc2 = {"coordinates": {"density": 0.5}}

        score = luminosity_proximity(bc1, bc2)
        assert score == 1.0

    def test_luminosity_proximity_opposite_density(self):
        """Luminosity proximity of opposite density should be 0.0."""
        from fractalsemantics.exp06_entanglement_detection import (
            luminosity_proximity,
        )

        bc1 = {"coordinates": {"density": 0.0}}
        bc2 = {"coordinates": {"density": 1.0}}

        score = luminosity_proximity(bc1, bc2)
        assert score == 0.0

    def test_luminosity_proximity_partial_difference(self):
        """Luminosity proximity with 0.25 difference should be 0.75."""
        from fractalsemantics.exp06_entanglement_detection import (
            luminosity_proximity,
        )

        bc1 = {"coordinates": {"density": 0.5}}
        bc2 = {"coordinates": {"density": 0.75}}

        score = luminosity_proximity(bc1, bc2)
        assert abs(score - 0.75) < 0.001


class TestLineageAffinity:
    """Test lineage affinity scoring."""

    def test_lineage_affinity_same_lineage(self):
        """Lineage affinity of same generation should be 1.0."""
        from fractalsemantics.exp06_entanglement_detection import lineage_affinity

        bc1 = {"coordinates": {"lineage": 5}}
        bc2 = {"coordinates": {"lineage": 5}}

        score = lineage_affinity(bc1, bc2)
        assert abs(score - 1.0) < 0.001

    def test_lineage_affinity_distance_one(self):
        """Lineage affinity with distance 1 should be 0.9."""
        from fractalsemantics.exp06_entanglement_detection import lineage_affinity

        bc1 = {"coordinates": {"lineage": 5}}
        bc2 = {"coordinates": {"lineage": 6}}

        score = lineage_affinity(bc1, bc2)
        assert abs(score - 0.9) < 0.001

    def test_lineage_affinity_exponential_decay(self):
        """Lineage affinity should decay exponentially with distance."""
        from fractalsemantics.exp06_entanglement_detection import lineage_affinity

        bc1 = {"coordinates": {"lineage": 1}}

        score_dist_1 = lineage_affinity(bc1, {"coordinates": {"lineage": 2}})
        score_dist_2 = lineage_affinity(bc1, {"coordinates": {"lineage": 3}})

        assert score_dist_2 < score_dist_1


class TestEntanglementScore:
    """Test complete entanglement score computation."""

    def test_entanglement_score_dataclass(self):
        """EntanglementScore should store component breakdown."""
        from fractalsemantics.exp06_entanglement_detection import EntanglementScore

        score = EntanglementScore(
            bitchain1_id="bc1",
            bitchain2_id="bc2",
            total_score=0.75,
            polarity_resonance=0.8,
            realm_affinity=1.0,
            adjacency_overlap=0.6,
            luminosity_proximity=0.9,
            lineage_affinity=0.5,
        )

        assert score.bitchain1_id == "bc1"
        assert score.total_score == 0.75

    def test_entanglement_score_to_dict(self):
        """EntanglementScore should serialize to dict."""
        from fractalsemantics.exp06_entanglement_detection import EntanglementScore

        score = EntanglementScore(
            bitchain1_id="bc1",
            bitchain2_id="bc2",
            total_score=0.75,
            polarity_resonance=0.8,
            realm_affinity=1.0,
            adjacency_overlap=0.6,
            luminosity_proximity=0.9,
            lineage_affinity=0.5,
        )

        score_dict = score.to_dict()
        assert isinstance(score_dict, dict)
        assert "components" in score_dict
        assert score_dict["total_score"] == 0.75

    def test_compute_entanglement_score_returns_valid_score(self):
        """compute_entanglement_score should return EntanglementScore with valid range."""
        from fractalsemantics.exp06_entanglement_detection import (
            compute_entanglement_score,
        )

        bc1 = {
            "id": "bc1",
            "coordinates": {
                "realm": "data",
                "lineage": 1,
                "adjacency": [],
                "horizon": "genesis",
                "resonance": 0.5,
                "velocity": 0.5,
                "density": 0.5,
            },
        }

        bc2 = {
            "id": "bc2",
            "coordinates": {
                "realm": "data",
                "lineage": 2,
                "adjacency": [],
                "horizon": "emergence",
                "resonance": 0.5,
                "velocity": 0.5,
                "density": 0.5,
            },
        }

        score = compute_entanglement_score(bc1, bc2)
        assert isinstance(score.total_score, float)
        assert 0.0 <= score.total_score <= 1.0

    def test_entanglement_score_symmetry(self):
        """Entanglement score should be symmetric: E(B1,B2) = E(B2,B1)."""
        from fractalsemantics.exp06_entanglement_detection import (
            compute_entanglement_score,
        )

        bc1 = {
            "id": "bc1",
            "coordinates": {
                "realm": "narrative",
                "lineage": 3,
                "adjacency": ["x"],
                "horizon": "peak",
                "resonance": 0.7,
                "velocity": 0.3,
                "density": 0.6,
            },
        }

        bc2 = {
            "id": "bc2",
            "coordinates": {
                "realm": "event",
                "lineage": 4,
                "adjacency": ["y", "z"],
                "horizon": "decay",
                "resonance": 0.4,
                "velocity": 0.6,
                "density": 0.4,
            },
        }

        score_12 = compute_entanglement_score(bc1, bc2)
        score_21 = compute_entanglement_score(bc2, bc1)

        assert abs(score_12.total_score - score_21.total_score) < 0.001
