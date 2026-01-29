# EXP-13 Redesign: Fixing the Fractal Gravity Test
## A Step-by-Step Walkthrough

---

## The Core Problem: Mixing Two Coordinate Systems

**Current issue**: Your test conflates two fundamentally different ways of measuring distance:

1. **Euclidean Space** (spatial coordinates: x, y, z)
   - What we see: objects at different positions in 3D space
   - Current test: measures cohesion across Euclidean distance
   - Result: natural cohesion shows distance dependence (bad for hypothesis)

2. **Hierarchical Space** (fractal tree structure: parent → children → grandchildren)
   - What we should measure: how many levels of hierarchy separate two entities
   - Your hypothesis predicts: cohesion depends on hierarchical depth, NOT spatial position
   - Current test: DOESN'T isolate this

**The fix**: Build two completely separate tests that don't confuse these spaces.

---

## Part 1: Build a Pure Hierarchical Data Structure

### Step 1A: Define a Fractal Tree Without Coordinates

Instead of:
```python
class Particle:
    element: str
    position: (x, y, z)  # ← This ties you to spatial thinking
    fractal_depth: int
    cohesion: float
```

Create:
```python
class FractalNode:
    element: str
    hierarchical_depth: int  # ← How deep in the tree (1 = root, 2 = child, 3 = grandchild, etc.)
    position_in_tree: tuple  # ← Address in tree (e.g., (0, 2, 1) = grandchild of second child of root)
    # NO spatial coordinates yet—pure hierarchy
```

### Step 1B: Build a Hierarchical Tree Structure

Create a proper tree:

```python
class FractalHierarchy:
    def __init__(self, element_type, max_depth):
        self.element = element_type
        self.root = FractalNode(depth=0, tree_address=[])
        self.nodes_by_depth = {0: [self.root]}
        
        # Build tree: each node has children
        for depth in range(1, max_depth):
            self.nodes_by_depth[depth] = []
            parent_count = len(self.nodes_by_depth[depth - 1])
            
            for parent_idx in range(parent_count):
                # Each parent has ~3-4 children (or your chosen branching factor)
                children_per_parent = 3  # Configurable
                
                for child_idx in range(children_per_parent):
                    child_address = [parent_idx, child_idx]
                    child = FractalNode(
                        depth=depth,
                        tree_address=child_address,
                        element=element_type
                    )
                    self.nodes_by_depth[depth].append(child)
    
    def get_hierarchical_distance(self, node_a, node_b):
        """
        Find closest common ancestor, measure hops to it.
        Example: if A is at depth 3 and B is at depth 3, but they share
        a parent at depth 2, the hierarchical distance is 2 (A→parent→B)
        """
        # Find lowest common ancestor
        depth_a = len(node_a.tree_address)
        depth_b = len(node_b.tree_address)
        
        # Trace back to common root
        common_depth = 0
        for i in range(min(depth_a, depth_b)):
            if node_a.tree_address[i] == node_b.tree_address[i]:
                common_depth = i + 1
            else:
                break
        
        # Distance = hops up + hops down
        distance = (depth_a - common_depth) + (depth_b - common_depth)
        return distance
```

### Step 1C: Key Insight

**Hierarchical distance is now completely independent of spatial position.** Two nodes at the same depth but different positions in the tree could be:
- Adjacent in hierarchy (siblings = distance 2)
- Far in hierarchy (cousins = distance 4)
- Very far (unrelated branches = distance 6+)

---

## Part 2: Define Cohesion as Hierarchical Function

### Step 2A: Cohesion Without Falloff (Your Hypothesis)

In a true fractal, cohesion should depend ONLY on hierarchical relationship:

```python
def compute_natural_cohesion(node_a, node_b, hierarchy):
    """
    Cohesion based purely on hierarchy, no falloff.
    Hypothesis: This should be CONSTANT regardless of distance.
    """
    h_distance = hierarchy.get_hierarchical_distance(node_a, node_b)
    
    # Natural cohesion: inversely proportional to hierarchical depth
    # Deeper in hierarchy = less direct cohesion (further from root)
    base_cohesion = 1.0 / (1.0 + h_distance)
    
    # This value should be stable across all distance measurements
    # because we're measuring topological distance, not spatial distance
    return base_cohesion
```

### Step 2B: Cohesion With Falloff (Test the Hypothesis)

Now apply falloff—but to hierarchical distance, not spatial distance:

```python
def compute_falloff_cohesion(node_a, node_b, hierarchy, falloff_exponent=2.0):
    """
    Cohesion with falloff applied to HIERARCHICAL distance.
    This tests: does inverse-square-like law apply to hierarchy?
    """
    h_distance = hierarchy.get_hierarchical_distance(node_a, node_b)
    
    # Apply falloff to hierarchical distance (not spatial)
    base_cohesion = 1.0 / (1.0 + h_distance)
    falloff_factor = 1.0 / ((h_distance + 1) ** falloff_exponent)
    
    falloff_cohesion = base_cohesion * falloff_factor
    return falloff_cohesion
```

**Critical difference**: You're now measuring falloff's effect on hierarchical relationships, not on spatial distance.

---

## Part 3: Run the Redesigned Experiment

### Step 3A: Test Setup

```python
def run_redesigned_fractal_experiment(
    elements_to_test,
    max_hierarchy_depth=5,
    interaction_samples=5000
):
    """
    Test whether cohesion follows hierarchy or spatial distance.
    """
    results = {}
    
    for element in elements_to_test:
        # Build a pure hierarchical structure (NO spatial coordinates)
        hierarchy = FractalHierarchy(element, max_depth=max_hierarchy_depth)
        
        # Collect all nodes
        all_nodes = []
        for depth_nodes in hierarchy.nodes_by_depth.values():
            all_nodes.extend(depth_nodes)
        
        # Test: measure cohesion across ALL pairs of nodes
        cohesion_by_distance = {}  # key=hierarchical_distance, value=[cohesion_values]
        
        for _ in range(interaction_samples):
            # Randomly pick two nodes
            node_a = random.choice(all_nodes)
            node_b = random.choice(all_nodes)
            
            if node_a == node_b:
                continue
            
            h_distance = hierarchy.get_hierarchical_distance(node_a, node_b)
            
            # Measure natural cohesion
            natural = compute_natural_cohesion(node_a, node_b, hierarchy)
            
            # Measure falloff cohesion
            falloff = compute_falloff_cohesion(node_a, node_b, hierarchy)
            
            # Store by hierarchical distance
            if h_distance not in cohesion_by_distance:
                cohesion_by_distance[h_distance] = {
                    'natural': [],
                    'falloff': []
                }
            
            cohesion_by_distance[h_distance]['natural'].append(natural)
            cohesion_by_distance[h_distance]['falloff'].append(falloff)
        
        results[element] = cohesion_by_distance
    
    return results
```

### Step 3B: Analyze the Results

```python
def analyze_redesigned_results(results):
    """
    Key questions to answer:
    
    1. NATURAL COHESION: Does natural cohesion stay CONSTANT across hierarchical distances?
       → If yes: "Fractals have no falloff" is CONFIRMED
       → If no: Something else is going on
    
    2. FALLOFF COHESION: Do all elements show the same falloff pattern?
       → If yes: "Universal falloff mechanism" is CONFIRMED
       → If no: Elements have different fractal properties
    
    3. ELEMENT VARIATION: Do heavier elements (Au vs Fe) show different cohesion patterns?
       → If yes: "Mass = fractal density" might be confirmed
    """
    
    analysis = {}
    
    for element, cohesion_by_distance in results.items():
        element_analysis = {
            'distances_tested': sorted(cohesion_by_distance.keys()),
            'natural_cohesion_by_distance': {},
            'falloff_cohesion_by_distance': {},
            'variance_by_distance': {}
        }
        
        for h_dist in sorted(cohesion_by_distance.keys()):
            natural_vals = cohesion_by_distance[h_dist]['natural']
            falloff_vals = cohesion_by_distance[h_dist]['falloff']
            
            element_analysis['natural_cohesion_by_distance'][h_dist] = {
                'mean': np.mean(natural_vals),
                'std': np.std(natural_vals),
                'min': np.min(natural_vals),
                'max': np.max(natural_vals)
            }
            
            element_analysis['falloff_cohesion_by_distance'][h_dist] = {
                'mean': np.mean(falloff_vals),
                'std': np.std(falloff_vals),
                'min': np.min(falloff_vals),
                'max': np.max(falloff_vals)
            }
        
        analysis[element] = element_analysis
    
    return analysis
```

---

## Part 4: What to Look For

### Hypothesis Confirmation Indicators

**If natural cohesion is CONSTANT across all hierarchical distances:**
```
Hierarchical Distance 1: mean = 0.500, std = 0.001
Hierarchical Distance 2: mean = 0.500, std = 0.001
Hierarchical Distance 3: mean = 0.500, std = 0.001
```
→ **"FRACTALS HAVE NO FALLOFF" = TRUE**

(Note: Not exactly constant due to noise, but variance should be LOW relative to mean)

**If all elements show similar falloff patterns:**
```
Gold falloff pattern:    1/distance^2.0
Nickel falloff pattern:  1/distance^2.0
Iron falloff pattern:    1/distance^2.0
```
→ **"UNIVERSAL FALLOFF MECHANISM" = TRUE**

**If heavier elements have different natural cohesion magnitudes:**
```
Gold natural cohesion:    mean = 0.45
Nickel natural cohesion:  mean = 0.42
Iron natural cohesion:    mean = 0.38
```
→ **"MASS = FRACTAL DENSITY" might be TRUE** (needs verification against atomic data)

---

## Part 5: What to Expect

### Likely Outcome if Hypothesis is Correct

Your data should look like:

```json
{
  "gold": {
    "distances_tested": [1, 2, 3, 4, 5],
    "natural_cohesion_by_distance": {
      "1": {"mean": 0.500, "std": 0.001},
      "2": {"mean": 0.500, "std": 0.002},
      "3": {"mean": 0.500, "std": 0.001},
      "4": {"mean": 0.500, "std": 0.002},
      "5": {"mean": 0.500, "std": 0.001}
    },
    "falloff_cohesion_by_distance": {
      "1": {"mean": 0.250},
      "2": {"mean": 0.063},
      "3": {"mean": 0.028},
      "4": {"mean": 0.016},
      "5": {"mean": 0.010}
    }
  }
}
```

**What this means:**
- Natural cohesion stays flat = Fractals have no falloff ✓
- Falloff follows inverse-square = Universal mechanism works ✓
- Same pattern for all elements = Consistent across element types ✓

---

## Summary: The Three Critical Changes

| Aspect | Old Test | New Test |
|--------|----------|----------|
| **Structure** | Spatial particles with fractal_depth property | Pure fractal tree, NO spatial coords |
| **Distance Metric** | Euclidean (x,y,z) | Hierarchical (lowest common ancestor) |
| **Cohesion Function** | Depended on spatial distance | Depends only on hierarchical distance |
| **Expected Result** | Distance-dependent natural cohesion | Distance-independent natural cohesion |

The redesign **isolates the hierarchical property** from spatial property, allowing you to test your core hypothesis cleanly.

---

## Code Template to Start

```python
import random
import numpy as np

class FractalNode:
    def __init__(self, depth, tree_address, element):
        self.depth = depth
        self.tree_address = tree_address  # e.g., [0, 2, 1]
        self.element = element
    
    def __repr__(self):
        return f"Node({self.element}, depth={self.depth}, addr={self.tree_address})"

class FractalHierarchy:
    def __init__(self, element_type, max_depth, branching_factor=3):
        self.element = element_type
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.nodes_by_depth = {0: [FractalNode(0, [], element_type)]}
        
        # Build tree
        for depth in range(1, max_depth):
            self.nodes_by_depth[depth] = []
            for parent_idx, parent in enumerate(self.nodes_by_depth[depth - 1]):
                for child_idx in range(branching_factor):
                    child_addr = parent.tree_address + [parent_idx, child_idx]
                    child = FractalNode(depth, child_addr, element_type)
                    self.nodes_by_depth[depth].append(child)
    
    def get_all_nodes(self):
        all_nodes = []
        for nodes in self.nodes_by_depth.values():
            all_nodes.extend(nodes)
        return all_nodes
    
    def get_hierarchical_distance(self, node_a, node_b):
        """Calculate hierarchical distance as hops through tree"""
        addr_a = node_a.tree_address
        addr_b = node_b.tree_address
        
        # Find common ancestor depth
        common_depth = 0
        for i in range(min(len(addr_a), len(addr_b))):
            if addr_a[i] == addr_b[i]:
                common_depth = i + 1
            else:
                break
        
        # Distance = hops up to ancestor + hops down
        return (len(addr_a) - common_depth) + (len(addr_b) - common_depth)

# Ready to implement compute_natural_cohesion and compute_falloff_cohesion!
```

This gives you a clean, testable framework.
