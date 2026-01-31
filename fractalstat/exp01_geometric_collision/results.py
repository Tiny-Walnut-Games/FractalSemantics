"""
EXP-01: Geometric Collision Resistance Test - Results Processing

Handles saving and processing of experiment results to JSON files, providing
utilities for result serialization and file management.

This module provides:
- save_results: Main function for saving experiment results
- Result validation and formatting
- File path management and directory creation
- JSON serialization with proper formatting
"""

import json
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from pathlib import Path


def save_results(results: Dict[str, Any], output_file: Optional[str] = None) -> str:
    """
    Save experiment results to JSON file with comprehensive metadata.
    
    Args:
        results: Dictionary containing experiment results and metadata
        output_file: Optional custom filename. If None, generates timestamped filename.
        
    Returns:
        Full path to the saved results file
        
    Raises:
        IOError: If file cannot be written
        TypeError: If results cannot be serialized to JSON
    """
    # Validate input
    if not isinstance(results, dict):
        raise TypeError("Results must be a dictionary")
    
    # Generate output filename if not provided
    if output_file is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_file = f"exp01_geometric_collision_{timestamp}.json"

    # Ensure results directory exists
    results_dir = Path(__file__).resolve().parent.parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / output_file

    # Add metadata if not already present
    if "metadata" not in results:
        results["metadata"] = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "file_format": "EXP-01 Geometric Collision Resistance Results",
            "version": "1.0.0"
        }

    try:
        # Write results to file with proper formatting
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            f.write("\n")  # Add trailing newline

        print(f"Results saved to: {output_path}")
        return str(output_path)

    except (IOError, OSError) as e:
        raise IOError(f"Failed to save results to {output_path}: {e}")
    except (TypeError, ValueError) as e:
        raise TypeError(f"Results cannot be serialized to JSON: {e}")


def validate_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and enhance experiment results with additional metadata.
    
    Args:
        results: Raw experiment results dictionary
        
    Returns:
        Validated and enhanced results dictionary
    """
    if not isinstance(results, dict):
        raise ValueError("Results must be a dictionary")
    
    # Add validation metadata
    validation_info = {
        "validation_timestamp": datetime.now(timezone.utc).isoformat(),
        "validation_version": "1.0.0",
        "required_fields": [
            "experiment_metadata",
            "geometric_validation", 
            "coordinate_spaces",
            "results"
        ],
        "validation_passed": True,
        "validation_errors": []
    }

    # Check for required fields
    for field in validation_info["required_fields"]:
        if field not in results:
            validation_info["validation_passed"] = False
            validation_info["validation_errors"].append(f"Missing required field: {field}")

    # Add result statistics if results are present
    if "results" in results and isinstance(results["results"], list):
        result_stats = {
            "total_dimensions_tested": len(results["results"]),
            "total_collisions": sum(r.get("collisions", 0) for r in results["results"]),
            "average_collision_rate": sum(r.get("collision_rate", 0) for r in results["results"]) / len(results["results"]) if results["results"] else 0,
            "highest_dimension": max(r.get("dimension", 0) for r in results["results"]) if results["results"] else 0,
            "lowest_dimension": min(r.get("dimension", 0) for r in results["results"]) if results["results"] else 0
        }
        results["result_statistics"] = result_stats

    results["validation"] = validation_info
    return results


def load_results(file_path: str) -> Dict[str, Any]:
    """
    Load experiment results from a JSON file.
    
    Args:
        file_path: Path to the results JSON file
        
    Returns:
        Dictionary containing loaded results
        
    Raises:
        FileNotFoundError: If the file does not exist
        IOError: If the file cannot be read
        json.JSONDecodeError: If the file contains invalid JSON
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {file_path}")
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            results = json.load(f)
        
        # Validate loaded results
        return validate_results(results)
        
    except (IOError, OSError) as e:
        raise IOError(f"Failed to read results from {file_path}: {e}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in results file {file_path}: {e}")


def create_summary_report(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a concise summary report from detailed experiment results.
    
    Args:
        results: Detailed experiment results dictionary
        
    Returns:
        Summary report dictionary with key metrics
    """
    summary = {
        "experiment_summary": {
            "name": results.get("experiment_metadata", {}).get("experiment_name", "EXP-01"),
            "sample_size": results.get("experiment_metadata", {}).get("sample_size", "Unknown"),
            "dimensions_tested": len(results.get("results", [])),
            "timestamp": results.get("experiment_metadata", {}).get("timestamp", "Unknown")
        }
    }

    # Add geometric validation summary
    geometric_validation = results.get("geometric_validation", {})
    summary["geometric_validation"] = {
        "low_dimension_collision_rate": f"{geometric_validation.get('low_dimensions_avg_collision_rate', 0) * 100:.2f}%",
        "high_dimension_collision_rate": f"{geometric_validation.get('high_dimensions_avg_collision_rate', 0) * 100:.2f}%",
        "geometric_transition_confirmed": geometric_validation.get("geometric_transition_confirmed", False),
        "improvement_factor": f"{geometric_validation.get('geometric_improvement_factor', 0):.0f}x"
    }

    # Add collision statistics
    if "results" in results:
        total_collisions = sum(r.get("collisions", 0) for r in results["results"])
        total_unique = sum(r.get("unique_coordinates", 0) for r in results["results"])
        summary["collision_statistics"] = {
            "total_collisions": total_collisions,
            "total_unique_coordinates": total_unique,
            "collision_rate_range": {
                "min": f"{min(r.get('collision_rate', 0) * 100 for r in results['results']) if results['results'] else 0:.2f}%",
                "max": f"{max(r.get('collision_rate', 0) * 100 for r in results['results']) if results['results'] else 0:.2f}%"
            }
        }

    # Add validation status
    validation = results.get("validation", {})
    summary["validation_status"] = {
        "passed": validation.get("validation_passed", False),
        "errors": validation.get("validation_errors", []),
        "timestamp": validation.get("validation_timestamp", "Unknown")
    }

    return summary


def export_summary_to_file(results: Dict[str, Any], output_file: Optional[str] = None) -> str:
    """
    Export a summary report to a separate JSON file.
    
    Args:
        results: Detailed experiment results
        output_file: Optional custom filename for summary
        
    Returns:
        Path to the saved summary file
    """
    summary = create_summary_report(results)
    
    if output_file is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_file = f"exp01_summary_{timestamp}.json"
    
    return save_results(summary, output_file)