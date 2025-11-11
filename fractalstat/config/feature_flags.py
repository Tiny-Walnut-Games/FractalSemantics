"""
FractalStat Feature Flags and Experiment Configuration

Provides centralized configuration management for all experiments using TOML files.
Supports environment-specific overrides (dev, ci, production).

Usage:
    from fractalstat.config import ExperimentConfig
    
    config = ExperimentConfig()
    
    if config.is_enabled("EXP-01"):
        sample_size = config.get("EXP-01", "sample_size", 1000)
        iterations = config.get("EXP-01", "iterations", 10)
        # Run experiment...
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Handle Python 3.11+ vs earlier versions for tomllib/tomli
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        raise ImportError(
            "tomli is required for Python < 3.11. Install with: pip install tomli"
        )


class ExperimentConfig:
    """
    Centralized experiment configuration manager.
    
    Loads configuration from TOML files with environment-specific overrides.
    
    Configuration precedence (highest to lowest):
    1. Environment-specific config (experiments.{env}.toml)
    2. Base config (experiments.toml)
    
    Environment is determined by FRACTALSTAT_ENV environment variable.
    Defaults to 'dev' if not set.
    """

    def __init__(self, config_file: str = "experiments.toml"):
        """
        Initialize experiment configuration.
        
        Args:
            config_file: Base configuration file name (default: experiments.toml)
        """
        self.config_dir = Path(__file__).parent
        self.config: Dict[str, Any] = {}
        self.env = os.getenv("FRACTALSTAT_ENV", "dev")
        
        # Load base configuration
        self._load_base_config(config_file)
        
        # Load environment-specific overrides
        self._load_env_overrides()

    def _load_base_config(self, config_file: str) -> None:
        """Load the base configuration file."""
        config_path = self.config_dir / config_file
        
        if not config_path.exists():
            raise FileNotFoundError(
                f"Base configuration file not found: {config_path}"
            )
        
        with open(config_path, "rb") as f:
            self.config = tomllib.load(f)

    def _load_env_overrides(self) -> None:
        """Load environment-specific configuration overrides."""
        env_config_path = self.config_dir / f"experiments.{self.env}.toml"
        
        if not env_config_path.exists():
            # No environment-specific config, use base config only
            return
        
        with open(env_config_path, "rb") as f:
            env_config = tomllib.load(f)
        
        # Merge environment config into base config
        self._merge_configs(env_config)

    def _merge_configs(self, override_config: Dict[str, Any]) -> None:
        """
        Merge override configuration into base configuration.
        
        Args:
            override_config: Configuration to merge (takes precedence)
        """
        # Merge top-level experiments section
        if "experiments" in override_config:
            if "experiments" not in self.config:
                self.config["experiments"] = {}
            
            # Merge enabled list if present
            if "enabled" in override_config["experiments"]:
                self.config["experiments"]["enabled"] = override_config["experiments"]["enabled"]
            
            # Merge individual experiment configs
            for key, value in override_config["experiments"].items():
                if key == "enabled":
                    continue  # Already handled
                
                if isinstance(value, dict):
                    # Merge experiment-specific settings
                    if key not in self.config["experiments"]:
                        self.config["experiments"][key] = {}
                    self.config["experiments"][key].update(value)
                else:
                    # Direct override
                    self.config["experiments"][key] = value

    def is_enabled(self, experiment: str) -> bool:
        """
        Check if an experiment is enabled.
        
        Args:
            experiment: Experiment ID (e.g., "EXP-01")
        
        Returns:
            True if experiment is in the enabled list, False otherwise
        """
        enabled_list = self.config.get("experiments", {}).get("enabled", [])
        return experiment in enabled_list

    def get(self, experiment: str, key: str, default: Any = None) -> Any:
        """
        Get a configuration value for an experiment.
        
        Args:
            experiment: Experiment ID (e.g., "EXP-01")
            key: Configuration key (e.g., "sample_size")
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        """
        exp_config = self.config.get("experiments", {}).get(experiment, {})
        return exp_config.get(key, default)

    def get_all(self, experiment: str) -> Dict[str, Any]:
        """
        Get all configuration for an experiment.
        
        Args:
            experiment: Experiment ID (e.g., "EXP-01")
        
        Returns:
            Dictionary of all configuration values for the experiment
        """
        return self.config.get("experiments", {}).get(experiment, {})

    def get_enabled_experiments(self) -> List[str]:
        """
        Get list of all enabled experiments.
        
        Returns:
            List of enabled experiment IDs
        """
        return self.config.get("experiments", {}).get("enabled", [])

    def get_environment(self) -> str:
        """
        Get the current environment name.
        
        Returns:
            Environment name (e.g., "dev", "ci", "production")
        """
        return self.env

    def __repr__(self) -> str:
        """String representation of configuration."""
        enabled = self.get_enabled_experiments()
        return f"ExperimentConfig(env='{self.env}', enabled={enabled})"


# Convenience function for quick access
def get_config() -> ExperimentConfig:
    """
    Get a singleton instance of ExperimentConfig.
    
    Returns:
        ExperimentConfig instance
    """
    if not hasattr(get_config, "_instance"):
        get_config._instance = ExperimentConfig()
    return get_config._instance
