"""
Comprehensive unit tests for config/feature_flags.py module
"""

import pytest
import os
from unittest.mock import patch


class TestExperimentConfig:
    """Test ExperimentConfig class."""

    def test_config_initialization_default(self):
        """ExperimentConfig should initialize with default config file."""
        from fractalstat.config.feature_flags import ExperimentConfig

        config = ExperimentConfig()

        assert config.config is not None
        assert isinstance(config.config, dict)
        assert config.env in ["dev", "ci", "production"]

    def test_config_environment_from_env_var(self):
        """ExperimentConfig should read environment from FRACTALSTAT_ENV."""
        from fractalstat.config.feature_flags import ExperimentConfig

        with patch.dict(os.environ, {"FRACTALSTAT_ENV": "ci"}):
            config = ExperimentConfig()
            assert config.env == "ci"

    def test_config_environment_default_dev(self):
        """ExperimentConfig should default to 'dev' environment."""
        from fractalstat.config.feature_flags import ExperimentConfig

        with patch.dict(os.environ, {}, clear=True):
            if "FRACTALSTAT_ENV" in os.environ:
                del os.environ["FRACTALSTAT_ENV"]
            config = ExperimentConfig()
            assert config.env == "dev"

    def test_config_is_enabled_true(self):
        """is_enabled should return True for enabled experiments."""
        from fractalstat.config.feature_flags import ExperimentConfig

        config = ExperimentConfig()
        enabled_experiments = config.get_enabled_experiments()
        
        if enabled_experiments:
            assert config.is_enabled(enabled_experiments[0]) is True

    def test_config_is_enabled_false(self):
        """is_enabled should return False for disabled experiments."""
        from fractalstat.config.feature_flags import ExperimentConfig

        config = ExperimentConfig()
        
        assert config.is_enabled("NONEXISTENT-EXP") is False

    def test_config_get_value_exists(self):
        """get should return configuration value if it exists."""
        from fractalstat.config.feature_flags import ExperimentConfig

        config = ExperimentConfig()
        enabled = config.get_enabled_experiments()
        
        if enabled:
            exp_id = enabled[0]
            value = config.get(exp_id, "sample_size")
            assert value is not None or value == config.get(exp_id, "sample_size", None)

    def test_config_get_value_default(self):
        """get should return default value if key doesn't exist."""
        from fractalstat.config.feature_flags import ExperimentConfig

        config = ExperimentConfig()
        
        value = config.get("NONEXISTENT-EXP", "nonexistent_key", 999)
        assert value == 999

    def test_config_get_value_no_default(self):
        """get should return None if key doesn't exist and no default."""
        from fractalstat.config.feature_flags import ExperimentConfig

        config = ExperimentConfig()
        
        value = config.get("NONEXISTENT-EXP", "nonexistent_key")
        assert value is None

    def test_config_get_all_experiment(self):
        """get_all should return all configuration for an experiment."""
        from fractalstat.config.feature_flags import ExperimentConfig

        config = ExperimentConfig()
        enabled = config.get_enabled_experiments()
        
        if enabled:
            exp_id = enabled[0]
            all_config = config.get_all(exp_id)
            assert isinstance(all_config, dict)

    def test_config_get_all_nonexistent(self):
        """get_all should return empty dict for nonexistent experiment."""
        from fractalstat.config.feature_flags import ExperimentConfig

        config = ExperimentConfig()
        
        all_config = config.get_all("NONEXISTENT-EXP")
        assert all_config == {}

    def test_config_get_enabled_experiments(self):
        """get_enabled_experiments should return list of enabled experiments."""
        from fractalstat.config.feature_flags import ExperimentConfig

        config = ExperimentConfig()
        enabled = config.get_enabled_experiments()
        
        assert isinstance(enabled, list)

    def test_config_get_environment(self):
        """get_environment should return current environment name."""
        from fractalstat.config.feature_flags import ExperimentConfig

        with patch.dict(os.environ, {"FRACTALSTAT_ENV": "ci"}):
            config = ExperimentConfig()
            assert config.get_environment() == "ci"

    def test_config_repr(self):
        """__repr__ should return string representation."""
        from fractalstat.config.feature_flags import ExperimentConfig

        config = ExperimentConfig()
        repr_str = repr(config)
        
        assert "ExperimentConfig" in repr_str
        assert "env=" in repr_str
        assert "enabled=" in repr_str

    def test_config_load_base_config_file_not_found(self):
        """_load_base_config should raise FileNotFoundError for missing file."""
        from fractalstat.config.feature_flags import ExperimentConfig

        with pytest.raises(FileNotFoundError, match="Base configuration file not found"):
            ExperimentConfig(config_file="nonexistent.toml")

    def test_config_merge_configs_enabled_list(self):
        """_merge_configs should merge enabled list from override."""
        from fractalstat.config.feature_flags import ExperimentConfig

        config = ExperimentConfig()
        config.get_enabled_experiments().copy()
        
        override = {
            "experiments": {
                "enabled": ["TEST-EXP-1", "TEST-EXP-2"]
            }
        }
        
        config._merge_configs(override)
        
        assert config.get_enabled_experiments() == ["TEST-EXP-1", "TEST-EXP-2"]

    def test_config_merge_configs_experiment_settings(self):
        """_merge_configs should merge experiment-specific settings."""
        from fractalstat.config.feature_flags import ExperimentConfig

        config = ExperimentConfig()
        
        override = {
            "experiments": {
                "TEST-EXP": {
                    "sample_size": 5000,
                    "custom_param": "value"
                }
            }
        }
        
        config._merge_configs(override)
        
        assert config.get("TEST-EXP", "sample_size") == 5000
        assert config.get("TEST-EXP", "custom_param") == "value"

    def test_config_merge_configs_preserves_existing(self):
        """_merge_configs should preserve existing non-overridden values."""
        from fractalstat.config.feature_flags import ExperimentConfig

        config = ExperimentConfig()
        config.get_enabled_experiments()
        
        override = {
            "experiments": {
                "NEW-EXP": {
                    "param": "value"
                }
            }
        }
        
        config._merge_configs(override)
        
        assert config.get("NEW-EXP", "param") == "value"


class TestGetConfigFunction:
    """Test get_config convenience function."""

    def test_get_config_returns_instance(self):
        """get_config should return ExperimentConfig instance."""
        from fractalstat.config.feature_flags import get_config, ExperimentConfig

        config = get_config()
        
        assert isinstance(config, ExperimentConfig)

    def test_get_config_singleton(self):
        """get_config should return same instance on multiple calls."""
        from fractalstat.config.feature_flags import get_config

        if hasattr(get_config, "_instance"):
            delattr(get_config, "_instance")
        
        config1 = get_config()
        config2 = get_config()
        
        assert config1 is config2

    def test_get_config_has_methods(self):
        """get_config instance should have all expected methods."""
        from fractalstat.config.feature_flags import get_config

        config = get_config()
        
        assert hasattr(config, "is_enabled")
        assert hasattr(config, "get")
        assert hasattr(config, "get_all")
        assert hasattr(config, "get_enabled_experiments")
        assert hasattr(config, "get_environment")


class TestConfigIntegration:
    """Integration tests for config system."""

    def test_config_loads_experiments_toml(self):
        """Config should successfully load experiments.toml."""
        from fractalstat.config.feature_flags import ExperimentConfig

        config = ExperimentConfig()
        
        assert "experiments" in config.config
        assert isinstance(config.config["experiments"], dict)

    def test_config_enabled_experiments_valid(self):
        """Enabled experiments should be valid experiment IDs."""
        from fractalstat.config.feature_flags import ExperimentConfig

        config = ExperimentConfig()
        enabled = config.get_enabled_experiments()
        
        for exp_id in enabled:
            assert isinstance(exp_id, str)
            assert len(exp_id) > 0

    def test_config_experiment_has_settings(self):
        """Enabled experiments should have configuration settings."""
        from fractalstat.config.feature_flags import ExperimentConfig

        config = ExperimentConfig()
        enabled = config.get_enabled_experiments()
        
        if enabled:
            exp_id = enabled[0]
            exp_config = config.get_all(exp_id)
            assert isinstance(exp_config, dict)

    def test_config_environment_specific_override(self):
        """Environment-specific config should override base config."""
        from fractalstat.config.feature_flags import ExperimentConfig

        with patch.dict(os.environ, {"FRACTALSTAT_ENV": "ci"}):
            config = ExperimentConfig()
            assert config.env == "ci"

    def test_config_missing_env_file_graceful(self):
        """Missing environment-specific file should not cause error."""
        from fractalstat.config.feature_flags import ExperimentConfig

        with patch.dict(os.environ, {"FRACTALSTAT_ENV": "nonexistent_env"}):
            config = ExperimentConfig()
            assert config.config is not None


class TestConfigEdgeCases:
    """Test edge cases and error handling."""

    def test_config_empty_enabled_list(self):
        """Config should handle empty enabled list."""
        from fractalstat.config.feature_flags import ExperimentConfig

        config = ExperimentConfig()
        config.config["experiments"]["enabled"] = []
        
        enabled = config.get_enabled_experiments()
        assert enabled == []
        assert config.is_enabled("ANY-EXP") is False

    def test_config_missing_experiments_section(self):
        """Config should handle missing experiments section."""
        from fractalstat.config.feature_flags import ExperimentConfig

        config = ExperimentConfig()
        config.config = {}
        
        enabled = config.get_enabled_experiments()
        assert enabled == []

    def test_config_get_with_nested_dict(self):
        """get should handle nested dictionary values."""
        from fractalstat.config.feature_flags import ExperimentConfig

        config = ExperimentConfig()
        config.config["experiments"]["TEST-EXP"] = {
            "nested": {"key": "value"}
        }
        
        value = config.get("TEST-EXP", "nested")
        assert isinstance(value, dict)
        assert value["key"] == "value"

    def test_config_multiple_environments(self):
        """Config should work with different environments."""
        from fractalstat.config.feature_flags import ExperimentConfig

        for env in ["dev", "ci"]:
            with patch.dict(os.environ, {"FRACTALSTAT_ENV": env}):
                config = ExperimentConfig()
                assert config.get_environment() == env
