#!/usr/bin/env python3
"""
GUI State Manager

A robust state management system for the FractalSemantics GUI application.
Provides proper state initialization, cleanup, and recovery mechanisms to prevent
page unresponsiveness after experiments complete.
"""

import logging
import threading
import time
from typing import Any, Optional

import streamlit as st

logger = logging.getLogger(__name__)


class GUIStateManager:
    """Manages Streamlit session state with robust initialization and recovery."""

    def __init__(self):
        self._lock = threading.Lock()
        self._initialized = False
        self._fallback_state = self._get_fallback_state()

    def _get_fallback_state(self) -> dict[str, Any]:
        """Get fallback state values for corrupted session state."""
        return {
            'experiment_results': [],
            'batch_result': None,
            'is_running': False,
            'current_experiment_id': None,
            'progress_data': [],
            'selected_experiments': ["EXP-01", "EXP-02", "EXP-03"]
        }

    def ensure_state_initialized(self) -> bool:
        """Ensure all required session state variables are properly initialized."""
        with self._lock:
            try:
                # Check if already initialized
                if self._is_state_valid():
                    return True

                # Initialize all required session state variables
                required_vars = self._fallback_state.keys()
                for var_name in required_vars:
                    if var_name not in st.session_state:
                        st.session_state[var_name] = self._fallback_state[var_name]
                        logger.info(f"Initialized session state variable: {var_name}")

                # Set up environment variables for subprocess communication
                self._setup_environment()

                # run the initialization method
                self._initialized = True
                logger.info("Session state initialization completed successfully")
                return True

            except Exception as e:
                logger.error(f"Session state initialization failed: {e}")
                self._recover_from_corruption()
                return False

    def _is_state_valid(self) -> bool:
        """Check if session state is valid and properly initialized."""
        try:
            # Check if all required variables exist and are of correct type
            required_vars = self._fallback_state.keys()
            for var_name in required_vars:
                if var_name not in st.session_state:
                    return False

                # Basic type validation
                expected_type = type(self._fallback_state[var_name])
                if not isinstance(st.session_state[var_name], expected_type):
                    return False

            return True
        except Exception:
            return False

    def _setup_environment(self):
        """Set up environment variables for subprocess communication."""
        import os
        from pathlib import Path

        try:
            # Create results directory if it doesn't exist
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)

            # Set progress file path
            progress_file = results_dir / "gui_progress.jsonl"
            os.environ["FRACTALSEMANTICS_PROGRESS_FILE"] = str(progress_file)

            logger.debug(f"Set progress file environment variable: {progress_file}")

        except Exception as e:
            logger.error(f"Failed to setup environment variables: {e}")

    def _recover_from_corruption(self):
        """Recover from corrupted session state by resetting to fallback values."""
        try:
            logger.warning("Recovering from corrupted session state")

            # Clear corrupted state and set fallback values
            for var_name, fallback_value in self._fallback_state.items():
                st.session_state[var_name] = fallback_value

            # Reset environment
            self._setup_environment()

            logger.info("Session state recovery completed")

        except Exception as e:
            logger.error(f"Session state recovery failed: {e}")

    def cleanup_state(self):
        """Clean up session state after experiments complete."""
        with self._lock:
            try:
                # Reset running state
                st.session_state.is_running = False

                # Clear progress data
                st.session_state.progress_data = []

                # Ensure other state variables are valid
                if not self._is_state_valid():
                    self.ensure_state_initialized()

                logger.debug("Session state cleanup completed")

            except Exception as e:
                logger.error(f"Session state cleanup failed: {e}")

    def get_state_snapshot(self) -> dict[str, Any]:
        """Get a snapshot of current session state for debugging."""
        try:
            snapshot = {}
            for key, value in st.session_state.items():
                if not key.startswith('_'):  # Skip internal Streamlit variables
                    snapshot[key] = str(value) if len(str(value)) > 100 else value
            return snapshot
        except Exception as e:
            logger.error(f"Failed to get state snapshot: {e}")
            return {}

    def validate_state_integrity(self) -> bool:
        """Validate that session state is in a consistent state."""
        try:
            # Check for common state corruption issues
            if not isinstance(st.session_state.experiment_results, list):
                logger.error("experiment_results is not a list")
                return False

            if not isinstance(st.session_state.is_running, bool):
                logger.error("is_running is not a boolean")
                return False

            if not isinstance(st.session_state.progress_data, list):
                logger.error("progress_data is not a list")
                return False

            return True

        except Exception as e:
            logger.error(f"State integrity validation failed: {e}")
            return False

    def safe_state_update(self, updates: dict[str, Any]) -> bool:
        """Safely update session state with error handling."""
        with self._lock:
            try:
                for key, value in updates.items():
                    st.session_state[key] = value

                return True

            except Exception as e:
                logger.error(f"Safe state update failed: {e}")
                return False

    def reset_to_defaults(self):
        """Reset session state to default values."""
        with self._lock:
            try:
                logger.info("Resetting session state to defaults")

                # Clear all session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]

                # Re-initialize with defaults
                self.ensure_state_initialized()

                logger.info("Session state reset completed")

            except Exception as e:
                logger.error(f"Session state reset failed: {e}")


# Global state manager instance
state_manager = GUIStateManager()


def ensure_session_state_initialized() -> bool:
    """Global function to ensure session state is properly initialized."""
    return state_manager.ensure_state_initialized()


def cleanup_session_state():
    """Global function to clean up session state."""
    state_manager.cleanup_state()


def validate_session_state() -> bool:
    """Global function to validate session state integrity."""
    return state_manager.validate_state_integrity()


def get_session_state_snapshot() -> dict[str, Any]:
    """Global function to get session state snapshot."""
    return state_manager.get_state_snapshot()


def safe_session_state_update(updates: dict[str, Any]) -> bool:
    """Global function to safely update session state."""
    return state_manager.safe_state_update(updates)


def reset_session_state():
    """Global function to reset session state to defaults."""
    state_manager.reset_to_defaults()
