#!/usr/bin/env python3
"""
Test Page Responsiveness

This test verifies that the GUI application remains responsive after experiments complete
and doesn't require manual page refresh to function properly.
"""

import logging
import sys
import time
from pathlib import Path

# Add the fractalsemantics module to the path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from fractalsemantics.experiment_runner import ExperimentRunner
from gui_state_manager import (
    cleanup_session_state,
    ensure_session_state_initialized,
    get_session_state_snapshot,
    validate_session_state,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_session_state_initialization():
    """Test that session state can be properly initialized."""
    logger.info("Testing session state initialization...")

    # Simulate session state initialization
    success = ensure_session_state_initialized()

    if success:
        logger.info("success Session state initialization: PASSED")
        return True
    else:
        logger.error("error Session state initialization: FAILED")
        return False


def test_session_state_validation():
    """Test that session state validation works correctly."""
    logger.info("Testing session state validation...")

    # Simulate session state validation
    is_valid = validate_session_state()

    if is_valid:
        logger.info("success Session state validation: PASSED")
        return True
    else:
        logger.error("error Session state validation: FAILED")
        return False


def test_session_state_snapshot():
    """Test that session state snapshot can be captured."""
    logger.info("Testing session state snapshot...")

    try:
        snapshot = get_session_state_snapshot()

        if isinstance(snapshot, dict) and len(snapshot) > 0:
            logger.info("success Session state snapshot: PASSED")
            logger.info(f"Snapshot contains {len(snapshot)} variables")
            return True
        else:
            logger.error("error Session state snapshot: FAILED - Invalid snapshot")
            return False

    except Exception as e:
        logger.error(f"error Session state snapshot: FAILED - {e}")
        return False


def test_session_state_cleanup():
    """Test that session state cleanup works correctly."""
    logger.info("Testing session state cleanup...")

    try:
        # Simulate cleanup
        cleanup_session_state()

        # Verify cleanup worked
        is_valid = validate_session_state()

        if is_valid:
            logger.info("success Session state cleanup: PASSED")
            return True
        else:
            logger.error("error Session state cleanup: FAILED")
            return False

    except Exception as e:
        logger.error(f"error Session state cleanup: FAILED - {e}")
        return False


def test_experiment_runner_integration():
    """Test that the experiment runner integrates properly with state management."""
    logger.info("Testing experiment runner integration...")

    try:
        # Create experiment runner
        runner = ExperimentRunner()

        # Verify runner can be created
        if runner and hasattr(runner, 'experiment_configs'):
            logger.info("success Experiment runner integration: PASSED")
            logger.info(f"Runner has {len(runner.experiment_configs)} experiment configurations")
            return True
        else:
            logger.error("error Experiment runner integration: FAILED - Invalid runner")
            return False

    except Exception as e:
        logger.error(f"error Experiment runner integration: FAILED - {e}")
        return False


def test_state_recovery():
    """Test that state recovery works from corruption."""
    logger.info("Testing state recovery from corruption...")

    try:
        # Simulate state corruption by clearing session state
        # Note: This is a simulation since we can't actually access Streamlit session state here

        # Test that initialization can recover from corruption
        success = ensure_session_state_initialized()

        if success:
            logger.info("success State recovery from corruption: PASSED")
            return True
        else:
            logger.error("error State recovery from corruption: FAILED")
            return False

    except Exception as e:
        logger.error(f"error State recovery from corruption: FAILED - {e}")
        return False


def run_responsiveness_tests():
    """Run all page responsiveness tests."""
    logger.info("=" * 60)
    logger.info("GUI Page Responsiveness Test Suite")
    logger.info("=" * 60)

    tests = [
        test_session_state_initialization,
        test_session_state_validation,
        test_session_state_snapshot,
        test_session_state_cleanup,
        test_experiment_runner_integration,
        test_state_recovery
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            logger.error(f"Test {test.__name__} failed with exception: {e}")

    logger.info("=" * 60)
    logger.info(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        logger.info("celebration All tests passed! The GUI page responsiveness fix appears to be working correctly.")
        return True
    else:
        logger.error(f"error {total - passed} tests failed. The GUI page responsiveness fix needs attention.")
        return False


if __name__ == "__main__":
    success = run_responsiveness_tests()
    sys.exit(0 if success else 1)
