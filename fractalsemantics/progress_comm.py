#!/usr/bin/env python3
"""
Progress Communication Module for FractalSemantics Experiments

This module provides a lightweight, non-destructive communication protocol
that allows experiments to send progress updates to the main experiment runner
in real-time. The communication uses JSON lines over stderr to separate
progress messages from experiment output.

Key Features:
- Non-destructive: Experiments work independently without any changes
- Real-time progress updates with structured JSON messages
- Thread-safe progress reporting
- Graceful fallback when communication fails
- Educational content support in progress messages

Usage:
    from fractalsemantics.progress_comm import ProgressReporter

    # Initialize progress reporter
    progress = ProgressReporter(experiment_id="EXP-01")

    # Report progress at key stages
    progress.update(25, "Generating bit-chains...")
    progress.update(50, "Computing coordinates...")
    progress.update(75, "Validating collisions...")
    progress.complete("Analysis complete!")
"""

import ast
import builtins
import contextlib
import json
import os
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class ProgressMessage:
    """Structured progress message for experiment communication."""

    timestamp: str
    experiment_id: str
    progress_percent: float
    stage: str
    message: str
    metadata: Optional[dict[str, any]] = None
    message_type: str = "progress"  # progress, status, warning, error, complete

    def to_json(self) -> str:
        """Convert to JSON string for transmission."""
        return json.dumps(asdict(self), ensure_ascii=False)


class ProgressReporter:
    """
    Progress reporter for experiments to communicate with the main runner.

    This class provides a simple API for experiments to report their progress
    in real-time. Messages are sent as JSON lines over stderr, which allows
    the main runner to parse progress updates while preserving stdout for
    experiment output.

    The reporter is thread-safe and handles communication failures gracefully.
    If progress reporting fails (e.g., when running experiments independently),
    the experiment continues normally without any errors.
    """

    def __init__(self, experiment_id: str, enabled: bool = True):
        """
        Initialize progress reporter.

        Args:
            experiment_id: Identifier for the experiment (e.g., "EXP-01")
            enabled: Whether progress reporting is enabled
        """
        self.experiment_id = experiment_id
        self.enabled = enabled
        self._lock = threading.Lock()
        self._last_message_time = 0.0
        self._min_message_interval = 0.1  # Minimum 100ms between messages

        # Check for progress file path from environment
        self._progress_file = None
        progress_file_env = os.environ.get("FRACTALSEMANTICS_PROGRESS_FILE")
        if progress_file_env:
            self._progress_file = Path(progress_file_env)

    def _should_send_message(self) -> bool:
        """Check if enough time has passed since last message."""
        current_time = time.time()
        if current_time - self._last_message_time >= self._min_message_interval:
            self._last_message_time = current_time
            return True
        return False

    def _send_message(self, message: ProgressMessage) -> bool:
        """
        Send progress message to stderr and optionally to file.

        Returns:
            True if message was sent successfully, False otherwise
        """
        if not self.enabled:
            return False

        success = False

        try:
            # Use a special marker to identify progress messages
            json_line = f"__PROGRESS__:{message.to_json()}\n"

            with self._lock:
                # Write directly to stderr buffer to avoid buffering issues
                sys.stderr.write(json_line)
                sys.stderr.flush()

            success = True

        except (OSError, BrokenPipeError):
            # Communication failed (e.g., stderr closed, pipe broken)
            # This is expected when experiments run independently
            pass
        except Exception:
            # Other unexpected errors - log but don't crash the experiment
            try:
                with self._lock:
                    sys.stderr.write("__PROGRESS_ERROR__: Failed to send progress message\n")
                    sys.stderr.flush()
            except ast.ParseError:
                pass

        # Also write to file if configured
        if self._progress_file:
            try:
                progress_data = {
                    "experiment_id": message.experiment_id,
                    "progress": message.progress_percent,
                    "stage": message.stage,
                    "message": message.message,
                    "timestamp": message.timestamp,
                    "message_type": message.message_type
                }
                if write_progress_to_file(self._progress_file, progress_data):
                    success = True
            except Exception:
                # Silently fail - don't crash experiment for progress reporting
                pass

        return success

    def update(self, progress_percent: float, stage: str, message: str = "",
               metadata: Optional[dict[str, any]] = None) -> bool:
        """
        Report progress update.

        Args:
            progress_percent: Progress percentage (0.0 to 100.0)
            stage: Current stage of the experiment
            message: Additional message text
            metadata: Optional metadata dictionary

        Returns:
            True if message was sent successfully, False otherwise
        """
        if not self._should_send_message():
            return False

        # Clamp progress to valid range
        progress_percent = max(0.0, min(100.0, progress_percent))

        message_obj = ProgressMessage(
            timestamp=datetime.now(timezone.utc).isoformat(),
            experiment_id=self.experiment_id,
            progress_percent=progress_percent,
            stage=stage,
            message=message,
            metadata=metadata,
            message_type="progress"
        )

        return self._send_message(message_obj)

    def status(self, stage: str, message: str, metadata: Optional[dict[str, any]] = None) -> bool:
        """
        Report a status update (non-progress message).

        Args:
            stage: Current stage of the experiment
            message: Status message
            metadata: Optional metadata dictionary

        Returns:
            True if message was sent successfully, False otherwise
        """
        if not self._should_send_message():
            return False

        message_obj = ProgressMessage(
            timestamp=datetime.now(timezone.utc).isoformat(),
            experiment_id=self.experiment_id,
            progress_percent=-1.0,  # Special value for non-progress messages
            stage=stage,
            message=message,
            metadata=metadata,
            message_type="status"
        )

        return self._send_message(message_obj)

    def warning(self, stage: str, message: str, metadata: Optional[dict[str, any]] = None) -> bool:
        """
        Report a warning message.

        Args:
            stage: Current stage of the experiment
            message: Warning message
            metadata: Optional metadata dictionary

        Returns:
            True if message was sent successfully, False otherwise
        """
        if not self._should_send_message():
            return False

        message_obj = ProgressMessage(
            timestamp=datetime.now(timezone.utc).isoformat(),
            experiment_id=self.experiment_id,
            progress_percent=-1.0,
            stage=stage,
            message=message,
            metadata=metadata,
            message_type="warning"
        )

        return self._send_message(message_obj)

    def error(self, stage: str, message: str, metadata: Optional[dict[str, any]] = None) -> bool:
        """
        Report an error message.

        Args:
            stage: Current stage of the experiment
            message: Error message
            metadata: Optional metadata dictionary

        Returns:
            True if message was sent successfully, False otherwise
        """
        if not self._should_send_message():
            return False

        message_obj = ProgressMessage(
            timestamp=datetime.now(timezone.utc).isoformat(),
            experiment_id=self.experiment_id,
            progress_percent=-1.0,
            stage=stage,
            message=message,
            metadata=metadata,
            message_type="error"
        )

        return self._send_message(message_obj)

    def complete(self, message: str = "Experiment completed",
                 metadata: Optional[dict[str, any]] = None) -> bool:
        """
        Report experiment completion.

        Args:
            message: Completion message
            metadata: Optional metadata dictionary

        Returns:
            True if message was sent successfully, False otherwise
        """
        message_obj = ProgressMessage(
            timestamp=datetime.now(timezone.utc).isoformat(),
            experiment_id=self.experiment_id,
            progress_percent=100.0,
            stage="Complete",
            message=message,
            metadata=metadata,
            message_type="complete"
        )

        return self._send_message(message_obj)

    def disable(self):
        """Disable progress reporting."""
        with self._lock:
            self.enabled = False

    def enable(self):
        """Enable progress reporting."""
        with self._lock:
            self.enabled = True


def parse_progress_message(line: str) -> Optional[ProgressMessage]:
    """
    Parse a progress message from a line of text.

    Args:
        line: Line of text that may contain a progress message

    Returns:
        ProgressMessage object if parsing successful, None otherwise
    """
    if not line.startswith("__PROGRESS__:"):
        return None

    try:
        # Extract JSON part after the marker
        json_str = line[len("__PROGRESS__:"):].strip()
        data = json.loads(json_str)

        # Validate required fields
        required_fields = ['timestamp', 'experiment_id', 'progress_percent', 'stage', 'message', 'message_type']
        for field in required_fields:
            if field not in data:
                return None

        return ProgressMessage(**data)

    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def is_progress_message(line: str) -> bool:
    """
    Check if a line contains a progress message.

    Args:
        line: Line of text to check

    Returns:
        True if line contains a progress message, False otherwise
    """
    return line.startswith("__PROGRESS__:")


def write_progress_to_file(progress_file: Path, progress_data: dict) -> bool:
    """
    Write progress data to file atomically.

    Uses temp file + rename pattern to ensure atomic writes and prevent corruption.

    Args:
        progress_file: Path to the progress file
        progress_data: Dictionary containing progress information

    Returns:
        True if write was successful, False otherwise
    """
    try:
        # Ensure directory exists
        progress_file.parent.mkdir(parents=True, exist_ok=True)

        # Write to temporary file first
        temp_fd, temp_path = tempfile.mkstemp(
            dir=progress_file.parent,
            prefix=".progress_",
            suffix=".json.tmp"
        )

        try:
            # Write JSON data to temp file
            with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())  # Ensure data is written to disk

            # Atomically rename temp file to target file
            # On Windows, need to remove target first if it exists
            if os.name == 'nt' and progress_file.exists():
                os.replace(temp_path, str(progress_file))
            else:
                os.rename(temp_path, str(progress_file))

            return True

        except Exception:
            # Clean up temp file on error
            with contextlib.suppress(builtins.BaseException):
                os.unlink(temp_path)
            raise

    except Exception:
        # Silently fail - progress reporting should not crash experiments
        return False


def read_progress_from_file(progress_file: Path) -> Optional[dict]:
    """
    Read progress data from file.

    Args:
        progress_file: Path to the progress file

    Returns:
        Dictionary containing progress data, or None if file doesn't exist or can't be read
    """
    try:
        if not progress_file.exists():
            return None

        with open(progress_file, encoding='utf-8') as f:
            return json.load(f)

    except (OSError, json.JSONDecodeError):
        return None


def clear_progress_file(progress_file: Path) -> bool:
    """
    Clear the progress file.

    Args:
        progress_file: Path to the progress file

    Returns:
        True if file was cleared, False otherwise
    """
    try:
        if progress_file.exists():
            progress_file.unlink()
        return True
    except Exception:
        return False


# Convenience functions for backward compatibility and ease of use

def create_progress_reporter(experiment_id: str, enabled: bool = True) -> ProgressReporter:
    """
    Create a progress reporter instance.

    This is the recommended way to create progress reporters in experiments.

    Args:
        experiment_id: Identifier for the experiment (e.g., "EXP-01")
        enabled: Whether progress reporting is enabled

    Returns:
        ProgressReporter instance
    """
    return ProgressReporter(experiment_id, enabled)


def report_progress(experiment_id: str, progress_percent: float, stage: str,
                   message: str = "", metadata: Optional[dict[str, any]] = None) -> bool:
    """
    Convenience function to report progress without creating a reporter instance.

    This function creates a temporary reporter and sends a single progress message.
    It's useful for simple progress reporting scenarios.

    Args:
        experiment_id: Identifier for the experiment
        progress_percent: Progress percentage (0.0 to 100.0)
        stage: Current stage of the experiment
        message: Additional message text
        metadata: Optional metadata dictionary

    Returns:
        True if message was sent successfully, False otherwise
    """
    reporter = ProgressReporter(experiment_id)
    return reporter.update(progress_percent, stage, message, metadata)


def report_status(experiment_id: str, stage: str, message: str,
                 metadata: Optional[dict[str, any]] = None) -> bool:
    """
    Convenience function to report a status message.

    Args:
        experiment_id: Identifier for the experiment
        stage: Current stage of the experiment
        message: Status message
        metadata: Optional metadata dictionary

    Returns:
        True if message was sent successfully, False otherwise
    """
    reporter = ProgressReporter(experiment_id)
    return reporter.status(stage, message, metadata)


def report_completion(experiment_id: str, message: str = "Experiment completed",
                     metadata: Optional[dict[str, any]] = None) -> bool:
    """
    Convenience function to report experiment completion.

    Args:
        experiment_id: Identifier for the experiment
        message: Completion message
        metadata: Optional metadata dictionary

    Returns:
        True if message was sent successfully, False otherwise
    """
    reporter = ProgressReporter(experiment_id)
    return reporter.complete(message, metadata)


# Example usage and testing

if __name__ == "__main__":
    # Example of how to use the progress reporter in an experiment
    progress = create_progress_reporter("EXP-01")

    print("Starting experiment...")

    # Simulate experiment stages
    progress.update(0, "Initialization", "Setting up experiment environment")
    time.sleep(0.1)

    progress.update(25, "Data Generation", "Generating random bit-chains")
    time.sleep(0.1)

    progress.update(50, "Computation", "Computing FractalSemantics coordinates")
    time.sleep(0.1)

    progress.update(75, "Validation", "Validating collision resistance")
    time.sleep(0.1)

    progress.complete("Analysis complete - zero collisions detected")

    print("Experiment finished!")
