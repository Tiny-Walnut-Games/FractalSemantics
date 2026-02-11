"""
Subprocess Communication Module for FractalSemantics Experiments

This module provides utilities for experiments to communicate progress and status
back to the main experiment runner via subprocess communication. It builds on the
existing progress_comm.py system but adds subprocess-specific functionality.

The communication uses:
- Environment variables to pass configuration
- JSON Lines format for structured communication
- Thread-safe file operations
- Graceful degradation when communication fails

Usage in experiments:
    from fractalsemantics.subprocess_comm import send_subprocess_progress
    
    # Send progress update
    send_subprocess_progress("EXP-01", 50.0, "Generation", "50% of bit-chains generated", "info")
"""

import os
import json
import sys
import threading
import time
from typing import Optional, Dict, Any
from pathlib import Path

# Import the existing progress communication system
from fractalsemantics.progress_comm import (
    create_progress_reporter,
    report_progress,
    report_status,
    report_completion
)


class SubprocessCommunicator:
    """
    Enhanced communicator for subprocess experiments.
    
    This class extends the basic progress communication to handle subprocess-specific
    scenarios like parent process detection, subprocess ID tracking, and enhanced
    error handling.
    """
    
    def __init__(self, experiment_id: str):
        """
        Initialize subprocess communicator.
        
        Args:
            experiment_id: The experiment identifier (e.g., "EXP-01")
        """
        self.experiment_id = experiment_id
        self.subprocess_id = os.getpid()
        self.parent_process_id = os.getppid()
        self.start_time = time.time()
        
        # Get progress file from environment (inherited from parent)
        self.progress_file = os.environ.get("FRACTALSEMANTICS_PROGRESS_FILE")
        
        # Subprocess-specific metadata
        self.metadata = {
            "subprocess_id": self.subprocess_id,
            "parent_process_id": self.parent_process_id,
            "start_time": self.start_time,
            "experiment_id": experiment_id
        }
    
    def send_progress(self, 
                     progress: float, 
                     stage: str, 
                     message: str, 
                     level: str = "info") -> bool:
        """
        Send progress update from subprocess.
        
        Args:
            progress: Progress percentage (0.0 to 100.0)
            stage: Current stage of the experiment
            message: Detailed progress message
            level: Message level (info, warning, error, success)
            
        Returns:
            True if message was sent successfully, False otherwise
        """
        try:
            # Add subprocess metadata to the message
            enhanced_message = f"[Subprocess {self.subprocess_id}] {message}"
            
            # Create progress message
            progress_msg = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%f%z"),
                "experiment_id": self.experiment_id,
                "progress_percent": progress,
                "stage": stage,
                "message": enhanced_message,
                "metadata": self.metadata,
                "message_type": "progress"
            }
            
            # Write to progress file if configured
            if self.progress_file:
                try:
                    # Ensure directory exists
                    Path(self.progress_file).parent.mkdir(parents=True, exist_ok=True)
                    
                    # Write JSON line to file
                    with open(self.progress_file, 'a') as f:
                        f.write(json.dumps(progress_msg) + '\n')
                    
                    # Also send to stderr for immediate feedback
                    print(f"__PROGRESS__:{json.dumps(progress_msg)}", file=sys.stderr)
                    
                    return True
                except Exception as e:
                    print(f"Warning: Failed to write to progress file: {e}", file=sys.stderr)
            
            # Fallback: send using the existing progress communication system
            from fractalsemantics.progress_comm import report_progress
            report_progress(
                experiment_id=self.experiment_id,
                progress_percent=progress,
                stage=stage,
                message=enhanced_message
            )
            
            return True
            
        except Exception as e:
            # Log error but don't fail the experiment
            print(f"Warning: Failed to send progress message: {e}", file=sys.stderr)
            return False
    
    def send_status(self, status: str, details: Optional[str] = None) -> bool:
        """
        Send status update from subprocess.
        
        Args:
            status: Status message (e.g., "Starting", "Completed", "Failed")
            details: Optional additional details
            
        Returns:
            True if message was sent successfully, False otherwise
        """
        message = status
        if details:
            message += f": {details}"
            
        return self.send_progress(0.0, "Status", message, "info")
    
    def send_error(self, error_message: str, exception: Optional[Exception] = None) -> bool:
        """
        Send error message from subprocess.
        
        Args:
            error_message: Error description
            exception: Optional exception object for additional context
            
        Returns:
            True if message was sent successfully, False otherwise
        """
        if exception:
            error_message += f" (Exception: {type(exception).__name__}: {str(exception)})"
            
        return self.send_progress(0.0, "Error", error_message, "error")
    
    def send_completion(self, success: bool, details: Optional[str] = None) -> bool:
        """
        Send completion message from subprocess.
        
        Args:
            success: Whether the experiment completed successfully
            details: Optional completion details
            
        Returns:
            True if message was sent successfully, False otherwise
        """
        status = "Completed" if success else "Failed"
        message = f"{status} experiment {self.experiment_id}"
        if details:
            message += f": {details}"
            
        progress = 100.0 if success else 0.0
        level = "success" if success else "error"
        
        return self.send_progress(progress, "Completion", message, level)
    
    def get_runtime_info(self) -> Dict[str, Any]:
        """
        Get runtime information for debugging.
        
        Returns:
            Dictionary with subprocess runtime information
        """
        current_time = time.time()
        runtime = current_time - self.start_time
        
        return {
            "subprocess_id": self.subprocess_id,
            "parent_process_id": self.parent_process_id,
            "start_time": self.start_time,
            "current_time": current_time,
            "runtime_seconds": runtime,
            "experiment_id": self.experiment_id,
            "progress_file": self.progress_file
        }


# Global communicator instance for each subprocess
_communicator: Optional[SubprocessCommunicator] = None
_communicator_lock = threading.Lock()


def get_subprocess_communicator(experiment_id: str) -> SubprocessCommunicator:
    """
    Get or create a subprocess communicator for the current process.
    
    Args:
        experiment_id: The experiment identifier
        
    Returns:
        SubprocessCommunicator instance
    """
    global _communicator
    
    with _communicator_lock:
        if _communicator is None:
            _communicator = SubprocessCommunicator(experiment_id)
        return _communicator


def send_subprocess_progress(experiment_id: str,
                           progress: float,
                           stage: str,
                           message: str,
                           level: str = "info") -> bool:
    """
    Convenience function to send progress from subprocess.
    
    This is the main function that experiments should use for progress reporting.
    
    Args:
        experiment_id: The experiment identifier (e.g., "EXP-01")
        progress: Progress percentage (0.0 to 100.0)
        stage: Current stage of the experiment
        message: Detailed progress message
        level: Message level (info, warning, error, success)
        
    Returns:
        True if message was sent successfully, False otherwise
        
    Example:
        # In an experiment file
        from fractalsemantics.subprocess_comm import send_subprocess_progress
        
        # Send progress update
        send_subprocess_progress("EXP-01", 25.0, "Generation", "Generating bit-chains", "info")
        send_subprocess_progress("EXP-01", 50.0, "Analysis", "Analyzing collisions", "info")
        send_subprocess_progress("EXP-01", 100.0, "Complete", "Experiment completed successfully", "success")
    """
    try:
        communicator = get_subprocess_communicator(experiment_id)
        return communicator.send_progress(progress, stage, message, level)
    except Exception as e:
        # Fallback to basic progress communication if subprocess communicator fails
        try:
            report_progress(experiment_id, progress, stage, message)
            return True
        except Exception:
            # Final fallback - just print to stderr
            print(f"[{experiment_id}] Progress: {progress}% - {stage}: {message}", file=sys.stderr)
            return False


def send_subprocess_status(experiment_id: str, status: str, details: Optional[str] = None) -> bool:
    """
    Send status update from subprocess.
    
    Args:
        experiment_id: The experiment identifier
        status: Status message
        details: Optional additional details
        
    Returns:
        True if message was sent successfully, False otherwise
    """
    try:
        communicator = get_subprocess_communicator(experiment_id)
        return communicator.send_status(status, details)
    except Exception:
        return False


def send_subprocess_error(experiment_id: str, error_message: str, exception: Optional[Exception] = None) -> bool:
    """
    Send error message from subprocess.
    
    Args:
        experiment_id: The experiment identifier
        error_message: Error description
        exception: Optional exception object
        
    Returns:
        True if message was sent successfully, False otherwise
    """
    try:
        communicator = get_subprocess_communicator(experiment_id)
        return communicator.send_error(error_message, exception)
    except Exception:
        return False


def send_subprocess_completion(experiment_id: str, success: bool, details: Optional[str] = None) -> bool:
    """
    Send completion message from subprocess.
    
    Args:
        experiment_id: The experiment identifier
        success: Whether the experiment completed successfully
        details: Optional completion details
        
    Returns:
        True if message was sent successfully, False otherwise
    """
    try:
        communicator = get_subprocess_communicator(experiment_id)
        return communicator.send_completion(success, details)
    except Exception:
        return False


def is_subprocess_communication_enabled() -> bool:
    """
    Check if subprocess communication is enabled.
    
    Returns:
        True if progress file is configured and accessible
    """
    progress_file = os.environ.get("FRACTALSEMANTICS_PROGRESS_FILE")
    if not progress_file:
        return False
    
    try:
        # Check if we can write to the progress file
        Path(progress_file).parent.mkdir(parents=True, exist_ok=True)
        with open(progress_file, 'a') as f:
            pass
        return True
    except Exception:
        return False


def get_subprocess_runtime_info(experiment_id: str) -> Dict[str, Any]:
    """
    Get runtime information for the current subprocess.
    
    Args:
        experiment_id: The experiment identifier
        
    Returns:
        Dictionary with runtime information
    """
    try:
        communicator = get_subprocess_communicator(experiment_id)
        return communicator.get_runtime_info()
    except Exception:
        return {
            "subprocess_id": os.getpid(),
            "parent_process_id": os.getppid(),
            "experiment_id": experiment_id,
            "progress_file": os.environ.get("FRACTALSEMANTICS_PROGRESS_FILE"),
            "error": "Failed to get runtime info"
        }


# Backward compatibility - these functions are now aliases for the subprocess versions
def send_progress_message(experiment: str, progress: float, stage: str, message: str, level: str = "info") -> bool:
    """
    Backward compatibility alias for send_subprocess_progress.
    
    This maintains compatibility with existing code that imports send_progress_message
    directly from this module.
    """
    return send_subprocess_progress(experiment, progress, stage, message, level)


if __name__ == "__main__":
    # Test the subprocess communication system
    print("Testing subprocess communication...")
    
    # Test basic progress sending
    success = send_subprocess_progress("TEST", 50.0, "Test Stage", "Testing progress communication", "info")
    print(f"Progress message sent: {success}")
    
    # Test status sending
    success = send_subprocess_status("TEST", "Test Status", "Testing status communication")
    print(f"Status message sent: {success}")
    
    # Test completion sending
    success = send_subprocess_completion("TEST", True, "Test completed successfully")
    print(f"Completion message sent: {success}")
    
    # Test runtime info
    runtime_info = get_subprocess_runtime_info("TEST")
    print(f"Runtime info: {runtime_info}")
    
    print("Subprocess communication test completed.")