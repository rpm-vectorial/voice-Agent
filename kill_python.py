"""
Kill Python - Utility to terminate all Python processes.

This module provides a utility to kill all running Python processes
except for the current process. This is useful for cleaning up
stray processes, especially when debugging voice agent issues.
"""

import os
import signal
import subprocess
import sys
from typing import List, Optional


def kill_python_processes() -> int:
    """Kill all Python processes except this one.
    
    Returns:
        int: Number of terminated processes
    """
    # Get the current process ID
    current_pid = os.getpid()
    terminated_count = 0
    
    # Find all Python processes
    try:
        if sys.platform == "darwin" or sys.platform == "linux":  # macOS or Linux
            python_pids = _get_unix_python_pids(current_pid)
            terminated_count = _kill_unix_processes(python_pids)
        elif sys.platform == "win32":  # Windows
            terminated_count = _kill_windows_processes(current_pid)
        
        return terminated_count
    
    except Exception as e:
        print(f"Error: {e}")
        return 0


def _get_unix_python_pids(current_pid: int) -> List[int]:
    """Get list of Python process IDs on Unix-like systems (macOS/Linux).
    
    Args:
        current_pid: Current process ID to exclude
        
    Returns:
        List of Python process IDs
    """
    # Use ps command to find Python processes
    result = subprocess.run(
        ["ps", "-e", "-o", "pid,command"], 
        capture_output=True, 
        text=True, 
        check=True
    )
    
    # Parse the output
    lines = result.stdout.strip().split('\n')
    python_pids = []
    
    for line in lines:
        if "python" in line.lower() and "kill_python.py" not in line:
            parts = line.strip().split()
            if parts:
                try:
                    pid = int(parts[0])
                    if pid != current_pid:  # Skip current process
                        python_pids.append(pid)
                except ValueError:
                    continue
    
    return python_pids


def _kill_unix_processes(python_pids: List[int]) -> int:
    """Kill Python processes on Unix-like systems by PID.
    
    Args:
        python_pids: List of process IDs to kill
        
    Returns:
        Number of terminated processes
    """
    terminated_count = 0
    
    # Kill the processes
    for pid in python_pids:
        try:
            print(f"Killing Python process with PID: {pid}")
            os.kill(pid, signal.SIGTERM)
            terminated_count += 1
        except Exception as e:
            print(f"Error killing process {pid}: {e}")
    
    return terminated_count


def _kill_windows_processes(current_pid: int) -> int:
    """Kill Python processes on Windows.
    
    Args:
        current_pid: Current process ID to exclude
        
    Returns:
        Number of terminated processes
    """
    terminated_count = 0
    
    # Use tasklist and taskkill commands
    result = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV"], 
        capture_output=True, 
        text=True
    )
    
    # Parse the output
    lines = result.stdout.strip().split('\n')
    for line in lines[1:]:  # Skip header line
        if "python.exe" in line:
            parts = line.strip().split(',')
            if len(parts) > 1:
                pid_part = parts[1].strip('"')
                try:
                    pid = int(pid_part)
                    if pid != current_pid:  # Skip current process
                        print(f"Killing Python process with PID: {pid}")
                        subprocess.run(["taskkill", "/F", "/PID", str(pid)])
                        terminated_count += 1
                except ValueError:
                    continue
    
    return terminated_count


def main() -> int:
    """Main entry point.
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    try:
        print("Killing all Python processes...")
        terminated_count = kill_python_processes()
        print(f"Terminated {terminated_count} Python processes")
        print("Done.")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 