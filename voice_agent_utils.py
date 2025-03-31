"""
Voice Agent Utilities - Control utilities for voice agent.

This module provides integrated utility functions for controlling the voice agent:
1. Creating stop files for graceful termination
2. Terminating Python processes for emergency shutdown
"""

import os
import sys
import time
import signal
import subprocess
from typing import List, Optional, Tuple


def create_stop_file(force: bool = False) -> bool:
    """Create a STOP or FORCE_STOP file to terminate the voice agent.
    
    Args:
        force: Whether to create a FORCE_STOP file for immediate termination
        
    Returns:
        True if successful, False if failed
    """
    file_name = "FORCE_STOP" if force else "STOP"
    try:
        with open(file_name, "w") as f:
            f.write(file_name)
        
        print(f"{file_name} file created. The voice agent should terminate {'immediately' if force else 'soon'}.")
        
        # Wait briefly to see if it worked
        time.sleep(1 if force else 2)
        
        # Check if the file was removed (which would indicate the agent is stopping)
        if not os.path.exists(file_name):
            print("Voice agent is shutting down...")
            return True
        else:
            print("Voice agent may not be running or is not responding to the file.")
            # Clean up the file
            try:
                os.remove(file_name)
            except Exception as e:
                print(f"Error cleaning up {file_name} file: {e}")
            return False
    except Exception as e:
        print(f"Error creating {file_name} file: {e}")
        return False


def kill_python_processes(exclude_current: bool = True) -> int:
    """Kill Python processes.
    
    Args:
        exclude_current: Whether to exclude the current process
        
    Returns:
        Number of terminated processes
    """
    # Get the current process ID if we're excluding it
    current_pid = os.getpid() if exclude_current else None
    terminated_count = 0
    
    # Find all Python processes
    try:
        if sys.platform == "darwin" or sys.platform == "linux":  # macOS or Linux
            terminated_count = _kill_unix_python_processes(current_pid)
        elif sys.platform == "win32":  # Windows
            terminated_count = _kill_windows_python_processes(current_pid)
        
        print(f"Terminated {terminated_count} Python processes")
        return terminated_count
    
    except Exception as e:
        print(f"Error killing Python processes: {e}")
        return 0


def _kill_unix_python_processes(exclude_pid: Optional[int] = None) -> int:
    """Kill Python processes on Unix-like systems.
    
    Args:
        exclude_pid: Process ID to exclude from termination
        
    Returns:
        Number of terminated processes
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
        if "python" in line.lower() and "voice_agent_utils.py" not in line:
            parts = line.strip().split()
            if parts:
                try:
                    pid = int(parts[0])
                    if exclude_pid is None or pid != exclude_pid:
                        python_pids.append(pid)
                except ValueError:
                    continue
    
    # Kill the processes
    terminated_count = 0
    for pid in python_pids:
        try:
            print(f"Killing Python process with PID: {pid}")
            os.kill(pid, signal.SIGTERM)
            terminated_count += 1
        except Exception as e:
            print(f"Error killing process {pid}: {e}")
    
    return terminated_count


def _kill_windows_python_processes(exclude_pid: Optional[int] = None) -> int:
    """Kill Python processes on Windows.
    
    Args:
        exclude_pid: Process ID to exclude from termination
        
    Returns:
        Number of terminated processes
    """
    # Use tasklist and taskkill commands
    result = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV"], 
        capture_output=True, 
        text=True
    )
    
    # Parse the output
    terminated_count = 0
    lines = result.stdout.strip().split('\n')
    for line in lines[1:]:  # Skip header line
        if "python.exe" in line:
            parts = line.strip().split(',')
            if len(parts) > 1:
                pid_part = parts[1].strip('"')
                try:
                    pid = int(pid_part)
                    if exclude_pid is None or pid != exclude_pid:
                        print(f"Killing Python process with PID: {pid}")
                        subprocess.run(["taskkill", "/F", "/PID", str(pid)])
                        terminated_count += 1
                except ValueError:
                    continue
    
    return terminated_count


def stop_voice_agent(force: bool = False, kill_processes: bool = False) -> Tuple[bool, int]:
    """Comprehensive utility to stop the voice agent.
    
    Args:
        force: Whether to use force stop
        kill_processes: Whether to also kill Python processes
        
    Returns:
        Tuple of (stop_file_created, processes_terminated)
    """
    stop_result = create_stop_file(force=force)
    processes_terminated = 0
    
    if kill_processes:
        print("Additionally killing Python processes...")
        processes_terminated = kill_python_processes()
    
    return (stop_result, processes_terminated)


if __name__ == "__main__":
    # Process command line arguments
    force_flag = False
    kill_flag = False
    
    for arg in sys.argv[1:]:
        if arg.lower() in ["force", "--force", "-f"]:
            force_flag = True
        elif arg.lower() in ["kill", "--kill", "-k"]:
            kill_flag = True
    
    # Apply the requested operation
    if kill_flag and not force_flag:
        print("Killing all Python processes...")
        kill_python_processes()
    else:
        print(f"Sending {'force ' if force_flag else ''}stop signal to Voice Agent...")
        stop_result, processes = stop_voice_agent(force=force_flag, kill_processes=kill_flag)
        
        if not stop_result and not kill_flag:
            user_input = input("Voice agent didn't respond to stop file. Kill all Python processes? (y/n): ")
            if user_input.lower() in ["y", "yes"]:
                kill_python_processes() 