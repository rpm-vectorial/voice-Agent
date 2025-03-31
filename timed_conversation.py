"""
Timed Conversation - Run voice agent with automatic time limit.

This module provides functionality to run the voice agent with an automatic
time limit, ensuring that it will shut down after a specified duration.
Useful for testing or when you want to ensure the voice agent doesn't
run indefinitely.
"""

import subprocess
import threading
import time
import os
import signal
import sys
from typing import Optional

# Local imports
try:
    import voice_agent_utils
    has_utils = True
except ImportError:
    has_utils = False


def run_voice_agent() -> subprocess.Popen:
    """Run the voice agent in a separate process.
    
    Returns:
        subprocess.Popen: Process object for the running voice agent
    """
    print("Starting voice agent...")
    try:
        process = subprocess.Popen(["python", "openai_pipecat.py"])
        return process
    except Exception as e:
        print(f"Error starting voice agent: {e}")
        raise


def create_stop_file() -> None:
    """Create a STOP file to gracefully terminate the voice agent."""
    if has_utils:
        voice_agent_utils.create_stop_file(force=False)
    else:
        try:
            with open("STOP", "w") as f:
                f.write("STOP")
            print("STOP file created. The voice agent should terminate soon.")
        except Exception as e:
            print(f"Error creating STOP file: {e}")
            raise


def timer_thread(seconds: int, process: subprocess.Popen) -> None:
    """Wait for the specified number of seconds and then terminate the process.
    
    Args:
        seconds: Number of seconds to wait before termination
        process: Process object for the running voice agent
    """
    print(f"Voice agent will run for {seconds} seconds.")
    time.sleep(seconds)
    
    print(f"Time's up! Shutting down voice agent...")
    
    try:
        # Try graceful termination first
        create_stop_file()
        
        # Wait for graceful termination
        time.sleep(3)
        
        # If still running, use more forceful methods
        if process.poll() is None:
            if has_utils:
                # Try force stop first
                print("Voice agent still running. Trying force stop...")
                voice_agent_utils.create_stop_file(force=True)
                time.sleep(2)
                
                # If still running, kill Python processes
                if process.poll() is None:
                    print("Voice agent still running. Killing Python processes...")
                    voice_agent_utils.kill_python_processes()
            else:
                # Use standard termination methods
                print("Voice agent still running. Sending termination signal...")
                if sys.platform == "win32":
                    process.terminate()
                else:
                    os.kill(process.pid, signal.SIGTERM)
                
                time.sleep(2)
                
                # If still running, force kill
                if process.poll() is None:
                    print("Voice agent still running. Force killing...")
                    if sys.platform == "win32":
                        os.system(f"taskkill /F /PID {process.pid}")
                    else:
                        os.kill(process.pid, signal.SIGKILL)
    except Exception as e:
        print(f"Error terminating voice agent: {e}")
        # Try force kill as last resort
        try:
            if process.poll() is None:
                if sys.platform == "win32":
                    os.system(f"taskkill /F /PID {process.pid}")
                else:
                    os.kill(process.pid, signal.SIGKILL)
        except:
            pass


def main() -> int:
    """Run the voice agent with a time limit.
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    # Check command line arguments for custom time limit
    time_limit = 60  # Default 60 seconds
    
    # Process command line arguments
    for i, arg in enumerate(sys.argv[1:]):
        if arg in ["-t", "--time"]:
            if i + 1 < len(sys.argv[1:]):
                try:
                    time_limit = int(sys.argv[i+2])
                    print(f"Using custom time limit: {time_limit} seconds")
                except (ValueError, IndexError):
                    print("Invalid time limit. Using default: 60 seconds")
    
    try:
        # Start the voice agent
        process = run_voice_agent()
        
        # Start the timer thread
        t = threading.Thread(target=timer_thread, args=(time_limit, process))
        t.daemon = True
        t.start()
        
        try:
            # Wait for the process to complete
            process.wait()
        except KeyboardInterrupt:
            print("\nUser interrupted. Stopping voice agent...")
            # Clean up
            create_stop_file()
            time.sleep(2)
            
            # If still running, try more forceful methods
            if process.poll() is None:
                if has_utils:
                    voice_agent_utils.stop_voice_agent(force=True)
                else:
                    process.terminate()
        
        print("Voice agent has been terminated.")
        return 0
    except Exception as e:
        print(f"Error in timed conversation: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 