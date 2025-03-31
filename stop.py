"""
Stop Utility - Control script for voice agent termination.

This module provides utilities to gracefully or forcefully
terminate the voice agent by creating control files that
the main process monitors.
"""

import os
import sys
import time
from typing import Optional


def create_stop_file() -> None:
    """Create a STOP file to gracefully terminate the voice agent.
    
    The voice agent will detect this file and terminate gracefully,
    completing any in-progress operations before shutting down.
    """
    try:
        with open("STOP", "w") as f:
            f.write("STOP")
        print("STOP file created. The voice agent should terminate soon.")
        
        # Wait to see if it worked
        time.sleep(2)
        
        # Check if the file was removed (which would indicate the agent is stopping)
        if not os.path.exists("STOP"):
            print("Voice agent is shutting down...")
        else:
            print("Voice agent may not be running or is not responding to the STOP file.")
            # Clean up the file
            try:
                os.remove("STOP")
            except Exception as e:
                print(f"Error cleaning up STOP file: {e}")
    except Exception as e:
        print(f"Error creating STOP file: {e}")


def create_force_stop_file() -> None:
    """Create a FORCE_STOP file to immediately terminate the voice agent.
    
    The voice agent will detect this file and exit immediately,
    without completing any in-progress operations.
    """
    try:
        with open("FORCE_STOP", "w") as f:
            f.write("FORCE_STOP")
        print("FORCE_STOP file created. The voice agent should terminate immediately.")
        
        # Wait to see if it worked
        time.sleep(1)
        
        # Clean up if still exists
        if os.path.exists("FORCE_STOP"):
            try:
                os.remove("FORCE_STOP")
                print("FORCE_STOP file removed. The voice agent may not be running.")
            except Exception as e:
                print(f"Error cleaning up FORCE_STOP file: {e}")
    except Exception as e:
        print(f"Error creating FORCE_STOP file: {e}")


def main() -> int:
    """Execute the stop operation based on command line arguments.
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    print("Sending stop signal to Voice Agent...")
    
    try:
        # If an argument is provided and it's "force", create a FORCE_STOP file
        if len(sys.argv) > 1 and sys.argv[1].lower() == "force":
            create_force_stop_file()
        else:
            create_stop_file()
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 