#!/usr/bin/env python3
"""
Wrapper script to run the dataset command interface with proper package structure.
"""

import sys
import os

# Set up the Python path to treat src as a package
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Now run the dataset command interface
if __name__ == "__main__":
    # Import and run the main function
    from dataset_command_interface import main
    main()