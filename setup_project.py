#!/usr/bin/env python
# setup_project.py
"""
Script to set up the project directory structure and create necessary files.
Run this script once to initialize your project.
"""

import os
import sys
import shutil

def create_directory(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    else:
        print(f"Directory already exists: {path}")

def create_empty_file(path):
    """Create empty file if it doesn't exist."""
    if not os.path.exists(path):
        with open(path, 'w') as f:
            pass
        print(f"Created file: {path}")
    else:
        print(f"File already exists: {path}")

def setup_project():
    """Set up project directory structure."""
    # Create main directories
    directories = [
        'data/raw',
        'data/processed',
        'data/ml',
        'analysis',
        'analysis/enhanced',
        'src/data',
        'src/models',
        'src/api',
        'models',
        'tests',
    ]
    
    for directory in directories:
        create_directory(directory)
    
    # Create __init__.py files
    init_files = [
        'src/__init__.py',
        'src/data/__init__.py',
        'src/models/__init__.py',
        'src/api/__init__.py',
        'tests/__init__.py',
    ]
    
    for init_file in init_files:
        create_empty_file(init_file)
    
    print("\nProject structure set up successfully!")
    print("\nNext steps:")
    print("1. Add your YouTube API key to a .env file")
    print("2. Run initial data collection: python initial_collection.py --regions US --analyze")
    print("3. Start the API server: python run_api.py")
    print("4. Run the ML pipeline: python run_ml_pipeline.py --step all")

if __name__ == "__main__":
    setup_project()