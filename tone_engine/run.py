#!/usr/bin/env python3
"""
Quick runner script for tone engine
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import and run main
from main import main

if __name__ == "__main__":
    main() 