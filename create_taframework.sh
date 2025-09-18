#!/bin/bash

# Create directories
mkdir -p taframework/taframework
mkdir -p taframework/tests

# Create taframework files
touch taframework/taframework/__init__.py
touch taframework/taframework/enums.py
touch taframework/taframework/data_classes.py
touch taframework/taframework/exceptions.py
touch taframework/taframework/profiler.py
touch taframework/taframework/validator.py
touch taframework/taframework/indicator_engine.py
touch taframework/taframework/comparators.py
touch taframework/taframework/query_parser.py
touch taframework/taframework/analyzer.py
touch taframework/taframework/signal_generator.py
touch taframework/taframework/utils.py

# Create test files
touch taframework/tests/__init__.py
touch taframework/tests/test_analyzer.py

# Create project root files
touch taframework/README.md
touch taframework/setup.py
touch taframework/requirements.txt

echo "Directory structure created successfully!"

