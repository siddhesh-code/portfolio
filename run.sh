#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

# Export Flask development settings
export FLASK_APP=webapp.py
export FLASK_ENV=development
export FLASK_DEBUG=1

# Run Flask application
python -m flask run --host=0.0.0.0 --port=5000
