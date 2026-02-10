#!/bin/bash
# FractalSemantics Launcher Script

echo "🚀 Starting FractalSemantics..."

# Activate virtual environment
source venv/bin/activate

# Set environment based on first argument
if [ "$1" = "dev" ]; then
    export FRACTALSEMANTICS_ENV=dev
    echo "Using development configuration (faster, smaller samples)"
elif [ "$1" = "ci" ]; then
    export FRACTALSEMANTICS_ENV=ci
    echo "Using CI configuration"
else
    export FRACTALSEMANTICS_ENV=production
    echo "Using production configuration"
fi

# Run experiments
python -m fractalsemantics.fractalsemantics_experiments
