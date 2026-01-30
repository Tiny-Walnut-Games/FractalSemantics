#!/bin/bash
# FractalStat Launcher Script

echo "🚀 Starting FractalStat..."

# Activate virtual environment
source venv/bin/activate

# Set environment based on first argument
if [ "$1" = "dev" ]; then
    export FRACTALSTAT_ENV=dev
    echo "Using development configuration (faster, smaller samples)"
elif [ "$1" = "ci" ]; then
    export FRACTALSTAT_ENV=ci
    echo "Using CI configuration"
else
    export FRACTALSTAT_ENV=production
    echo "Using production configuration"
fi

# Run experiments
python -m fractalstat.fractalstat_experiments
