#!/bin/bash

# ===== Run evaluations in the correct virtual environment =====

# Activate the virtual environment
source ./venv/bin/activate

# Define the scripts to run (relative paths)
script1="../evaluate/avg_cos_per_misspelling.py"
script2="../evaluate/human_scored_pairs.py"

# Counter for script tracking
count=0

# Run first script
((count++))
echo
echo "📊 Running evaluation script $count/2: $script1"
python "$script1"
if [[ $? -ne 0 ]]; then
    echo "❌ Error running $script1"
    exit 1
fi
echo "✅ Finished: $script1"

# Run second script
((count++))
echo
echo "📊 Running evaluation script $count/2: $script2"
python "$script2"
if [[ $? -ne 0 ]]; then
    echo "❌ Error running $script2"
    exit 1
fi
echo "✅ Finished: $script2"

echo
echo "🎉 All evaluations completed successfully!"
