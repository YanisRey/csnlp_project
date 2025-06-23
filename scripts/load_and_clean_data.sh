#!/bin/bash

# ===== Bash script to run preprocessing steps =====
# Ensure the virtual environment is activated
source ./venv/bin/activate

# Define scripts to run (same order as Python version)
scripts=(
  "../preprocess/text_preprocessing/load_data.py"
)

total=${#scripts[@]}

echo
echo "🔁 Starting preprocessing scripts..."

for ((i=0; i<total; i++)); do
    script="${scripts[$i]}"
    echo
    echo "🔄 Running script $((i+1))/$total: $script"
    python "$script"

    if [[ $? -ne 0 ]]; then
        echo "❌ Error while running $script"
        exit 1
    fi

    echo "✅ Finished: $script"
done

echo
echo "🎉 All preprocessing scripts completed successfully!"
