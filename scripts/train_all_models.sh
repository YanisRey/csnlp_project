#!/bin/bash

# ===== Bash script to run training processes =====
# Activate Python virtual environment
source ./venv/bin/activate

# Define training scripts to run
scripts=(
  "../training/train_embeddings.py"
  "../training/train_simplified_phonetics.py"
  "../training/train_finetuned_fasttext.py"
  "../training/patch_finetuned_fasttext.py"
)

total=${#scripts[@]}

echo
echo "🚀 Starting training scripts..."

for ((i=0; i<total; i++)); do
    script="${scripts[$i]}"
    echo
    echo "🚀 Running script $((i+1))/$total: $script"
    python "$script"

    if [[ $? -ne 0 ]]; then
        echo "❌ Error in script $script"
        echo "Last error code: $?"
        exit 1
    fi

    echo "✅ Completed: $script"
done

echo
echo "🎉 All training scripts executed successfully!"
