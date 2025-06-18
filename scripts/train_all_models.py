import subprocess

scripts = [
    "../training/train_embeddings.py",
    "../training/train_simplified_phonetics.py",
    "../training/train_finetuned_fasttext.py",
    "../training/patch_finetuned_fasttext.py"
]

for i, script in enumerate(scripts, 1):
    print(f"\n🚀 Running script {i}/{len(scripts)}: {script}")
    try:
        subprocess.run(["python", script], check=True)
        print(f"✅ Completed: {script}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in script {script}: {e}")
        break
