import subprocess

scripts = [
    r"..\preprocess\text_preprocessing\load_data.py",
    r"..\preprocess\text_preprocessing\simplify_phonetics.py",
    r"..\preprocess\misspellings_preprocessing\load_mispelling.py",
    r"..\preprocess\misspellings_preprocessing\clean_misspellings.py"
]

for i, script in enumerate(scripts, 1):
    print(f"\n🔄 Running script {i}/{len(scripts)}: {script}")
    try:
        subprocess.run(["python", script], check=True)
        print(f"✅ Finished: {script}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error while running {script}: {e}")
        break
