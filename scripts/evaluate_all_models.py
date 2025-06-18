import subprocess

scripts = [
    "../evaluate/avg_cos_per_misspelling.py",
    "../evaluate/human_scored_pairs.py"
]

for i, script in enumerate(scripts, 1):
    print(f"\n📊 Running evaluation script {i}/{len(scripts)}: {script}")
    try:
        subprocess.run(["python", script], check=True)
        print(f"✅ Finished: {script}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {script}: {e}")
        break
