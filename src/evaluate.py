import subprocess
import sys
from pathlib import Path

model_name = sys.argv[1] if len(sys.argv) > 1 else "lgbm_baseline"

scripts = [
    
    Path("src/threshold.py"),
    Path("src/metrics.py"),     
    Path("src/explain.py"),
    Path("src/fairness.py"),
]

for script in scripts:
    print(f"\n{'='*50}")
    print(f"Running {script.name} — model: {model_name}")
    print('='*50)
    subprocess.run([sys.executable, str(script), model_name], check=True)

print("\nEvaluation complete.") 