import subprocess
import sys


def run_step(step_number, description, script_path):
    print(f"\n[Step {step_number}] {description}")
    print("-" * 50)
    try:
        subprocess.run([sys.executable, script_path], check=True, text=True)
        print(f"[Step {step_number}] Done.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Step {step_number}] FAILED (exit code {e.returncode}). Pipeline stopped.")
        return False
    except FileNotFoundError:
        print(f"[Step {step_number}] FAILED — script not found: {script_path}. Pipeline stopped.")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("  TRANSACTION ANALYSIS PIPELINE")
    print("=" * 50)

    steps = [
        (1, "Merge Data              (merge.py)",        "ML/dataprocessing/merge.py"),
        (2, "Encode Features         (encoding.py)",     "ML/dataprocessing/encoding.py"),
        (3, "Normalize & Extract     (preprocessing.py)","ML/dataprocessing/preprocessing.py"),
        (4, "Anomaly Detection       (Isolation Forest)","ML/src/isolationforest.py"),
        (5, "Fraud Classification    (Random Forest)",   "ML/src/random_forest.py"),
        (6, "Spending Prediction     (RF Regressor)",    "ML/src/rdregressor.py"),
    ]

    for step_number, description, script in steps:
        if not run_step(step_number, description, script):
            break

   
    print("  PIPELINE COMPLETE")

