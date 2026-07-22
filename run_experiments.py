#!/usr/bin/env python3
"""
Run all experiments from the thesis:
  1. EDA on real electricity data
  2. EDA on synthetic validation data
  3. Main analysis: RSF-PHATE clustering on real contracts
  4. Synthetic validation: ARI = 0.48
  5. Multiproducto analysis: ARI = 0.26

Usage:
  python run_experiments.py

Expected runtime: ~15 min on modern CPU
"""

import subprocess
import sys
from pathlib import Path

def run_notebook(notebook_path, timeout=300):
    """Execute a Jupyter notebook and return success status."""
    notebook_path = Path(notebook_path)

    if not notebook_path.exists():
        print(f"❌ Notebook not found: {notebook_path}")
        return False

    print(f"\n{'='*80}")
    print(f"Running: {notebook_path.stem}")
    print(f"{'='*80}")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
             "--inplace", "--ExecutePreprocessor.timeout={}".format(timeout),
             str(notebook_path)],
            capture_output=False,
            timeout=timeout + 60
        )
        if result.returncode == 0:
            print(f"✓ {notebook_path.stem} completed")
            return True
        else:
            print(f"❌ {notebook_path.stem} failed with return code {result.returncode}")
            return False
    except subprocess.TimeoutExpired:
        print(f"❌ {notebook_path.stem} timed out after {timeout} seconds")
        return False
    except Exception as e:
        print(f"❌ {notebook_path.stem} error: {e}")
        return False


def main():
    """Main experiment runner."""
    notebooks_dir = Path("notebooks")

    if not notebooks_dir.exists():
        print(f"❌ Notebooks directory not found: {notebooks_dir}")
        sys.exit(1)

    # List of notebooks to run in order
    experiments = [
        ("1. EDA Real Data", notebooks_dir / "eda_datos_reales.ipynb"),
        ("2. EDA Synthetic Data", notebooks_dir / "eda_datos_sinteticos.ipynb"),
        ("3. Main Analysis (RSF-PHATE)", notebooks_dir / "analisis_churn_survival.ipynb"),
        ("4. Synthetic Validation (ARI=0.48)", notebooks_dir / "validacion_sintetica.ipynb"),
        ("5. Multiproducto Analysis (ARI=0.26)", notebooks_dir / "analisis_multiproducto.ipynb"),
    ]

    print("="*80)
    print("RSF-PHATE Thesis Experiments Runner")
    print("="*80)
    print(f"\nExecuting {len(experiments)} experiments...\n")

    results = {}
    for name, notebook_path in experiments:
        print(f"\n[{name}]")
        success = run_notebook(notebook_path, timeout=300)
        results[name] = success

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    for name, success in results.items():
        status = "✓ PASS" if success else "❌ FAIL"
        print(f"{status} — {name}")

    passed = sum(1 for s in results.values() if s)
    total = len(results)
    print(f"\nResults: {passed}/{total} experiments completed")

    if passed == total:
        print("\n✓ All experiments completed successfully!")
        print("Output files are in the notebooks/ directory.")
        print("\nKey results:")
        print("  • Real data: 3 clusters identified (Churn Risk n=226, Loyal n=577, Moderate n=365)")
        print("  • Synthetic validation: ARI = 0.48 (robust method)")
        print("  • Multiproducto: ARI(A,B) = 0.26 (gas reorganizes the segmentation; protective factor)")
        return 0
    else:
        print(f"\n❌ {total - passed} experiment(s) failed. Check output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
