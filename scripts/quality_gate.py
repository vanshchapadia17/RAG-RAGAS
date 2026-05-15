import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import SCORES_FILE, QUALITY_GATES


def run_quality_gate():
    print("=" * 40)
    print("Quality Gate Check")
    print("=" * 40)

    if not os.path.exists(SCORES_FILE):
        print("No scores file found!")
        print("Run evaluation first.")
        sys.exit(1)

    with open(SCORES_FILE) as f:
        scores = json.load(f)

    failed = []
    for metric, threshold in QUALITY_GATES.items():
        if metric not in scores:
            print(f"  [SKIP   ] {metric}: not found in scores file")
            continue

        score = scores[metric]
        passed = score >= threshold
        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"  [{status}] {metric}: {score:.4f} (need {threshold})")

        if not passed:
            failed.append(metric)

    print("=" * 40)
    if failed:
        print(f"FAILED — {len(failed)} gate(s) did not pass:")
        for f in failed:
            print(f"  - {f}")
        print("Deployment blocked.")
        sys.exit(1)
    else:
        print("ALL GATES PASSED — Safe to deploy!")
        sys.exit(0)


if __name__ == "__main__":
    run_quality_gate()