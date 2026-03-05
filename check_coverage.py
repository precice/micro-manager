"""
Script to check if test coverage is above a predefined threshold.
Run after coverage.py has generated a coverage report.
"""

import subprocess
import sys

THRESHOLD = 20  # Minimum required coverage percentage
# TODO: Increase threshold as test coverage improves


def get_coverage():
    result = subprocess.run(
        ["python3", "-m", "coverage", "report", "--format=total"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("Error running coverage report:")
        print(result.stderr)
        sys.exit(1)
    return int(result.stdout.strip())


if __name__ == "__main__":
    coverage = get_coverage()
    print("Total coverage: {}%".format(coverage))
    if coverage < THRESHOLD:
        print(
            "Coverage {}% is below the required threshold of {}%.".format(
                coverage, THRESHOLD
            )
        )
        sys.exit(1)
    else:
        print(
            "Coverage {}% meets the required threshold of {}%.".format(
                coverage, THRESHOLD
            )
        )
        sys.exit(0)
