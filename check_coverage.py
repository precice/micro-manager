"""
Script to check if test coverage is above a predefined threshold.
Reads total coverage percentage from stdin (output of coverage report --format=total).
"""

import sys

THRESHOLD = 60  # Minimum required coverage percentage


if __name__ == "__main__":
    try:
        coverage = int(sys.stdin.read().strip())
    except ValueError:
        print("Error: could not parse coverage value from stdin.")
        sys.exit(1)

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
