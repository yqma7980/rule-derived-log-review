"""Verify generated test fixtures without the separately held research tables."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synthetic-output", type=Path,
        default=Path(__file__).resolve().parents[1] / "examples/verified_synthetic",
    )
    args = parser.parse_args()
    required = ("summary.json", "synthetic_points.csv", "synthetic_blocks.csv")
    missing = [name for name in required if not (args.synthetic_output / name).is_file()]
    if missing:
        parser.error("Missing generated fixture files: " + ", ".join(missing))
    from verify_independent import verify_synthetic

    try:
        result = verify_synthetic(args.synthetic_output)
    except (OSError, ValueError, KeyError, AssertionError) as exc:
        parser.error("Synthetic verification failed: " + (str(exc) or type(exc).__name__))
    print(json.dumps({
        "status": "PASS", "synthetic": result,
        "scope": "Generated controls only; no research-table readback or model fitting",
    }, indent=2))


if __name__ == "__main__":
    main()
