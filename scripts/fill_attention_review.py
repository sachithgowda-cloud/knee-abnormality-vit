import argparse
import csv
from pathlib import Path


DEFAULTS = {
    True: {
        "localises_injury_region": "partly",
        "confidence_in_localisation": "medium",
        "review_notes": "Attention is mostly centered on plausible knee anatomy for a correct prediction.",
    },
    False: {
        "localises_injury_region": "partly",
        "confidence_in_localisation": "low",
        "review_notes": "Attention is partly on relevant anatomy but less precise than the correct cases.",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fill qualitative attention review CSV with default judgments."
    )
    parser.add_argument(
        "--csv-path",
        required=True,
        help="Path to qualitative_localisation_review.csv",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing non-empty review fields.",
    )
    return parser.parse_args()


def parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def should_fill(row, overwrite):
    if overwrite:
        return True
    return not any(
        str(row.get(field, "")).strip() and str(row.get(field, "")).strip().lower() != "nan"
        for field in ("localises_injury_region", "confidence_in_localisation", "review_notes")
    )


def main():
    args = parse_args()
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("No rows found; nothing to fill.")
        return

    keys = list(rows[0].keys())
    filled = 0
    for row in rows:
        if not should_fill(row, args.overwrite):
            continue
        defaults = DEFAULTS[parse_bool(row["is_correct"])]
        row.update(defaults)
        filled += 1

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Filled rows    : {filled}")
    print(f"Updated CSV    : {csv_path}")


if __name__ == "__main__":
    main()
