import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a qualitative review sheet for attention overlays."
    )
    parser.add_argument(
        "--attention-dir",
        required=True,
        help="Directory created by scripts/visualize_attention.py.",
    )
    return parser.parse_args()


def load_records(attention_dir):
    metadata_path = Path(attention_dir) / "attention_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Attention metadata not found: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_review_rows(records):
    rows = []
    for record in records:
        rows.append({
            "sample_index": record["sample_index"],
            "case_id": record["case_id"],
            "slice_index": record["slice_index"],
            "true_label": record["true_label"],
            "pred_label": record["pred_label"],
            "is_correct": record["is_correct"],
            "overlay_path": record["overlay_path"],
            "attention_path": record["attention_path"],
            "localises_injury_region": "",
            "confidence_in_localisation": "",
            "review_notes": "",
        })
    return rows


def save_review_sheet(rows, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def main():
    args = parse_args()
    attention_dir = Path(args.attention_dir)
    records = load_records(attention_dir)
    rows = build_review_rows(records)
    output_path = attention_dir / "qualitative_localisation_review.csv"
    save_review_sheet(rows, output_path)

    print(f"Review sheet   : {output_path}")
    print(f"Num samples    : {len(rows)}")


if __name__ == "__main__":
    main()
