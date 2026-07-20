from pathlib import Path
import csv
from collections import defaultdict

# ================== 路径配置 ==================

INPUT_CSV = Path(r"C:\Users\20552\Desktop\sam2_metric_results.csv")
OUTPUT_CSV = Path(r"C:\Users\20552\Desktop\per_category_results.csv")

# =============================================


def mean_value(rows, key):
    values = [float(r[key]) for r in rows if r[key] != ""]
    if len(values) == 0:
        return 0.0
    return sum(values) / len(values)


def main():
    if not INPUT_CSV.exists():
        print(f"Input CSV not found: {INPUT_CSV}")
        print("Please run evaluate_sam2_metrics.py first.")
        return

    with open(INPUT_CSV, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    category_groups = defaultdict(list)

    for r in rows:
        category = r["category"]
        if category is not None and category != "":
            category_groups[category].append(r)

    output_rows = []

    print("=" * 80)
    print("Per-category SAM2 Evaluation Results")
    print("=" * 80)

    for category in sorted(category_groups.keys()):
        cat_rows = category_groups[category]

        bad_rows = [r for r in cat_rows if r["status"] == "bad"]
        good_rows = [r for r in cat_rows if r["status"] == "good"]

        bad_total = len(bad_rows)
        good_total = len(good_rows)
        total_images = len(cat_rows)

        bad_with_mask = sum(1 for r in bad_rows if r["pred_has_mask"] == "1")
        bad_missed = sum(1 for r in bad_rows if r["pred_has_mask"] == "0")
        good_false_positives = sum(1 for r in good_rows if r["pred_has_mask"] == "1")

        all_miou = mean_value(cat_rows, "iou")
        all_dice = mean_value(cat_rows, "dice")
        all_bf1 = mean_value(cat_rows, "boundary_f1")

        defect_miou = mean_value(bad_rows, "iou")
        defect_dice = mean_value(bad_rows, "dice")
        defect_bf1 = mean_value(bad_rows, "boundary_f1")

        output_rows.append({
            "category": category,
            "total_images": total_images,
            "bad_images": bad_total,
            "good_images": good_total,
            "bad_with_mask": bad_with_mask,
            "bad_missed": bad_missed,
            "good_false_positives": good_false_positives,
            "all_miou": all_miou,
            "all_dice": all_dice,
            "all_boundary_f1": all_bf1,
            "defect_miou": defect_miou,
            "defect_dice": defect_dice,
            "defect_boundary_f1": defect_bf1,
        })

        print(f"Category: {category}")
        print(f"  Total images: {total_images}")
        print(f"  Bad images: {bad_total}")
        print(f"  Good images: {good_total}")
        print(f"  Bad with mask: {bad_with_mask}")
        print(f"  Bad missed: {bad_missed}")
        print(f"  Good false positives: {good_false_positives}")
        print(f"  Defect mIoU: {defect_miou:.4f}")
        print(f"  Defect Dice: {defect_dice:.4f}")
        print(f"  Defect Boundary-F1: {defect_bf1:.4f}")
        print("-" * 80)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "category",
            "total_images",
            "bad_images",
            "good_images",
            "bad_with_mask",
            "bad_missed",
            "good_false_positives",
            "all_miou",
            "all_dice",
            "all_boundary_f1",
            "defect_miou",
            "defect_dice",
            "defect_boundary_f1",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print("=" * 80)
    print(f"Per-category results saved to: {OUTPUT_CSV}")
    print("=" * 80)


if __name__ == "__main__":
    main()