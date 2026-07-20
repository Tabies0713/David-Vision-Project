from pathlib import Path
import csv
import shutil
from collections import defaultdict

# ================== 路径配置 ==================

INPUT_CSV = Path(r"C:\Users\20552\Desktop\sam2_metric_results.csv")
VIS_DIR = Path(r"C:\Users\20552\Desktop\sam2_results\visualizations")

OUTPUT_DIR = Path(r"C:\Users\20552\Desktop\failure_case_analysis")
OUTPUT_CSV = OUTPUT_DIR / "failure_case_summary.csv"
OUTPUT_TXT = OUTPUT_DIR / "failure_case_report.txt"

# 分割失败阈值：Dice 低于这个值，认为 SAM2 分割效果较差
POOR_DICE_THRESHOLD = 0.40

# =============================================


def copy_visualization(row, subfolder):
    image_name = row["image"]
    stem = Path(image_name).stem
    vis_path = VIS_DIR / f"{stem}_comparison.jpg"

    save_dir = OUTPUT_DIR / subfolder
    save_dir.mkdir(parents=True, exist_ok=True)

    if vis_path.exists():
        shutil.copy(vis_path, save_dir / vis_path.name)
        return str(save_dir / vis_path.name)
    else:
        return "visualization_not_found"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_CSV.exists():
        print(f"CSV not found: {INPUT_CSV}")
        return

    with open(INPUT_CSV, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    failure_rows = []

    detection_missed = []
    segmentation_poor = []
    false_positive = []

    for r in rows:
        status = r["status"]
        pred_has_mask = r["pred_has_mask"]
        dice = float(r["dice"])
        iou = float(r["iou"])
        bf1 = float(r["boundary_f1"])

        failure_type = None
        reason = None
        saved_visualization = None

        # 1. YOLO漏检：bad图，但是没有mask
        if status == "bad" and pred_has_mask == "0":
            failure_type = "Detection missed"
            reason = "YOLO did not generate a valid bounding box, so SAM2 received no prompt."
            saved_visualization = copy_visualization(r, "detection_missed")
            detection_missed.append(r)

        # 2. 分割较差：bad图，有mask，但Dice较低
        elif status == "bad" and pred_has_mask == "1" and dice < POOR_DICE_THRESHOLD:
            failure_type = "Poor segmentation"
            reason = "YOLO generated a box, but SAM2 mask has low overlap with ground truth."
            saved_visualization = copy_visualization(r, "poor_segmentation")
            segmentation_poor.append(r)

        # 3. 误检：good图，但是生成了mask
        elif status == "good" and pred_has_mask == "1":
            failure_type = "False positive"
            reason = "The image is normal, but YOLO predicted a defect box and SAM2 generated a mask."
            saved_visualization = copy_visualization(r, "false_positive")
            false_positive.append(r)

        if failure_type is not None:
            failure_rows.append({
                "image": r["image"],
                "category": r["category"],
                "status": r["status"],
                "failure_type": failure_type,
                "iou": iou,
                "dice": dice,
                "boundary_f1": bf1,
                "reason": reason,
                "visualization": saved_visualization,
            })

    # 按类别统计
    category_stats = defaultdict(lambda: {
        "detection_missed": 0,
        "poor_segmentation": 0,
        "false_positive": 0,
    })

    for r in failure_rows:
        cat = r["category"]
        if r["failure_type"] == "Detection missed":
            category_stats[cat]["detection_missed"] += 1
        elif r["failure_type"] == "Poor segmentation":
            category_stats[cat]["poor_segmentation"] += 1
        elif r["failure_type"] == "False positive":
            category_stats[cat]["false_positive"] += 1

    # 保存 CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "image",
            "category",
            "status",
            "failure_type",
            "iou",
            "dice",
            "boundary_f1",
            "reason",
            "visualization",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failure_rows)

    # 保存 TXT 报告
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("Failure Case Analysis Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total failure cases: {len(failure_rows)}\n")
        f.write(f"Detection missed cases: {len(detection_missed)}\n")
        f.write(f"Poor segmentation cases (Dice < {POOR_DICE_THRESHOLD}): {len(segmentation_poor)}\n")
        f.write(f"False positive cases: {len(false_positive)}\n")
        f.write("\n")

        f.write("Per-category failure statistics:\n")
        f.write("-" * 60 + "\n")
        for cat in sorted(category_stats.keys()):
            f.write(f"{cat}:\n")
            f.write(f"  Detection missed: {category_stats[cat]['detection_missed']}\n")
            f.write(f"  Poor segmentation: {category_stats[cat]['poor_segmentation']}\n")
            f.write(f"  False positive: {category_stats[cat]['false_positive']}\n")
            f.write("\n")

    print("=" * 80)
    print("Failure Case Analysis Finished")
    print(f"Total failure cases: {len(failure_rows)}")
    print(f"Detection missed cases: {len(detection_missed)}")
    print(f"Poor segmentation cases: {len(segmentation_poor)}")
    print(f"False positive cases: {len(false_positive)}")
    print("-" * 80)
    print(f"Failure CSV saved to: {OUTPUT_CSV}")
    print(f"Failure report saved to: {OUTPUT_TXT}")
    print(f"Copied visualizations saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()