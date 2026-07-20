from pathlib import Path
import csv

CSV_PATH = Path(r"C:\Users\20552\Desktop\sam2_metric_results.csv")
VIS_DIR = Path(r"C:\Users\20552\Desktop\sam2_results\visualizations")

with open(CSV_PATH, "r", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

# 只看 bad 图，因为展示缺陷分割最好选 bad 图
bad_rows = [r for r in rows if r["status"] == "bad"]

for r in bad_rows:
    r["iou"] = float(r["iou"])
    r["dice"] = float(r["dice"])
    r["boundary_f1"] = float(r["boundary_f1"])

# 效果好：按 Dice 从高到低
good_cases = sorted(
    [r for r in bad_rows if r["pred_has_mask"] == "1"],
    key=lambda x: x["dice"],
    reverse=True
)

# 失败：优先选没有 mask 的，其次选 Dice 最低的
failed_cases = [r for r in bad_rows if r["pred_has_mask"] == "0"]
if len(failed_cases) == 0:
    failed_cases = sorted(bad_rows, key=lambda x: x["dice"])

# 一般：Dice 接近 0.5~0.65 的
middle_cases = sorted(
    [r for r in bad_rows if r["pred_has_mask"] == "1"],
    key=lambda x: abs(x["dice"] - 0.55)
)

def print_case(title, case):
    stem = Path(case["image"]).stem
    vis_path = VIS_DIR / f"{stem}_comparison.jpg"

    print("=" * 80)
    print(title)
    print(f"Image: {case['image']}")
    print(f"Category: {case['category']}")
    print(f"pred_has_mask: {case['pred_has_mask']}")
    print(f"IoU: {case['iou']:.4f}")
    print(f"Dice: {case['dice']:.4f}")
    print(f"Boundary-F1: {case['boundary_f1']:.4f}")
    print(f"Visualization: {vis_path}")

print_case("GOOD CASE", good_cases[0])
print_case("MIDDLE CASE", middle_cases[0])
print_case("FAILED CASE", failed_cases[0])