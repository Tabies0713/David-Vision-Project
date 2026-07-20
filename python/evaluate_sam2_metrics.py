from pathlib import Path
import cv2
import numpy as np
import csv

# ================== 路径配置 ==================

# 合成后的 YOLO 数据集
YOLO_DATASET_ROOT = Path(r"C:\Users\20552\Desktop\dataset")

# 原始 6 个数据集，里面有 ground_truth
RAW_DATASET_ROOT = Path(r"C:\Users\20552\Desktop\dataset2")

# 测试图片路径
IMAGE_DIR = YOLO_DATASET_ROOT / "images" / "test"

# SAM2 生成的预测 mask
PRED_MASK_DIR = Path(r"C:\Users\20552\Desktop\sam2_results\masks")

# 输出结果
OUTPUT_CSV = Path(r"C:\Users\20552\Desktop\sam2_metric_results.csv")

# 你的 6 个类别
CATEGORIES = [
    "sheet_metal",
    "fruit_jelly",
    "wallplugs",
    "can",
    "fabric",
    "vial",
]

BOUNDARY_TOLERANCE = 2

# =============================================


def read_binary_mask(path, target_shape):
    """
    读取 mask，并转成 0/1 二值图。
    如果 path 不存在，则返回空 mask。
    """
    if path is None or not path.exists():
        return np.zeros(target_shape, dtype=np.uint8)

    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        return np.zeros(target_shape, dtype=np.uint8)

    if mask.shape != target_shape:
        mask = cv2.resize(
            mask,
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    mask = (mask > 0).astype(np.uint8)
    return mask


def parse_image_name(image_name):
    """
    解析合成后的图片名。

    例如：
    can_bad_001_underexposed.jpg
    → cat = can
    → status = bad
    → original_stem = 001_underexposed
    """
    stem = Path(image_name).stem

    for cat in CATEGORIES:
        if stem.startswith(cat + "_bad_"):
            original_stem = stem.replace(cat + "_bad_", "", 1)
            return cat, "bad", original_stem

        if stem.startswith(cat + "_good_"):
            original_stem = stem.replace(cat + "_good_", "", 1)
            return cat, "good", original_stem

    return None, None, stem


def find_gt_mask(image_name):
    """
    在原始 dataset2 里面寻找 ground_truth mask。
    """
    cat, status, original_stem = parse_image_name(image_name)

    if status == "good":
        return None

    if cat is None:
        return None

    gt_root = RAW_DATASET_ROOT / cat / cat / "test_public" / "ground_truth"

    if not gt_root.exists():
        return None

    # 常见情况：ground_truth/bad/xxx.png
    search_patterns = [
        f"{original_stem}.*",
        f"{original_stem}_mask.*",
        f"*{original_stem}*",
    ]

    for pattern in search_patterns:
        matches = list(gt_root.rglob(pattern))
        if len(matches) > 0:
            return matches[0]

    return None


def compute_iou(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        return 1.0

    return inter / union


def compute_dice(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()

    if total == 0:
        return 1.0

    return 2 * inter / total


def get_boundary(mask):
    mask = mask.astype(np.uint8)

    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)
    boundary = mask - eroded

    return boundary


def compute_boundary_f1(pred, gt, tolerance=2):
    pred_boundary = get_boundary(pred)
    gt_boundary = get_boundary(gt)

    if pred_boundary.sum() == 0 and gt_boundary.sum() == 0:
        return 1.0

    if pred_boundary.sum() == 0 or gt_boundary.sum() == 0:
        return 0.0

    kernel = np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8)

    pred_dilated = cv2.dilate(pred_boundary, kernel, iterations=1)
    gt_dilated = cv2.dilate(gt_boundary, kernel, iterations=1)

    precision = np.logical_and(pred_boundary, gt_dilated).sum() / (pred_boundary.sum() + 1e-8)
    recall = np.logical_and(gt_boundary, pred_dilated).sum() / (gt_boundary.sum() + 1e-8)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def main():
    image_paths = []

    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]:
        image_paths.extend(list(IMAGE_DIR.glob(ext)))

    results = []

    gt_found_count = 0
    gt_missing_count = 0
    bad_image_count = 0

    for img_path in sorted(image_paths):
        image_name = img_path.name
        stem = img_path.stem

        cat, status, original_stem = parse_image_name(image_name)

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        target_shape = (h, w)

        pred_mask_path = PRED_MASK_DIR / f"{stem}_sam2_mask.png"
        gt_mask_path = find_gt_mask(image_name)

        if status == "bad":
            bad_image_count += 1
            if gt_mask_path is not None:
                gt_found_count += 1
            else:
                gt_missing_count += 1
                print(f"[GT missing] {image_name}")

        pred = read_binary_mask(pred_mask_path, target_shape)
        gt = read_binary_mask(gt_mask_path, target_shape)

        iou = compute_iou(pred, gt)
        dice = compute_dice(pred, gt)
        bf1 = compute_boundary_f1(pred, gt, BOUNDARY_TOLERANCE)

        results.append({
            "image": image_name,
            "category": cat,
            "status": status,
            "gt_found": int(gt_mask_path is not None),
            "pred_has_mask": int(pred.sum() > 0),
            "iou": iou,
            "dice": dice,
            "boundary_f1": bf1,
        })

    all_iou = np.mean([r["iou"] for r in results])
    all_dice = np.mean([r["dice"] for r in results])
    all_bf1 = np.mean([r["boundary_f1"] for r in results])

    defect_results = [r for r in results if r["status"] == "bad" and r["gt_found"] == 1]

    if len(defect_results) > 0:
        defect_iou = np.mean([r["iou"] for r in defect_results])
        defect_dice = np.mean([r["dice"] for r in defect_results])
        defect_bf1 = np.mean([r["boundary_f1"] for r in defect_results])
    else:
        defect_iou = 0
        defect_dice = 0
        defect_bf1 = 0

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "category",
                "status",
                "gt_found",
                "pred_has_mask",
                "iou",
                "dice",
                "boundary_f1",
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    print("=" * 60)
    print("SAM2 Metric Evaluation Finished")
    print(f"Total test images: {len(results)}")
    print(f"Bad images: {bad_image_count}")
    print(f"GT found: {gt_found_count}")
    print(f"GT missing: {gt_missing_count}")
    print("-" * 60)
    print("All images:")
    print(f"mIoU: {all_iou:.4f}")
    print(f"Dice: {all_dice:.4f}")
    print(f"Boundary-F1: {all_bf1:.4f}")
    print("-" * 60)
    print("Defect images only:")
    print(f"mIoU: {defect_iou:.4f}")
    print(f"Dice: {defect_dice:.4f}")
    print(f"Boundary-F1: {defect_bf1:.4f}")
    print("-" * 60)
    print(f"Detailed results saved to: {OUTPUT_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()