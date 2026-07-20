from pathlib import Path
import csv
import time

import cv2
import numpy as np
import torch
from ultralytics import SAM


# ===================== Configuration =====================

TEST_IMAGE_DIR = Path(
    r"C:\Users\20552\Desktop\dataset\images\test"
)

RAW_DATASET_ROOT = Path(
    r"C:\Users\20552\Desktop\dataset2"
)

SAM2_MODEL_PATH = Path(
    r"C:\Users\20552\Desktop\sam2.1_t.pt"
)

OUTPUT_ROOT = Path(
    r"C:\Users\20552\Desktop\gtbox_sam2_results"
)

OUTPUT_MASK_DIR = OUTPUT_ROOT / "masks"
OUTPUT_CSV = OUTPUT_ROOT / "gtbox_sam2_metric_results.csv"
OUTPUT_SUMMARY = OUTPUT_ROOT / "gtbox_sam2_summary.txt"

DEVICE = "cpu"
BOUNDARY_TOLERANCE = 2
MIN_COMPONENT_AREA = 10
WARMUP_RUNS = 1

CATEGORIES = [
    "sheet_metal",
    "fruit_jelly",
    "wallplugs",
    "can",
    "fabric",
    "vial",
]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}

# =========================================================


def sync_device():
    if DEVICE != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()


def parse_image_name(image_name: str):
    """
    Example:
      can_bad_001_underexposed.jpg
      -> category = can
      -> status = bad
      -> original_stem = 001_underexposed
    """
    stem = Path(image_name).stem

    for category in CATEGORIES:
        bad_prefix = f"{category}_bad_"
        good_prefix = f"{category}_good_"

        if stem.startswith(bad_prefix):
            return category, "bad", stem[len(bad_prefix):]

        if stem.startswith(good_prefix):
            return category, "good", stem[len(good_prefix):]

    return None, None, stem


def find_gt_mask(image_name: str):
    category, status, original_stem = parse_image_name(image_name)

    if status == "good" or category is None:
        return None

    gt_dir = (
        RAW_DATASET_ROOT
        / category
        / category
        / "test_public"
        / "ground_truth"
        / "bad"
    )

    if not gt_dir.exists():
        return None

    candidates = []

    for ext in IMAGE_EXTENSIONS:
        candidates.extend([
            gt_dir / f"{original_stem}{ext}",
            gt_dir / f"{original_stem}_mask{ext}",
            gt_dir / f"{original_stem}_gt{ext}",
            gt_dir / f"{original_stem}_ground_truth{ext}",
        ])

    base_id = original_stem.split("_")[0]

    for ext in IMAGE_EXTENSIONS:
        candidates.extend([
            gt_dir / f"{base_id}{ext}",
            gt_dir / f"{base_id}_mask{ext}",
            gt_dir / f"{base_id}_gt{ext}",
        ])

    for path in candidates:
        if path.exists():
            return path

    masks = [
        path for path in gt_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    for path in masks:
        if path.stem == original_stem:
            return path

    for path in masks:
        if path.stem.startswith(original_stem):
            return path

    for path in masks:
        if path.stem.startswith(base_id):
            return path

    return None


def read_binary_mask(path: Path | None, target_shape):
    if path is None or not path.exists():
        return np.zeros(target_shape, dtype=np.uint8)

    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        return np.zeros(target_shape, dtype=np.uint8)

    if mask.shape != target_shape:
        mask = cv2.resize(
            mask,
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    return (mask > 0).astype(np.uint8)


def gt_mask_to_boxes(gt_mask):
    """
    Generate one GT bounding box for each connected defect component.
    Output format: [[x1, y1, x2, y2], ...]
    """
    binary = (gt_mask > 0).astype(np.uint8)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    boxes = []

    for component_id in range(1, num_labels):
        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        w = int(stats[component_id, cv2.CC_STAT_WIDTH])
        h = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[component_id, cv2.CC_STAT_AREA])

        if area < MIN_COMPONENT_AREA:
            continue

        x2 = x + w - 1
        y2 = y + h - 1
        boxes.append([x, y, x2, y2])

    return boxes


def sam_result_to_binary_mask(result, target_shape):
    """
    Merge all SAM2 instance masks into one binary defect mask.
    """
    if result is None:
        return np.zeros(target_shape, dtype=np.uint8), 0

    if result.masks is None or result.masks.data is None:
        return np.zeros(target_shape, dtype=np.uint8), 0

    masks = result.masks.data.cpu().numpy()

    if masks.size == 0:
        return np.zeros(target_shape, dtype=np.uint8), 0

    merged = np.any(masks > 0.5, axis=0).astype(np.uint8)

    if merged.shape != target_shape:
        merged = cv2.resize(
            merged,
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    return merged, len(masks)


def compute_iou(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        return 1.0

    return intersection / union


def compute_dice(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()

    if total == 0:
        return 1.0

    return 2.0 * intersection / total


def get_boundary(mask):
    mask = mask.astype(np.uint8)

    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)

    return mask - eroded


def compute_boundary_f1(pred, gt, tolerance=2):
    pred_boundary = get_boundary(pred)
    gt_boundary = get_boundary(gt)

    if pred_boundary.sum() == 0 and gt_boundary.sum() == 0:
        return 1.0

    if pred_boundary.sum() == 0 or gt_boundary.sum() == 0:
        return 0.0

    kernel = np.ones(
        (2 * tolerance + 1, 2 * tolerance + 1),
        dtype=np.uint8,
    )

    pred_dilated = cv2.dilate(pred_boundary, kernel, iterations=1)
    gt_dilated = cv2.dilate(gt_boundary, kernel, iterations=1)

    precision = (
        np.logical_and(pred_boundary, gt_dilated).sum()
        / (pred_boundary.sum() + 1e-8)
    )

    recall = (
        np.logical_and(gt_boundary, pred_dilated).sum()
        / (gt_boundary.sum() + 1e-8)
    )

    if precision + recall == 0:
        return 0.0

    return 2.0 * precision * recall / (precision + recall)


def main():
    if not TEST_IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Test image directory not found: {TEST_IMAGE_DIR}"
        )

    if not RAW_DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Raw dataset root not found: {RAW_DATASET_ROOT}"
        )

    if not SAM2_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"SAM2 model not found: {SAM2_MODEL_PATH}"
        )

    OUTPUT_MASK_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path for path in TEST_IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise RuntimeError(
            f"No test images found in: {TEST_IMAGE_DIR}"
        )

    sam2_model = SAM(str(SAM2_MODEL_PATH))

    # Find the first bad image for warm-up.
    warmup_image = None
    warmup_boxes = None

    for image_path in image_paths:
        _, status, _ = parse_image_name(image_path.name)

        if status != "bad":
            continue

        image = cv2.imread(str(image_path))
        gt_path = find_gt_mask(image_path.name)

        if image is None or gt_path is None:
            continue

        h, w = image.shape[:2]
        gt_mask = read_binary_mask(gt_path, (h, w))
        boxes = gt_mask_to_boxes(gt_mask)

        if boxes:
            warmup_image = image_path
            warmup_boxes = boxes
            break

    if warmup_image is None:
        raise RuntimeError("No valid bad image was found for warm-up.")

    print("Warming up SAM2...")

    for _ in range(WARMUP_RUNS):
        _ = sam2_model(
            source=str(warmup_image),
            bboxes=warmup_boxes,
            device=DEVICE,
            save=False,
            verbose=False,
        )

    sync_device()
    print("Warm-up finished.")

    rows = []

    total_sam2_time = 0.0
    bad_count = 0
    gt_found_count = 0
    gt_missing_count = 0
    total_gt_boxes = 0
    images_with_gt_boxes = 0
    images_with_pred_masks = 0
    sam2_failed_count = 0

    for index, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[Skipped] Cannot read image: {image_path}")
            continue

        height, width = image.shape[:2]
        target_shape = (height, width)

        category, status, _ = parse_image_name(image_path.name)
        gt_path = find_gt_mask(image_path.name)
        gt_mask = read_binary_mask(gt_path, target_shape)

        boxes = []
        pred_mask = np.zeros(target_shape, dtype=np.uint8)
        predicted_instances = 0
        sam2_time = 0.0

        if status == "bad":
            bad_count += 1

            if gt_path is None:
                gt_missing_count += 1
            else:
                gt_found_count += 1
                boxes = gt_mask_to_boxes(gt_mask)

        if boxes:
            images_with_gt_boxes += 1
            total_gt_boxes += len(boxes)

            sync_device()
            start_time = time.perf_counter()

            sam_results = sam2_model(
                source=str(image_path),
                bboxes=boxes,
                device=DEVICE,
                save=False,
                verbose=False,
            )

            sync_device()
            sam2_time = time.perf_counter() - start_time
            total_sam2_time += sam2_time

            if sam_results and len(sam_results) > 0:
                pred_mask, predicted_instances = (
                    sam_result_to_binary_mask(
                        sam_results[0],
                        target_shape,
                    )
                )

            if pred_mask.sum() == 0:
                sam2_failed_count += 1
            else:
                images_with_pred_masks += 1

        score_iou = compute_iou(pred_mask, gt_mask)
        score_dice = compute_dice(pred_mask, gt_mask)
        score_bf1 = compute_boundary_f1(
            pred_mask,
            gt_mask,
            BOUNDARY_TOLERANCE,
        )

        output_mask_path = (
            OUTPUT_MASK_DIR / f"{image_path.stem}_gtbox_sam2_mask.png"
        )

        cv2.imwrite(str(output_mask_path), pred_mask * 255)

        rows.append({
            "image": image_path.name,
            "category": category,
            "status": status,
            "gt_found": int(gt_path is not None),
            "num_gt_boxes": len(boxes),
            "pred_has_mask": int(pred_mask.sum() > 0),
            "num_pred_instances": predicted_instances,
            "iou": score_iou,
            "dice": score_dice,
            "boundary_f1": score_bf1,
            "sam2_time_sec": sam2_time,
        })

        print(
            f"[{index:03d}/{len(image_paths)}] {image_path.name} | "
            f"GT boxes={len(boxes)} | "
            f"IoU={score_iou:.4f} | "
            f"Dice={score_dice:.4f} | "
            f"BF1={score_bf1:.4f} | "
            f"SAM2={sam2_time:.4f}s"
        )

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    all_iou = np.mean([row["iou"] for row in rows])
    all_dice = np.mean([row["dice"] for row in rows])
    all_bf1 = np.mean([row["boundary_f1"] for row in rows])

    defect_rows = [
        row for row in rows
        if row["status"] == "bad" and row["gt_found"] == 1
    ]

    defect_iou = np.mean([row["iou"] for row in defect_rows])
    defect_dice = np.mean([row["dice"] for row in defect_rows])
    defect_bf1 = np.mean(
        [row["boundary_f1"] for row in defect_rows]
    )

    total_images = len(rows)
    average_sam2_all = total_sam2_time / total_images

    average_sam2_bad = (
        total_sam2_time / images_with_gt_boxes
        if images_with_gt_boxes > 0
        else 0.0
    )

    summary_lines = [
        "=" * 72,
        "GT-box -> SAM2 Evaluation Finished",
        "=" * 72,
        f"Device: {DEVICE}",
        f"Total test images: {total_images}",
        f"Bad images: {bad_count}",
        f"GT found: {gt_found_count}",
        f"GT missing: {gt_missing_count}",
        f"Images with GT boxes: {images_with_gt_boxes}",
        f"Total GT boxes: {total_gt_boxes}",
        f"Images with predicted masks: {images_with_pred_masks}",
        f"SAM2 failed cases: {sam2_failed_count}",
        "-" * 72,
        "All images:",
        f"mIoU: {all_iou:.4f}",
        f"Dice: {all_dice:.4f}",
        f"Boundary-F1: {all_bf1:.4f}",
        "-" * 72,
        "Defect images only:",
        f"mIoU: {defect_iou:.4f}",
        f"Dice: {defect_dice:.4f}",
        f"Boundary-F1: {defect_bf1:.4f}",
        "-" * 72,
        "SAM2 inference time with GT-box prompts:",
        f"Total SAM2 time: {total_sam2_time:.4f} s",
        f"Average over all 169 images: {average_sam2_all:.4f} s/image",
        f"Average over prompted bad images: {average_sam2_bad:.4f} s/image",
        "-" * 72,
        f"Detailed CSV: {OUTPUT_CSV}",
        f"Predicted masks: {OUTPUT_MASK_DIR}",
        "=" * 72,
    ]

    summary_text = "\n".join(summary_lines)

    OUTPUT_SUMMARY.write_text(
        summary_text,
        encoding="utf-8",
    )

    print(summary_text)


if __name__ == "__main__":
    main()
