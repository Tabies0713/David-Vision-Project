
from pathlib import Path
import csv
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

MODEL_PATH = Path(r"C:\Users\20552\Desktop\seg1.pt")
TEST_DIR = Path(r"C:\Users\20552\Desktop\dataset_seg_fixed\images\test")
RAW_ROOT = Path(r"C:\Users\20552\Desktop\dataset2")
OUTPUT_DIR = Path(r"C:\Users\20552\Desktop\yoloseg_evaluation")
MASK_DIR = OUTPUT_DIR / "masks"
CSV_PATH = OUTPUT_DIR / "yoloseg_metric_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "yoloseg_summary.txt"

DEVICE = "cpu"
IMGSZ = 640
CONF = 0.25
BOUNDARY_TOLERANCE = 2

CATEGORIES = [
    "sheet_metal", "fruit_jelly", "wallplugs",
    "can", "fabric", "vial"
]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def sync():
    if DEVICE != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()


def parse_name(name):
    stem = Path(name).stem
    for cat in CATEGORIES:
        p_bad = f"{cat}_bad_"
        p_good = f"{cat}_good_"
        if stem.startswith(p_bad):
            return cat, "bad", stem[len(p_bad):]
        if stem.startswith(p_good):
            return cat, "good", stem[len(p_good):]
    return None, None, stem


def find_gt(name):
    cat, status, original_stem = parse_name(name)
    if status == "good" or cat is None:
        return None

    gt_dir = RAW_ROOT / cat / cat / "test_public" / "ground_truth" / "bad"
    if not gt_dir.exists():
        return None

    candidates = []
    for ext in EXTS:
        candidates += [
            gt_dir / f"{original_stem}{ext}",
            gt_dir / f"{original_stem}_mask{ext}",
            gt_dir / f"{original_stem}_gt{ext}",
        ]

    base_id = original_stem.split("_")[0]
    for ext in EXTS:
        candidates += [
            gt_dir / f"{base_id}{ext}",
            gt_dir / f"{base_id}_mask{ext}",
            gt_dir / f"{base_id}_gt{ext}",
        ]

    for p in candidates:
        if p.exists():
            return p

    masks = [p for p in gt_dir.iterdir()
             if p.is_file() and p.suffix.lower() in EXTS]

    for p in masks:
        if p.stem == original_stem:
            return p
    for p in masks:
        if p.stem.startswith(original_stem):
            return p
    for p in masks:
        if p.stem.startswith(base_id):
            return p

    return None


def read_mask(path, shape):
    if path is None or not path.exists():
        return np.zeros(shape, dtype=np.uint8)

    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros(shape, dtype=np.uint8)

    if mask.shape != shape:
        mask = cv2.resize(
            mask,
            (shape[1], shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    return (mask > 0).astype(np.uint8)


def pred_mask_from_result(result, shape):
    if result.masks is None or result.masks.data is None:
        return np.zeros(shape, dtype=np.uint8), 0

    masks = result.masks.data.cpu().numpy()
    if len(masks) == 0:
        return np.zeros(shape, dtype=np.uint8), 0

    merged = np.any(masks > 0.5, axis=0).astype(np.uint8)
    if merged.shape != shape:
        merged = cv2.resize(
            merged,
            (shape[1], shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    return merged, len(masks)


def iou(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return 1.0 if union == 0 else inter / union


def dice(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()
    return 1.0 if total == 0 else 2.0 * inter / total


def boundary(mask):
    mask = mask.astype(np.uint8)
    if mask.sum() == 0:
        return np.zeros_like(mask)
    kernel = np.ones((3, 3), np.uint8)
    return mask - cv2.erode(mask, kernel, iterations=1)


def boundary_f1(pred, gt, tolerance=2):
    pb = boundary(pred)
    gb = boundary(gt)

    if pb.sum() == 0 and gb.sum() == 0:
        return 1.0
    if pb.sum() == 0 or gb.sum() == 0:
        return 0.0

    kernel = np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8)
    pd = cv2.dilate(pb, kernel, iterations=1)
    gd = cv2.dilate(gb, kernel, iterations=1)

    precision = np.logical_and(pb, gd).sum() / (pb.sum() + 1e-8)
    recall = np.logical_and(gb, pd).sum() / (gb.sum() + 1e-8)

    return 0.0 if precision + recall == 0 else (
        2 * precision * recall / (precision + recall)
    )


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test folder not found: {TEST_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p for p in TEST_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in EXTS
    )

    model = YOLO(str(MODEL_PATH))

    print("Warming up...")
    for _ in range(3):
        model.predict(
            source=str(images[0]),
            imgsz=IMGSZ,
            conf=CONF,
            device=DEVICE,
            save=False,
            verbose=False,
        )
    sync()

    rows = []
    total_time = 0.0
    images_with_masks = 0
    total_instances = 0
    bad_count = 0
    gt_found = 0
    gt_missing = 0
    bad_missed = 0
    good_fp = 0

    for idx, image_path in enumerate(images, 1):
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        h, w = image.shape[:2]

        sync()
        start = time.perf_counter()
        result = model.predict(
            source=str(image_path),
            imgsz=IMGSZ,
            conf=CONF,
            device=DEVICE,
            save=False,
            verbose=False,
        )[0]
        sync()

        infer_time = time.perf_counter() - start
        total_time += infer_time

        pred, instances = pred_mask_from_result(result, (h, w))
        pred_has_mask = int(pred.sum() > 0)

        if pred_has_mask:
            images_with_masks += 1
        total_instances += instances

        cat, status, _ = parse_name(image_path.name)
        gt_path = find_gt(image_path.name)
        gt = read_mask(gt_path, (h, w))

        if status == "bad":
            bad_count += 1
            if gt_path is None:
                gt_missing += 1
            else:
                gt_found += 1
            if pred_has_mask == 0:
                bad_missed += 1
        elif status == "good" and pred_has_mask == 1:
            good_fp += 1

        score_iou = iou(pred, gt)
        score_dice = dice(pred, gt)
        score_bf1 = boundary_f1(pred, gt, BOUNDARY_TOLERANCE)

        cv2.imwrite(
            str(MASK_DIR / f"{image_path.stem}_yoloseg_mask.png"),
            pred * 255
        )

        rows.append({
            "image": image_path.name,
            "category": cat,
            "status": status,
            "gt_found": int(gt_path is not None),
            "pred_has_mask": pred_has_mask,
            "num_instances": instances,
            "iou": score_iou,
            "dice": score_dice,
            "boundary_f1": score_bf1,
            "inference_time_sec": infer_time,
        })

        print(
            f"[{idx:03d}/{len(images)}] {image_path.name} | "
            f"instances={instances} | IoU={score_iou:.4f} | "
            f"Dice={score_dice:.4f} | BF1={score_bf1:.4f} | "
            f"time={infer_time:.4f}s"
        )

    with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    all_iou = np.mean([r["iou"] for r in rows])
    all_dice = np.mean([r["dice"] for r in rows])
    all_bf1 = np.mean([r["boundary_f1"] for r in rows])

    defect_rows = [
        r for r in rows
        if r["status"] == "bad" and r["gt_found"] == 1
    ]

    defect_iou = np.mean([r["iou"] for r in defect_rows])
    defect_dice = np.mean([r["dice"] for r in defect_rows])
    defect_bf1 = np.mean([r["boundary_f1"] for r in defect_rows])

    avg_time = total_time / len(rows)
    fps = 1.0 / avg_time if avg_time > 0 else 0.0

    summary = f"""
======================================================================
YOLO26m-seg Evaluation Finished
======================================================================
Device: {DEVICE}
Total test images: {len(rows)}
Bad images: {bad_count}
GT found: {gt_found}
GT missing: {gt_missing}
Images with predicted masks: {images_with_masks}
Total predicted instances: {total_instances}
Bad images missed: {bad_missed}
Good-image false positives: {good_fp}
----------------------------------------------------------------------
All images:
mIoU: {all_iou:.4f}
Dice: {all_dice:.4f}
Boundary-F1: {all_bf1:.4f}
----------------------------------------------------------------------
Defect images only:
mIoU: {defect_iou:.4f}
Dice: {defect_dice:.4f}
Boundary-F1: {defect_bf1:.4f}
----------------------------------------------------------------------
Inference time:
Total inference time: {total_time:.4f} s
Average inference time: {avg_time:.4f} s/image
FPS: {fps:.2f}
----------------------------------------------------------------------
CSV: {CSV_PATH}
Predicted masks: {MASK_DIR}
======================================================================
""".strip()

    SUMMARY_PATH.write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
