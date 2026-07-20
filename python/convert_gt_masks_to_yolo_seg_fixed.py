from pathlib import Path
import os
import stat
import shutil
import cv2
import numpy as np

# ===================== 路径配置 =====================

# 已经划分好的 YOLO 检测数据集，继续沿用同一 train/val/test 划分
SOURCE_IMAGE_ROOT = Path(r"C:\Users\20552\Desktop\dataset\images")

# 原始 6 类数据集，其中包含 test_public/ground_truth/bad
RAW_DATASET_ROOT = Path(r"C:\Users\20552\Desktop\dataset2")

# 新的 YOLO segmentation 数据集
OUTPUT_ROOT = Path(r"C:\Users\20552\Desktop\dataset_seg_fixed")

CATEGORIES = [
    "sheet_metal",
    "fruit_jelly",
    "wallplugs",
    "can",
    "fabric",
    "vial",
]

CLASS_ID = 0
CLASS_NAME = "defect"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# 与原 bbox 转换脚本保持一致：忽略面积小于 10 像素的噪声区域
MIN_COMPONENT_AREA = 10

# 多边形简化程度；越小越贴近原 mask 边界
APPROX_EPSILON_RATIO = 0.001

# ====================================================


def remove_readonly(func, path, _):
    """Windows 下删除只读文件时使用。"""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def reset_output_folder():
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT, onerror=remove_readonly)

    for split in ["train", "val", "test"]:
        (OUTPUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)


def list_images(folder: Path):
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def parse_merged_image_name(image_name: str):
    """
    还原 transfer_format.py 的重命名规则。

    例如：
      can_bad_000_regular.png
        -> category = can
        -> status = bad
        -> original_stem = 000_regular

      sheet_metal_good_005.png
        -> category = sheet_metal
        -> status = good
        -> original_stem = 005
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


def find_gt_mask(category: str, original_stem: str):
    """
    使用 transfer_format.py 中相同的 GT 匹配逻辑。

    先找完整原始 stem：
      000_regular.png
      000_regular_mask.png

    找不到时再用 base_id：
      000_regular -> 000
      009_shift_3 -> 009
    """
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

    # 1. 完整 stem
    for ext in IMAGE_EXTENSIONS:
        candidates.extend([
            gt_dir / f"{original_stem}{ext}",
            gt_dir / f"{original_stem}_mask{ext}",
            gt_dir / f"{original_stem}_gt{ext}",
            gt_dir / f"{original_stem}_ground_truth{ext}",
        ])

    # 2. 去掉 regular / shift / exposure 等后缀，仅保留原编号
    base_id = original_stem.split("_")[0]

    for ext in IMAGE_EXTENSIONS:
        candidates.extend([
            gt_dir / f"{base_id}{ext}",
            gt_dir / f"{base_id}_mask{ext}",
            gt_dir / f"{base_id}_gt{ext}",
        ])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # 3. 模糊匹配兜底
    masks = list_images(gt_dir)

    for mask_path in masks:
        if mask_path.stem == original_stem:
            return mask_path

    for mask_path in masks:
        if mask_path.stem.startswith(original_stem):
            return mask_path

    for mask_path in masks:
        if mask_path.stem.startswith(base_id):
            return mask_path

    return None


def read_image_unicode(path: Path, flags):
    """兼容特殊路径的 OpenCV 读取方式。"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def mask_to_yolo_polygons(mask_path: Path, image_path: Path):
    """
    将二值 GT mask 转成 YOLO instance segmentation polygon 标签。

    每一个独立缺陷区域输出一行：
      class_id x1 y1 x2 y2 x3 y3 ...
    所有坐标归一化到 0~1。
    """
    image = read_image_unicode(image_path, cv2.IMREAD_COLOR)
    mask = read_image_unicode(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    if mask is None:
        raise RuntimeError(f"Cannot read mask: {mask_path}")

    image_h, image_w = image.shape[:2]

    if mask.shape[:2] != (image_h, image_w):
        mask = cv2.resize(
            mask,
            (image_w, image_h),
            interpolation=cv2.INTER_NEAREST,
        )

    binary = (mask > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    yolo_lines = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < MIN_COMPONENT_AREA:
            continue

        perimeter = cv2.arcLength(contour, True)
        epsilon = APPROX_EPSILON_RATIO * perimeter

        polygon = cv2.approxPolyDP(
            contour,
            epsilon,
            True,
        ).reshape(-1, 2)

        if len(polygon) < 3:
            continue

        coordinates = []

        for x, y in polygon:
            x_norm = min(max(float(x) / image_w, 0.0), 1.0)
            y_norm = min(max(float(y) / image_h, 0.0), 1.0)
            coordinates.extend([x_norm, y_norm])

        line = str(CLASS_ID) + " " + " ".join(
            f"{value:.6f}" for value in coordinates
        )
        yolo_lines.append(line)

    return yolo_lines


def process_split(split: str):
    source_dir = SOURCE_IMAGE_ROOT / split
    output_image_dir = OUTPUT_ROOT / "images" / split
    output_label_dir = OUTPUT_ROOT / "labels" / split

    image_paths = list_images(source_dir)

    stats = {
        "images": len(image_paths),
        "good": 0,
        "bad": 0,
        "matched_gt": 0,
        "missing_gt": [],
        "empty_bad_masks": [],
        "instances": 0,
    }

    for index, image_path in enumerate(image_paths, start=1):
        category, status, original_stem = parse_merged_image_name(image_path.name)

        if category is None or status is None:
            raise RuntimeError(
                f"Cannot parse merged image name: {image_path.name}"
            )

        destination_image = output_image_dir / image_path.name
        destination_label = output_label_dir / f"{image_path.stem}.txt"

        # 不复制只读属性，避免 PermissionError
        shutil.copyfile(image_path, destination_image)

        if status == "good":
            stats["good"] += 1
            destination_label.write_text("", encoding="utf-8")
            continue

        stats["bad"] += 1

        gt_path = find_gt_mask(category, original_stem)

        if gt_path is None:
            stats["missing_gt"].append(image_path.name)
            destination_label.write_text("", encoding="utf-8")
            continue

        polygon_lines = mask_to_yolo_polygons(gt_path, image_path)

        if not polygon_lines:
            stats["empty_bad_masks"].append(
                f"{image_path.name} -> {gt_path}"
            )
            destination_label.write_text("", encoding="utf-8")
            continue

        destination_label.write_text(
            "\n".join(polygon_lines) + "\n",
            encoding="utf-8",
        )

        stats["matched_gt"] += 1
        stats["instances"] += len(polygon_lines)

        if index % 100 == 0 or index == len(image_paths):
            print(f"{split}: {index}/{len(image_paths)}")

    return stats


def write_yaml():
    yaml_path = OUTPUT_ROOT / "defect_seg.yaml"

    yaml_text = f"""path: {OUTPUT_ROOT.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  {CLASS_ID}: {CLASS_NAME}
"""

    yaml_path.write_text(yaml_text, encoding="utf-8")
    return yaml_path


def main():
    if not SOURCE_IMAGE_ROOT.exists():
        raise FileNotFoundError(
            f"Source image root not found: {SOURCE_IMAGE_ROOT}"
        )

    if not RAW_DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Raw dataset root not found: {RAW_DATASET_ROOT}"
        )

    reset_output_folder()

    all_stats = {}

    for split in ["train", "val", "test"]:
        print(f"\nProcessing {split}...")
        all_stats[split] = process_split(split)

    yaml_path = write_yaml()

    print("\n" + "=" * 72)
    print("YOLO segmentation dataset conversion finished")
    print("=" * 72)

    has_error = False

    for split, stats in all_stats.items():
        print(
            f"{split}: "
            f"images={stats['images']}, "
            f"good={stats['good']}, "
            f"bad={stats['bad']}, "
            f"matched_gt={stats['matched_gt']}, "
            f"instances={stats['instances']}, "
            f"missing_gt={len(stats['missing_gt'])}, "
            f"empty_bad_masks={len(stats['empty_bad_masks'])}"
        )

        if stats["missing_gt"]:
            has_error = True
            print(f"\nMissing GT in {split}:")
            for name in stats["missing_gt"][:30]:
                print("  ", name)

            if len(stats["missing_gt"]) > 30:
                print(
                    f"  ... and {len(stats['missing_gt']) - 30} more"
                )

        if stats["empty_bad_masks"]:
            has_error = True
            print(f"\nEmpty converted masks in {split}:")
            for item in stats["empty_bad_masks"][:30]:
                print("  ", item)

            if len(stats["empty_bad_masks"]) > 30:
                print(
                    f"  ... and {len(stats['empty_bad_masks']) - 30} more"
                )

    print(f"\nDataset: {OUTPUT_ROOT}")
    print(f"YAML: {yaml_path}")

    if has_error:
        raise RuntimeError(
            "Conversion incomplete. Do not train until "
            "missing/empty masks are fixed."
        )

    print("\nAll bad images matched non-empty GT masks.")
    print("The dataset is ready for YOLO26m-seg training.")


if __name__ == "__main__":
    main()
