from pathlib import Path
import shutil
import random
import csv

import cv2
import numpy as np


# =========================================================
# 1. 路径与参数设置
# =========================================================

# 你的原始数据集路径
ROOT_DIR = Path(r"C:\Users\20552\Desktop\dataset")

# 输出 YOLO 数据集路径
# 注意：脚本不会删除旧文件夹，而是自动新建 mvtec_yolo_defect_fixed / fixed_1 / fixed_2
OUT_BASE = Path(r"C:\Users\20552\Desktop\mvtec_yolo_defect_fixed")

# 6 个类别文件夹
CATEGORIES = [
    "can",
    "fabric",
    "fruit_jelly",
    "sheet_metal",
    "vial",
    "wallplugs",
]

# train / val / test 划分比例
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

# 所有缺陷统一成一个类别
CLASS_ID = 0
CLASS_NAME = "defect"

# 小于这个面积的白色区域会被当作噪声忽略
MIN_COMPONENT_AREA = 10

# 随机种子，保证每次划分一致
RANDOM_SEED = 42

# 支持的图片格式
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# =========================================================
# 2. 工具函数
# =========================================================

def get_available_out_dir(base_dir: Path) -> Path:
    """
    自动生成一个不冲突的输出文件夹。
    比如：
    mvtec_yolo_defect_fixed
    mvtec_yolo_defect_fixed_1
    mvtec_yolo_defect_fixed_2
    """
    if not base_dir.exists():
        return base_dir

    i = 1
    while True:
        candidate = Path(str(base_dir) + f"_{i}")
        if not candidate.exists():
            return candidate
        i += 1


def list_images(folder: Path):
    """列出某个文件夹下的图片"""
    if not folder.exists():
        print(f"[WARNING] Folder does not exist: {folder}")
        return []

    return sorted([
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ])


def split_files(files):
    """按 70 / 10 / 20 划分 train / val / test"""
    files = files[:]
    random.shuffle(files)

    n = len(files)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    return {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:],
    }


def cv_imread_unicode(path: Path, flags=cv2.IMREAD_COLOR):
    """
    兼容中文路径/特殊路径的 OpenCV 读图函数。
    你现在改成 dataset 后一般不会有问题，但保留这个更稳。
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, flags)
        return img
    except Exception:
        return None


def find_mask_for_bad_image(bad_img_path: Path, gt_dir: Path):
    """
    为 bad 图片寻找对应的 ground truth mask。

    现在你的结构应该是：
    dataset/
      can/
        can/
          test_public/
            bad/
            good/
            ground_truth/
              bad/
    """

    stem = bad_img_path.stem
    suffix = bad_img_path.suffix.lower()

    candidates = []

    # 1. 完全同名
    candidates.append(gt_dir / f"{stem}{suffix}")

    # 2. 常见 mask 命名方式
    for ext in IMG_EXTS:
        candidates.append(gt_dir / f"{stem}{ext}")
        candidates.append(gt_dir / f"{stem}_mask{ext}")
        candidates.append(gt_dir / f"{stem}_gt{ext}")
        candidates.append(gt_dir / f"{stem}_ground_truth{ext}")

    # 3. 去掉增强后缀，只保留编号
    # 例如 009_shift_3 -> 009
    # 例如 007_overexposed -> 007
    # 例如 000_regular -> 000
    base_id = stem.split("_")[0]
    for ext in IMG_EXTS:
        candidates.append(gt_dir / f"{base_id}{ext}")
        candidates.append(gt_dir / f"{base_id}_mask{ext}")
        candidates.append(gt_dir / f"{base_id}_gt{ext}")

    for c in candidates:
        if c.exists():
            return c

    # 4. 模糊匹配
    all_masks = list_images(gt_dir)

    for mask_path in all_masks:
        if mask_path.stem == stem:
            return mask_path

    for mask_path in all_masks:
        if mask_path.stem.startswith(stem):
            return mask_path

    for mask_path in all_masks:
        if mask_path.stem.startswith(base_id):
            return mask_path

    return None


def mask_to_yolo_bboxes(mask_path: Path, image_path: Path):
    """
    把 ground_truth 黑白 mask 转成 YOLO bbox。
    每一个独立白色区域生成一个 bbox。
    """

    img = cv_imread_unicode(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    img_h, img_w = img.shape[:2]

    mask = cv_imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Cannot read mask: {mask_path}")

    # 如果 mask 尺寸和原图不一样，缩放到原图大小
    if mask.shape[:2] != (img_h, img_w):
        mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

    # 大于 0 的像素都视为缺陷区域
    binary = (mask > 0).astype(np.uint8) * 255

    # 找每一个独立白色区域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    yolo_lines = []

    # 0 是背景，所以从 1 开始
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < MIN_COMPONENT_AREA:
            continue

        # 转成 YOLO 格式：class x_center y_center width height
        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        box_w = w / img_w
        box_h = h / img_h

        # 防止越界
        x_center = min(max(x_center, 0), 1)
        y_center = min(max(y_center, 0), 1)
        box_w = min(max(box_w, 0), 1)
        box_h = min(max(box_h, 0), 1)

        line = f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"
        yolo_lines.append(line)

    return yolo_lines


def make_output_dirs(out_dir: Path):
    """创建 YOLO 输出文件夹"""
    for split in ["train", "val", "test"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_data_yaml(out_dir: Path):
    """生成 data.yaml"""
    yaml_text = f"""path: {out_dir.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  {CLASS_ID}: {CLASS_NAME}
"""

    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")
    return yaml_path


# =========================================================
# 3. 主程序
# =========================================================

def main():
    random.seed(RANDOM_SEED)

    out_dir = get_available_out_dir(OUT_BASE)
    make_output_dirs(out_dir)

    mapping_rows = []
    summary_rows = []

    total_bad = 0
    total_good = 0
    total_missing_masks = 0
    total_bbox = 0

    print("=" * 80)
    print("Output folder:")
    print(out_dir)
    print("=" * 80)

    for category in CATEGORIES:
        category_root = ROOT_DIR / category / category / "test_public"

        bad_dir = category_root / "bad"
        good_dir = category_root / "good"

        # 重点：ground_truth 下面还有一个 bad 文件夹
        gt_dir = category_root / "ground_truth" / "bad"

        print("=" * 80)
        print(f"Processing category: {category}")
        print(f"bad: {bad_dir}")
        print(f"good: {good_dir}")
        print(f"ground_truth: {gt_dir}")

        bad_images = list_images(bad_dir)
        good_images = list_images(good_dir)

        bad_splits = split_files(bad_images)
        good_splits = split_files(good_images)

        category_bad_count = 0
        category_good_count = 0
        category_missing_masks = 0
        category_bbox_count = 0

        # -------------------------------------------------
        # 处理 bad 图片
        # -------------------------------------------------
        for split, files in bad_splits.items():
            for img_path in files:
                new_img_name = f"{category}_bad_{img_path.stem}{img_path.suffix.lower()}"
                new_label_name = Path(new_img_name).with_suffix(".txt").name

                dst_img_path = out_dir / "images" / split / new_img_name
                dst_label_path = out_dir / "labels" / split / new_label_name

                shutil.copy2(img_path, dst_img_path)

                mask_path = find_mask_for_bad_image(img_path, gt_dir)

                if mask_path is None:
                    bbox_lines = []
                    category_missing_masks += 1
                    total_missing_masks += 1

                    print(f"[WARNING] Missing mask for: {img_path}")
                    dst_label_path.write_text("", encoding="utf-8")
                else:
                    bbox_lines = mask_to_yolo_bboxes(mask_path, img_path)
                    dst_label_path.write_text("\n".join(bbox_lines), encoding="utf-8")

                category_bad_count += 1
                total_bad += 1

                category_bbox_count += len(bbox_lines)
                total_bbox += len(bbox_lines)

                mapping_rows.append({
                    "split": split,
                    "category": category,
                    "type": "bad",
                    "original_image_path": str(img_path),
                    "original_mask_path": str(mask_path) if mask_path else "",
                    "new_image_name": new_img_name,
                    "new_label_name": new_label_name,
                    "bbox_count": len(bbox_lines),
                })

        # -------------------------------------------------
        # 处理 good 图片
        # -------------------------------------------------
        for split, files in good_splits.items():
            for img_path in files:
                new_img_name = f"{category}_good_{img_path.stem}{img_path.suffix.lower()}"
                new_label_name = Path(new_img_name).with_suffix(".txt").name

                dst_img_path = out_dir / "images" / split / new_img_name
                dst_label_path = out_dir / "labels" / split / new_label_name

                shutil.copy2(img_path, dst_img_path)

                # good 图片无缺陷，label 文件为空
                dst_label_path.write_text("", encoding="utf-8")

                category_good_count += 1
                total_good += 1

                mapping_rows.append({
                    "split": split,
                    "category": category,
                    "type": "good",
                    "original_image_path": str(img_path),
                    "original_mask_path": "",
                    "new_image_name": new_img_name,
                    "new_label_name": new_label_name,
                    "bbox_count": 0,
                })

        summary_rows.append({
            "category": category,
            "bad_images": category_bad_count,
            "good_images": category_good_count,
            "missing_masks": category_missing_masks,
            "bbox_count": category_bbox_count,
            "train_images": len(bad_splits["train"]) + len(good_splits["train"]),
            "val_images": len(bad_splits["val"]) + len(good_splits["val"]),
            "test_images": len(bad_splits["test"]) + len(good_splits["test"]),
        })

        print(f"Done category: {category}")
        print(f"Bad images: {category_bad_count}")
        print(f"Good images: {category_good_count}")
        print(f"Missing masks: {category_missing_masks}")
        print(f"Generated bboxes: {category_bbox_count}")

    # -------------------------------------------------
    # 保存 mapping.csv
    # -------------------------------------------------
    mapping_csv = out_dir / "mapping.csv"
    with open(mapping_csv, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "split",
            "category",
            "type",
            "original_image_path",
            "original_mask_path",
            "new_image_name",
            "new_label_name",
            "bbox_count",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(mapping_rows)

    # -------------------------------------------------
    # 保存 summary.csv
    # -------------------------------------------------
    summary_csv = out_dir / "summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "category",
            "bad_images",
            "good_images",
            "missing_masks",
            "bbox_count",
            "train_images",
            "val_images",
            "test_images",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    yaml_path = write_data_yaml(out_dir)

    print("=" * 80)
    print("ALL DONE.")
    print(f"Output dataset: {out_dir}")
    print(f"data.yaml: {yaml_path}")
    print(f"mapping.csv: {mapping_csv}")
    print(f"summary.csv: {summary_csv}")
    print("-" * 80)
    print(f"Total bad images: {total_bad}")
    print(f"Total good images: {total_good}")
    print(f"Total missing masks: {total_missing_masks}")
    print(f"Total generated bboxes: {total_bbox}")
    print("=" * 80)


if __name__ == "__main__":
    main()