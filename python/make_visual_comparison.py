from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np

# ================== 路径配置 ==================

YOLO_DATASET_ROOT = Path(r"C:\Users\20552\Desktop\dataset")
RAW_DATASET_ROOT = Path(r"C:\Users\20552\Desktop\dataset2")

IMAGE_DIR = YOLO_DATASET_ROOT / "images" / "test"
PRED_MASK_DIR = Path(r"C:\Users\20552\Desktop\sam2_results\masks")
PROMPT_FILE = Path(r"C:\Users\20552\Desktop\sam2_prompts.txt")

OUTPUT_DIR = Path(r"C:\Users\20552\Desktop\sam2_results\visualizations")

CATEGORIES = [
    "sheet_metal",
    "fruit_jelly",
    "wallplugs",
    "can",
    "fabric",
    "vial",
]

# =============================================


def read_prompts(prompt_file):
    prompts = defaultdict(list)

    if not prompt_file.exists():
        return prompts

    with open(prompt_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) != 5:
                continue

            image_name = parts[0]
            x1, y1, x2, y2 = map(int, parts[1:])
            prompts[image_name].append([x1, y1, x2, y2])

    return prompts


def parse_image_name(image_name):
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
    cat, status, original_stem = parse_image_name(image_name)

    if status == "good":
        return None

    if cat is None:
        return None

    gt_root = RAW_DATASET_ROOT / cat / cat / "test_public" / "ground_truth"

    if not gt_root.exists():
        return None

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


def read_binary_mask(path, target_shape):
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


def overlay_mask(image, mask, color, alpha=0.45):
    """
    在原图上叠加 mask。
    color 使用 BGR 格式。
    """
    overlay = image.copy()
    mask_bool = mask.astype(bool)

    if mask_bool.sum() == 0:
        return overlay

    color_array = np.array(color, dtype=np.float32)

    overlay[mask_bool] = (
        image[mask_bool].astype(np.float32) * (1 - alpha)
        + color_array * alpha
    ).astype(np.uint8)

    return overlay


def draw_boxes(image, boxes):
    img = image.copy()

    for box in boxes:
        x1, y1, x2, y2 = box

        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            3
        )

    return img


def put_title(image, title):
    img = image.copy()

    cv2.rectangle(
        img,
        (0, 0),
        (img.shape[1], 45),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        img,
        title,
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return img


def resize_panel(image, size=(640, 480)):
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def make_error_map(image, pred_mask, gt_mask):
    """
    绿色：预测正确区域 TP
    红色：多预测区域 FP
    蓝色：漏分区域 FN
    """
    base = image.copy()

    tp = np.logical_and(pred_mask == 1, gt_mask == 1)
    fp = np.logical_and(pred_mask == 1, gt_mask == 0)
    fn = np.logical_and(pred_mask == 0, gt_mask == 1)

    color_map = np.zeros_like(image)

    color_map[tp] = (0, 255, 0)      # green: TP
    color_map[fp] = (0, 0, 255)      # red: FP
    color_map[fn] = (255, 0, 0)      # blue: FN

    mask_any = np.logical_or.reduce([tp, fp, fn])

    if mask_any.sum() == 0:
        return base

    base[mask_any] = (
        base[mask_any].astype(np.float32) * 0.45
        + color_map[mask_any].astype(np.float32) * 0.55
    ).astype(np.uint8)

    return base


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prompts = read_prompts(PROMPT_FILE)

    image_paths = []

    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]:
        image_paths.extend(list(IMAGE_DIR.glob(ext)))

    image_paths = sorted(image_paths)

    count = 0

    for img_path in image_paths:
        image_name = img_path.name
        stem = img_path.stem

        image = cv2.imread(str(img_path))

        if image is None:
            print(f"[Skip] Cannot read image: {image_name}")
            continue

        h, w = image.shape[:2]
        target_shape = (h, w)

        pred_mask_path = PRED_MASK_DIR / f"{stem}_sam2_mask.png"
        gt_mask_path = find_gt_mask(image_name)

        pred_mask = read_binary_mask(pred_mask_path, target_shape)
        gt_mask = read_binary_mask(gt_mask_path, target_shape)

        boxes = prompts.get(image_name, [])

        # 1. 原图
        panel_original = image.copy()

        # 2. YOLO框 + SAM2预测mask
        panel_pred = image.copy()
        panel_pred = overlay_mask(
            panel_pred,
            pred_mask,
            color=(0, 0, 255),
            alpha=0.45
        )
        panel_pred = draw_boxes(panel_pred, boxes)

        # 3. Ground truth mask
        panel_gt = image.copy()
        panel_gt = overlay_mask(
            panel_gt,
            gt_mask,
            color=(0, 255, 0),
            alpha=0.45
        )

        # 4. TP / FP / FN 对比图
        panel_error = make_error_map(image, pred_mask, gt_mask)

        panel_original = put_title(resize_panel(panel_original), "Original Image")
        panel_pred = put_title(resize_panel(panel_pred), "YOLO Box + SAM2 Mask")
        panel_gt = put_title(resize_panel(panel_gt), "Ground Truth Mask")
        panel_error = put_title(resize_panel(panel_error), "TP / FP / FN Comparison")

        top = np.hstack([panel_original, panel_pred])
        bottom = np.hstack([panel_gt, panel_error])

        comparison = np.vstack([top, bottom])

        save_path = OUTPUT_DIR / f"{stem}_comparison.jpg"
        cv2.imwrite(str(save_path), comparison)

        count += 1
        print(f"Saved: {save_path}")

    print("=" * 60)
    print("Visualization finished.")
    print(f"Total visualization images: {count}")
    print(f"Saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()