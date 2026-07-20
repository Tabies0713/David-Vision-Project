from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import SAM


# =========================
# 你只需要改这里
# =========================

IMAGE_DIR = Path(r"C:\Users\20552\Desktop\dataset\images\test")

PROMPT_FILE = Path(r"C:\Users\20552\Desktop\sam2_prompts.txt")

SAM2_MODEL_PATH = r"C:\Users\20552\Desktop\sam2.1_t.pt"

OUTPUT_DIR = Path(r"C:\Users\20552\Desktop\sam2_results")

# =========================


def read_prompts(prompt_file: Path):
    """
    读取 sam2_prompts.txt
    格式：
    image_name x1 y1 x2 y2
    """
    prompts = defaultdict(list)

    with open(prompt_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) != 5:
                continue

            image_name = parts[0]
            x1, y1, x2, y2 = map(int, parts[1:])

            prompts[image_name].append([x1, y1, x2, y2])

    return prompts


def find_image(image_dir: Path, image_name: str):
    """
    根据图片名寻找图片。
    如果 jpg 找不到，会自动尝试 png/jpeg。
    """
    img_path = image_dir / image_name

    if img_path.exists():
        return img_path

    stem = Path(image_name).stem

    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    return None


def save_combined_mask(mask_list, save_path: Path):
    """
    多个 bbox 可能生成多个 mask，这里合成一张二值 mask。
    """
    if len(mask_list) == 0:
        return False

    combined = np.zeros_like(mask_list[0], dtype=np.uint8)

    for mask in mask_list:
        mask = mask.astype(np.uint8)
        combined = np.maximum(combined, mask)

    combined = combined * 255
    cv2.imwrite(str(save_path), combined)

    return True


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mask_dir = OUTPUT_DIR / "masks"
    vis_dir = OUTPUT_DIR / "visualizations"

    mask_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    prompts = read_prompts(PROMPT_FILE)

    print(f"Loaded {len(prompts)} images with YOLO box prompts.")

    model = SAM(SAM2_MODEL_PATH)

    success_count = 0
    failed_count = 0

    for image_name, boxes in prompts.items():
        img_path = find_image(IMAGE_DIR, image_name)

        if img_path is None:
            print(f"[Missing image] {image_name}")
            failed_count += 1
            continue

        print(f"Processing: {image_name}, boxes: {len(boxes)}")

        try:
            results = model(
                source=str(img_path),
                bboxes=boxes,
                save=False,
                verbose=False
            )

            if len(results) == 0 or results[0].masks is None:
                print(f"[No mask] {image_name}")
                failed_count += 1
                continue

            masks_tensor = results[0].masks.data.cpu().numpy()

            mask_list = []
            for m in masks_tensor:
                binary_mask = (m > 0.5).astype(np.uint8)
                mask_list.append(binary_mask)

            save_name = Path(image_name).stem + "_sam2_mask.png"
            save_path = mask_dir / save_name

            ok = save_combined_mask(mask_list, save_path)

            if ok:
                success_count += 1
            else:
                failed_count += 1

        except Exception as e:
            print(f"[Error] {image_name}: {e}")
            failed_count += 1

    print("=" * 60)
    print("SAM2 inference finished.")
    print(f"Success images: {success_count}")
    print(f"Failed images: {failed_count}")
    print(f"Mask results saved to: {mask_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()