from pathlib import Path
import time
import csv
import cv2
import torch
from ultralytics import YOLO, SAM

# ================== Path configuration ==================

IMAGE_DIR = Path(r"C:\Users\20552\Desktop\dataset\images\test")

YOLO_MODEL_PATH = Path(r"C:\Users\20552\Desktop\baseline.pt")
SAM2_MODEL_PATH = Path(r"C:\Users\20552\Desktop\sam2.1_t.pt")

OUTPUT_CSV = Path(r"C:\Users\20552\Desktop\inference_time_results_rerun.csv")
OUTPUT_SUMMARY = Path(r"C:\Users\20552\Desktop\inference_time_summary_rerun.txt")

# Current environment is CPU-only.
# If CUDA is available later, change this to 0.
DEVICE = "cpu"

IMGSZ = 640
CONF = 0.25
YOLO_WARMUP_RUNS = 3

# ========================================================


def sync_device():
    """Wait for CUDA operations to finish before reading the timer."""
    if DEVICE != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()


def get_image_paths(image_dir: Path):
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]:
        image_paths.extend(image_dir.glob(ext))
    return sorted(image_paths)


def validate_paths():
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")

    if not YOLO_MODEL_PATH.exists():
        raise FileNotFoundError(f"YOLO model not found: {YOLO_MODEL_PATH}")

    if not SAM2_MODEL_PATH.exists():
        raise FileNotFoundError(f"SAM2 model not found: {SAM2_MODEL_PATH}")


def main():
    validate_paths()

    image_paths = get_image_paths(IMAGE_DIR)

    if not image_paths:
        raise FileNotFoundError(f"No test images found in: {IMAGE_DIR}")

    print("=" * 70)
    print("YOLO26m + SAM2 Pipeline Timing")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Total test images: {len(image_paths)}")
    print(f"YOLO model: {YOLO_MODEL_PATH}")
    print(f"SAM2 model: {SAM2_MODEL_PATH}")
    print("=" * 70)

    # Load models before timing.
    yolo_model = YOLO(str(YOLO_MODEL_PATH))
    sam2_model = SAM(str(SAM2_MODEL_PATH))

    # -------------------- Warm-up --------------------
    print("Warming up YOLO...")

    for _ in range(YOLO_WARMUP_RUNS):
        _ = yolo_model.predict(
            source=str(image_paths[0]),
            imgsz=IMGSZ,
            conf=CONF,
            device=DEVICE,
            save=False,
            verbose=False,
        )

    sync_device()

    print("Warming up SAM2...")

    first_image = cv2.imread(str(image_paths[0]))
    if first_image is None:
        raise RuntimeError(f"Could not read image for warm-up: {image_paths[0]}")

    height, width = first_image.shape[:2]
    dummy_box = [[0, 0, width - 1, height - 1]]

    _ = sam2_model(
        source=str(image_paths[0]),
        bboxes=dummy_box,
        device=DEVICE,
        save=False,
        verbose=False,
    )

    sync_device()
    print("Warm-up finished.")
    print("=" * 70)

    # -------------------- Formal timing --------------------
    rows = []

    total_yolo_time = 0.0
    total_sam2_time = 0.0
    total_pipeline_time = 0.0

    images_with_boxes = 0
    total_boxes = 0

    for index, img_path in enumerate(image_paths, start=1):
        # ---------- YOLO wall-clock time ----------
        sync_device()
        start_yolo = time.perf_counter()

        yolo_results = yolo_model.predict(
            source=str(img_path),
            imgsz=IMGSZ,
            conf=CONF,
            device=DEVICE,
            save=False,
            verbose=False,
        )

        sync_device()
        yolo_time = time.perf_counter() - start_yolo

        boxes = []
        yolo_preprocess_ms = 0.0
        yolo_inference_ms = 0.0
        yolo_postprocess_ms = 0.0

        if yolo_results:
            result = yolo_results[0]

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy().tolist()

            speed = getattr(result, "speed", None) or {}
            yolo_preprocess_ms = float(speed.get("preprocess", 0.0))
            yolo_inference_ms = float(speed.get("inference", 0.0))
            yolo_postprocess_ms = float(speed.get("postprocess", 0.0))

        num_boxes = len(boxes)
        total_boxes += num_boxes

        # ---------- SAM2 wall-clock time ----------
        sam2_time = 0.0

        if num_boxes > 0:
            images_with_boxes += 1

            sync_device()
            start_sam2 = time.perf_counter()

            _ = sam2_model(
                source=str(img_path),
                bboxes=boxes,
                device=DEVICE,
                save=False,
                verbose=False,
            )

            sync_device()
            sam2_time = time.perf_counter() - start_sam2

        pipeline_time = yolo_time + sam2_time

        total_yolo_time += yolo_time
        total_sam2_time += sam2_time
        total_pipeline_time += pipeline_time

        rows.append(
            {
                "image": img_path.name,
                "num_boxes": num_boxes,
                "yolo_wall_time_sec": yolo_time,
                "sam2_wall_time_sec": sam2_time,
                "pipeline_wall_time_sec": pipeline_time,
                "yolo_preprocess_ms": yolo_preprocess_ms,
                "yolo_inference_ms": yolo_inference_ms,
                "yolo_postprocess_ms": yolo_postprocess_ms,
            }
        )

        print(
            f"[{index:03d}/{len(image_paths)}] {img_path.name} | "
            f"boxes={num_boxes} | "
            f"YOLO={yolo_time:.4f}s | "
            f"SAM2={sam2_time:.4f}s | "
            f"Pipeline={pipeline_time:.4f}s"
        )

    # -------------------- Save CSV --------------------
    fieldnames = [
        "image",
        "num_boxes",
        "yolo_wall_time_sec",
        "sam2_wall_time_sec",
        "pipeline_wall_time_sec",
        "yolo_preprocess_ms",
        "yolo_inference_ms",
        "yolo_postprocess_ms",
    ]

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # -------------------- Summary --------------------
    image_count = len(image_paths)

    avg_yolo = total_yolo_time / image_count
    avg_sam2_all = total_sam2_time / image_count
    avg_pipeline = total_pipeline_time / image_count

    avg_sam2_detected = (
        total_sam2_time / images_with_boxes if images_with_boxes > 0 else 0.0
    )

    yolo_fps = 1.0 / avg_yolo if avg_yolo > 0 else 0.0
    pipeline_fps = 1.0 / avg_pipeline if avg_pipeline > 0 else 0.0

    summary_lines = [
        "=" * 70,
        "Inference Time Evaluation Finished",
        "=" * 70,
        f"Device: {DEVICE}",
        f"Total test images: {image_count}",
        f"Images with YOLO boxes: {images_with_boxes}",
        f"Total detected boxes: {total_boxes}",
        "-" * 70,
        "YOLO26m baseline timing from this same run:",
        f"Total YOLO time: {total_yolo_time:.4f} s",
        f"Average YOLO time: {avg_yolo:.4f} s/image",
        f"YOLO FPS: {yolo_fps:.2f}",
        "-" * 70,
        "SAM2 timing:",
        f"Total SAM2 time: {total_sam2_time:.4f} s",
        f"Average SAM2 time over all images: {avg_sam2_all:.4f} s/image",
        f"Average SAM2 time over detected images: {avg_sam2_detected:.4f} s/image",
        "-" * 70,
        "Complete YOLO26m -> SAM2 pipeline timing:",
        f"Total pipeline time: {total_pipeline_time:.4f} s",
        f"Average pipeline time: {avg_pipeline:.4f} s/image",
        f"Pipeline FPS: {pipeline_fps:.2f}",
        "-" * 70,
        f"Detailed CSV saved to: {OUTPUT_CSV}",
        f"Summary saved to: {OUTPUT_SUMMARY}",
        "=" * 70,
    ]

    summary_text = "\n".join(summary_lines)

    OUTPUT_SUMMARY.write_text(summary_text, encoding="utf-8")
    print(summary_text)


if __name__ == "__main__":
    main()
