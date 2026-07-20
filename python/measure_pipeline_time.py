from pathlib import Path
import time
import csv
import numpy as np
from ultralytics import YOLO, SAM

# ================== 路径配置 ==================

IMAGE_DIR = Path(r"C:\Users\20552\Desktop\dataset\images\test")

YOLO_MODEL_PATH = r"C:\Users\20552\Desktop\baseline.pt"

SAM2_MODEL_PATH = r"C:\Users\20552\Desktop\sam2.1_t.pt"

OUTPUT_CSV = Path(r"C:\Users\20552\Desktop\inference_time_results.csv")

# =============================================


def main():
    image_paths = []

    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]:
        image_paths.extend(list(IMAGE_DIR.glob(ext)))

    image_paths = sorted(image_paths)

    print(f"Total test images: {len(image_paths)}")

    yolo_model = YOLO(YOLO_MODEL_PATH)
    sam2_model = SAM(SAM2_MODEL_PATH)

    results = []

    total_yolo_time = 0.0
    total_sam2_time = 0.0
    total_pipeline_time = 0.0

    images_with_boxes = 0

    for img_path in image_paths:
        image_name = img_path.name

        # ---------- YOLO inference ----------
        start_yolo = time.perf_counter()

        yolo_results = yolo_model.predict(
            source=str(img_path),
            imgsz=640,
            conf=0.25,
            verbose=False
        )

        end_yolo = time.perf_counter()
        yolo_time = end_yolo - start_yolo

        boxes = []

        if len(yolo_results) > 0 and yolo_results[0].boxes is not None:
            boxes = yolo_results[0].boxes.xyxy.cpu().numpy().tolist()

        # ---------- SAM2 inference ----------
        sam2_time = 0.0

        if len(boxes) > 0:
            images_with_boxes += 1

            start_sam2 = time.perf_counter()

            _ = sam2_model(
                source=str(img_path),
                bboxes=boxes,
                save=False,
                verbose=False
            )

            end_sam2 = time.perf_counter()
            sam2_time = end_sam2 - start_sam2

        pipeline_time = yolo_time + sam2_time

        total_yolo_time += yolo_time
        total_sam2_time += sam2_time
        total_pipeline_time += pipeline_time

        results.append({
            "image": image_name,
            "num_boxes": len(boxes),
            "yolo_time_sec": yolo_time,
            "sam2_time_sec": sam2_time,
            "pipeline_time_sec": pipeline_time,
        })

        print(
            f"{image_name} | boxes: {len(boxes)} | "
            f"YOLO: {yolo_time:.4f}s | SAM2: {sam2_time:.4f}s | "
            f"Total: {pipeline_time:.4f}s"
        )

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "num_boxes",
                "yolo_time_sec",
                "sam2_time_sec",
                "pipeline_time_sec",
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    avg_yolo = total_yolo_time / len(image_paths)
    avg_sam2_all = total_sam2_time / len(image_paths)
    avg_pipeline = total_pipeline_time / len(image_paths)

    if images_with_boxes > 0:
        avg_sam2_detected = total_sam2_time / images_with_boxes
    else:
        avg_sam2_detected = 0.0

    print("=" * 60)
    print("Inference Time Evaluation Finished")
    print(f"Total test images: {len(image_paths)}")
    print(f"Images with YOLO boxes: {images_with_boxes}")
    print("-" * 60)
    print(f"Average YOLO time / image: {avg_yolo:.4f} s")
    print(f"Average SAM2 time / all images: {avg_sam2_all:.4f} s")
    print(f"Average SAM2 time / detected images: {avg_sam2_detected:.4f} s")
    print(f"Average total pipeline time / image: {avg_pipeline:.4f} s")
    print("-" * 60)
    print(f"Detailed time results saved to: {OUTPUT_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()