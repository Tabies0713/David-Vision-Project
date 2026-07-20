from pathlib import Path
import time

import torch
from ultralytics import YOLO


MODEL_PATH = r"C:\Users\20552\Desktop\baseline.pt"
TEST_DIR = Path(r"C:\Users\20552\Desktop\dataset\images\test")

IMGSZ = 640
CONF = 0.25
DEVICE = "cpu"  # GPU；使用CPU时改成 "cpu"


image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
image_paths = sorted(
    p for p in TEST_DIR.iterdir()
    if p.suffix.lower() in image_extensions
)

if not image_paths:
    raise FileNotFoundError(f"No test images found in: {TEST_DIR}")

model = YOLO(MODEL_PATH)

# GPU warm-up，不计入正式时间
for _ in range(3):
    model.predict(
        source=str(image_paths[0]),
        imgsz=IMGSZ,
        conf=CONF,
        device=DEVICE,
        verbose=False,
        save=False
    )

if torch.cuda.is_available() and DEVICE != "cpu":
    torch.cuda.synchronize()

# 正式测试整个 test set
start_time = time.perf_counter()

results = model.predict(
    source=[str(p) for p in image_paths],
    imgsz=IMGSZ,
    conf=CONF,
    device=DEVICE,
    verbose=False,
    save=False,
    stream=False
)

if torch.cuda.is_available() and DEVICE != "cpu":
    torch.cuda.synchronize()

total_time = time.perf_counter() - start_time
average_time = total_time / len(image_paths)

images_with_boxes = sum(len(result.boxes) > 0 for result in results)

print("=" * 50)
print("YOLO26m Baseline Inference Time")
print("=" * 50)
print(f"Total test images: {len(image_paths)}")
print(f"Images with detected boxes: {images_with_boxes}")
print(f"Total inference time: {total_time:.4f} s")
print(f"Average inference time: {average_time:.4f} s/image")
print(f"Average FPS: {1 / average_time:.2f}")