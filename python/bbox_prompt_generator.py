import os
from PIL import Image

# ================== 路径配置 ==================
label_dir = r"C:\Users\20552\Desktop\git\ultralytics\runs\detect\predict-2\labels"
image_dir = r"C:\Users\20552\Desktop\dataset\images\test"

output_file = "sam2_prompts.txt"
# =============================================


def yolo_to_xyxy(xc, yc, w, h, img_w, img_h):
    """
    YOLO (0-1 normalized) → original image coords
    """
    x1 = (xc - w / 2) * img_w
    y1 = (yc - h / 2) * img_h
    x2 = (xc + w / 2) * img_w
    y2 = (yc + h / 2) * img_h
    return int(x1), int(y1), int(x2), int(y2)


def get_image_size(img_name):
    """
    读取原图尺寸（关键步骤）
    """
    img_path = os.path.join(image_dir, img_name)

    # 如果扩展名不是jpg，自动尝试png
    if not os.path.exists(img_path):
        base = os.path.splitext(img_name)[0]
        img_path = os.path.join(image_dir, base + ".png")

    img = Image.open(img_path)
    return img.size  # (W, H)


def main():

    with open(output_file, "w") as out:

        for file in os.listdir(label_dir):
            if not file.endswith(".txt"):
                continue

            label_path = os.path.join(label_dir, file)
            img_name = file.replace(".txt", ".jpg")

            # 读取原图尺寸
            try:
                img_w, img_h = get_image_size(img_name)
            except:
                print(f"Image not found: {img_name}")
                continue

            with open(label_path, "r") as f:
                for line in f.readlines():

                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    cls, xc, yc, w, h = parts[:5]
                    xc, yc, w, h = map(float, [xc, yc, w, h])

                    # YOLO → SAM2 bbox（原图尺度）
                    x1, y1, x2, y2 = yolo_to_xyxy(
                        xc, yc, w, h, img_w, img_h
                    )

                    out.write(f"{img_name} {x1} {y1} {x2} {y2}\n")

    print("✔ SAM2 prompts saved to sam2_prompts.txt")


if __name__ == "__main__":
    main()