from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

model = YOLO("yolov8s-seg.pt")

def remove_bg_from_image(input_path, output_path):
    image = Image.open(input_path).convert("RGB")
    image_np = np.array(image)
    temp_input = "temp.jpg"
    cv2.imwrite(temp_input, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

    results = model(temp_input)
    masks = results[0].masks

    if masks is None or len(masks.data) == 0:
        print("❌ لم يتم اكتشاف أي منتج.")
        return

    mask = masks.data[0].cpu().numpy().astype(np.uint8)
    mask = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]))

    white_bg = np.ones_like(image_np) * 255
    result = np.where(mask[:, :, None], image_np, white_bg)

    result_image = Image.fromarray(result).convert("RGB")
    result_image = result_image.resize((612, 612), Image.LANCZOS)
    result_image.save(output_path)
    print(f"✅ تم حفظ الصورة في {output_path}")

# مثال
remove_bg_from_image("input.png", "output.png")
