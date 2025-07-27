from flask import Flask, request, send_file, jsonify
from PIL import Image
import io
import os
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# تحميل موديل YOLOv8 segmentation
model = YOLO("yolov8s-seg.pt")  # تأكد من تحميل الموديل مسبقًا

def process_image(image_stream):
    image = Image.open(image_stream).convert("RGB")
    image_np = np.array(image)
    
    # حفظ مؤقت للصورة لتحليلها بالموديل
    temp_input_path = "temp_input.jpg"
    cv2.imwrite(temp_input_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    
    results = model(temp_input_path)
    masks = results[0].masks

    if masks is None or len(masks.data) == 0:
        raise ValueError("لم يتم اكتشاف أي منتج في الصورة.")

    # نأخذ أول قناع فقط
    mask = masks.data[0].cpu().numpy().astype(np.uint8)
    mask = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]))

    # عزل المنتج بخلفية بيضاء
    white_bg = np.ones_like(image_np) * 255
    result_np = np.where(mask[:, :, None], image_np, white_bg)

    # تحويل النتيجة إلى صورة PIL وتعديل الحجم
    result_img = Image.fromarray(result_np).convert("RGB")
    result_img = result_img.resize((612, 612), Image.LANCZOS)

    return result_img

@app.route('/remove-bg', methods=['POST'])
def remove_bg():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        result_img = process_image(image_file.stream)

        buffer = io.BytesIO()
        result_img.save(buffer, format="PNG")
        buffer.seek(0)

        return send_file(buffer, mimetype='image/png')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)