import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# -----------------------
# 1️⃣ Load Model
# -----------------------
model_path = "D:/ml-member/models/tomato_model.h5"
model = load_model(model_path)
print("✅ Model loaded successfully!")

# -----------------------
# 2️⃣ Load Class Labels
# -----------------------
with open("tomato_classes.json", "r") as f:
    tomato_classes = json.load(f)

idx_to_class = {v: k for k, v in tomato_classes.items()}

# -----------------------
# 3️⃣ Prediction Function
# -----------------------
def predict_image(img_path):

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    pred = model.predict(img_array)
    class_idx = np.argmax(pred, axis=1)[0]
    confidence = pred[0][class_idx]

    pred_class_name = idx_to_class[class_idx]

    return {
        "disease": pred_class_name,
        "confidence": float(confidence)
    }

# -----------------------
# 4️⃣ Run Only When Directly Executed
# -----------------------
if __name__ == "__main__":

    test_dir = "D:/ml-member/test_images"

    for img_file in os.listdir(test_dir):
        img_path = os.path.join(test_dir, img_file)

        try:
            result = predict_image(img_path)
            print(f"Image: {img_file} → Disease: {result['disease']}, Confidence: {result['confidence']:.2f}")

        except Exception as e:
            print(f"❌ Failed to predict {img_file}: {e}")
