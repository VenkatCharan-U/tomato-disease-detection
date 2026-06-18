import os

# Path to your raw dataset
DATASET_PATH = r"C:/Users/mahas/OneDrive/Desktop/ml-member/dataset/raw"

# Load class names from your JSON
import json
with open("tomato_classes.json", "r") as f:
    tomato_classes = json.load(f)

print("✅ Checking dataset...")

for class_name in tomato_classes:
    folder_path = os.path.join(DATASET_PATH, class_name)
    if not os.path.exists(folder_path):
        print(f"⚠️ Missing folder: {class_name}")
        continue

    # Count image files
    img_count = len([
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if img_count == 0:
        print(f"⚠️ No images in {class_name}")
    elif img_count < 10:
        print(f"⚠️ Very few images in {class_name}: {img_count}")
    else:
        print(f"✅ {class_name}: {img_count} images")
