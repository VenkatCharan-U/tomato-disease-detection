import os
import shutil

# 👇 List of all dataset roots you want to use
SOURCE_DATASETS = [
    r"C:/Users/mahas/OneDrive/Desktop/ml-member/dataset/raw/color",
    r"C:/Users/mahas/OneDrive/Desktop/ml-member/dataset/raw/grayscale",
]

TARGET_TRAIN = r"C:/Users/mahas/OneDrive/Desktop/ml-member/dataset/train"
TARGET_VAL = r"C:/Users/mahas/OneDrive/Desktop/ml-member/dataset/val"

# Create train/val directories if not exist
os.makedirs(TARGET_TRAIN, exist_ok=True)
os.makedirs(TARGET_VAL, exist_ok=True)

# Define which classes we want
TOMATO_CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites",
    "Tomato___Target_Spot",
    "Tomato___Yellow_Leaf_Curl_Virus",
    "Tomato___Mosaic_virus",
    "Tomato___healthy",
]

from sklearn.model_selection import train_test_split

for dataset_root in SOURCE_DATASETS:
    for cls in TOMATO_CLASSES:
        class_folder = os.path.join(dataset_root, cls)
        if not os.path.exists(class_folder):
            print(f"⚠️ Skipping missing folder: {class_folder}")
            continue

        images = [os.path.join(class_folder, img) for img in os.listdir(class_folder) if img.endswith(".jpg")]
        if not images:
            print(f"⚠️ No images found in: {class_folder}")
            continue

        train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)

        # Copy to train
        os.makedirs(os.path.join(TARGET_TRAIN, cls), exist_ok=True)
        for img in train_imgs:
            shutil.copy(img, os.path.join(TARGET_TRAIN, cls))

        # Copy to val
        os.makedirs(os.path.join(TARGET_VAL, cls), exist_ok=True)
        for img in val_imgs:
            shutil.copy(img, os.path.join(TARGET_VAL, cls))

        print(f"✅ {cls}: {len(train_imgs)} train, {len(val_imgs)} val images copied.")
