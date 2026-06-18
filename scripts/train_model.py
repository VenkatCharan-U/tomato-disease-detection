import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from sklearn.utils import class_weight

# -----------------------
# 1️⃣ Load Class Labels
# -----------------------
if not os.path.exists("tomato_classes.json"):
    raise FileNotFoundError("tomato_classes.json not found! Generate it first.")

with open("tomato_classes.json", "r") as f:
    tomato_classes = json.load(f)

print("✅ Loaded Classes:", tomato_classes)

# -----------------------
# 2️⃣ Dataset paths
# -----------------------
train_dir = "C:/Users/mahas/OneDrive/Desktop/ml-member/dataset/train"
val_dir = "C:/Users/mahas/OneDrive/Desktop/ml-member/dataset/val"

# -----------------------
# 3️⃣ Filter out empty folders & warn
# -----------------------
def get_valid_subfolders(folder, class_labels):
    valid_folders = []
    for subfolder in os.listdir(folder):
        path = os.path.join(folder, subfolder)
        if os.path.isdir(path) and subfolder in class_labels:
            n_imgs = len([f for f in os.listdir(path) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
            if n_imgs > 0:
                valid_folders.append(subfolder)
                print(f"✅ {subfolder}: {n_imgs} images")
            else:
                print(f"⚠️ No images in {subfolder}, skipping.")
        else:
            print(f"⚠️ Skipping invalid folder: {subfolder}")
    return valid_folders

valid_train = get_valid_subfolders(train_dir, tomato_classes)
valid_val = get_valid_subfolders(val_dir, tomato_classes)

if not valid_train or not valid_val:
    raise ValueError("❌ No valid images found in training or validation directories.")

# -----------------------
# 4️⃣ Data Generators with Augmentation
# -----------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

# -----------------------
# 5️⃣ Compute class weights
# -----------------------
train_labels = train_gen.classes
class_weights_arr = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights_dict = dict(enumerate(class_weights_arr))
print("✅ Class weights:", class_weights_dict)

# -----------------------
# 6️⃣ Build Model
# -----------------------
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(len(tomato_classes), activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# -----------------------
# 7️⃣ Train Model with class weights
# -----------------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    class_weight=class_weights_dict
)

# -----------------------
# 8️⃣ Save Model
# -----------------------
os.makedirs("C:/Users/mahas/OneDrive/Desktop/ml-member/models", exist_ok=True)
model.save("C:/Users/mahas/OneDrive/Desktop/ml-member/models/tomato_model.h5")
model.save("tomato_disease_model.h5")  # For backend use
print("✅ Model training complete and saved!")
