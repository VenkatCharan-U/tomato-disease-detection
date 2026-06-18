from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File
import shutil
import sys
import os

# allow importing test_model
sys.path.append("..")

import test_model   # IMPORTANT: import whole file

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("✅ Backend started. Model already loaded.")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    temp_path = "temp.jpg"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # use already loaded model
    result = test_model.predict_image(temp_path)

    return result
