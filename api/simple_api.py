from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.saving import load_model
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import os
import tensorflow as tf


app = FastAPI()
# Model path needs to be set
app.state.model = load_model('api/train01.keras')



# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/predict")
def predict():
    print('### Predict')
    # Files
    for file in os.listdir('api'):
        if file.endswith('.png'):
            img = cv2.imread(os.path.join('api', file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print(f"Input shape: {img.shape}")

            # Resize & Normalize
            img_proc = tf.image.resize(img, size=(224, 224), method=tf.image.ResizeMethod.BICUBIC) / 255.0  # (224, 224, 3)

            # Add Batch Dimension (1, 224, 224, 3)
            img_proc = tf.expand_dims(img_proc, axis=0)
            print(f"Output shape: {img_proc.shape}")  # Expected: (1, 224, 224, 3)

            # Model Prediction
            prediction = app.state.model.predict(img_proc)[0]
            class_index = np.argmax(prediction)
            print(f"Prediction: {prediction}")
            print(f"Class index: {class_index}")
            os.remove(os.path.join('api', file))
            return {
                "prediction": float(class_index)
            }


@app.get("/")
def root():
    return {
        'greeting': 'Hello'
    }

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    # Get and safe file
    image_bytes = await file.read()
    image = np.array(Image.open(BytesIO(image_bytes)))
    cv2.imwrite(os.path.join('api', file.filename), image)
    return {
        "shape": image.shape
    }
