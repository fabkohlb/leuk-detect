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
#app.state.model = load_model('api/train01.keras')   # backup if try/except does not work

# Current model + BUCKET
model_name = "20250313-120438.keras"   # hardcoded - needs to be adapted if model changes 
bucket_name = os.environ.get("BUCKET")

# If model is not in /api download model_name, else use model_name in /api
try:
    app.state.model = load_model(f'api/{model_name}')

except:
    os.system(f"gsutil cp {bucket_name}/models/{model_name} .")
    app.state.model = load_model(f'api/{model_name}')


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
    
    predictions = []   # list to store predictions
    
    # Get all PNG images in the folder
    image_files = [file for file in os.listdir('api/images') if file.endswith('.png')]
    
    if not image_files:
        return {"message": "No images found to process"}
    
    # Files
    for file in image_files:
        file_path = os.path.join('api/images', file)
        
        # Read and process image
        img = cv2.imread(file_path)
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
        
        predictions.append({"filename": file, "prediction": int(class_index)})
        
        # Remove processed image
        os.remove(file_path)
    
    print({f"predictions": predictions})
    return {"predictions": predictions}


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
    cv2.imwrite(os.path.join('api/images', file.filename), image)
    return {
        "shape": image.shape
    }

