from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import os
from ml_logic.registry import load_model
from keras.utils import image_dataset_from_directory


app = FastAPI()

print("Local model path: ", os.environ.get('LOCAL_MODEL_PATH'))
print("Bucket name: ", os.environ.get('BUCKET_NAME'))
print("GCP Project ID: ", os.environ.get('GCP_PROJECT_ID'))
print("Model name: ", os.environ.get('PRODUCTION_MODEL_NAME'))

# Load params
model_name = os.environ.get('PRODUCTION_MODEL_NAME')
image_dir = '/tmp/images'
app.state.model = load_model(model_name)

# Check if image folder exists
if not os.path.exists(image_dir):
    os.makedirs(image_dir)


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
    print(f"Files in image dir befor pred: {os.listdir(image_dir)}")
    if len(os.listdir(image_dir)) == 0:
        return {"error": "No images found in image directory"}
    data = image_dataset_from_directory(
        directory=image_dir,
        labels='inferred',
        label_mode=None,
        batch_size=32,
        image_size=(224, 224),
        shuffle=False  # Ensure order consistency
    )
    if data is None:
        return {"error": "No images found in image directory"}

    file_names = os.listdir(image_dir)

    # Model Prediction
    prediction = app.state.model.predict(data)
    class_index = np.argmax(prediction)
    print(f"Prediction: {prediction}")
    print(f"Class index: {class_index}")

    predictions = []   # list to store predictions
    i = 0
    for pred in prediction:
        print(f"Prediction: {int(np.argmax(pred))}")
        predictions.append({"filename": file_names[i], "prediction": int(np.argmax(pred))})
        os.remove(os.path.join(image_dir, file_names[i]))
        i += 1

    print(predictions)
    print(f"Files in image dir after pred: {os.listdir(image_dir)}")
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
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(image_dir, file.filename), bgr_image)
    return {
        "shape": image.shape
    }
