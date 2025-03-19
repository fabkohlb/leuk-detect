from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import os
import torch
import tensorflow as tf
from torch.utils.data import Dataset
from ml_logic.registry import load_model
from ml_logic import params
from keras.utils import image_dataset_from_directory
from transformers import Trainer, TrainingArguments, AutoImageProcessor
from transformers import AutoModelForImageClassification
from torchvision import transforms


app = FastAPI()

print("Local model path: ", os.environ.get('LOCAL_MODEL_PATH'))
print("Bucket name: ", os.environ.get('BUCKET_NAME'))
print("GCP Project ID: ", os.environ.get('GCP_PROJECT_ID'))
print("Model name: ", os.environ.get('PRODUCTION_MODEL_NAME'))


# Load params
model_name = os.environ.get('PRODUCTION_MODEL_NAME')
image_dir = 'api/images' #'/tmp/images'
num_labels = 15
app.state.model = AutoModelForImageClassification.from_pretrained(os.path.join(os.environ.get('LOCAL_MODEL_PATH'), os.environ.get('PRODUCTION_MODEL_NAME')), num_labels=num_labels)
app.state.model.eval()
app.state.image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
app.state.transform_augmentation = transforms.Compose([
    transforms.RandomCrop(224),  # Random crop
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.RandomRotation(10),  # Random rotation (adjust as needed)
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Random resized crop
    transforms.RandomVerticalFlip(),  # Random vertical flip
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

# Helper

class CustomDataset(Dataset):
    def __init__(self, image_batch, image_processor, transform=None):
        self.image_batch = image_batch
        self.image_processor = image_processor
        self.transform = transform
        print("Custom Dataset created.")

    def __len__(self):
        return len(self.image_batch)

    def __getitem__(self, idx):
        image = self.image_batch[idx]

        # Step 1: Preprocess image using the Hugging Face processor (resize, normalization)
        inputs = self.image_processor(image, return_tensors="pt", do_rescale=False)
        pixel_values = inputs["pixel_values"].squeeze(0)  # Remove the batch dimension (since we have a single image)

        # Step 2: Convert the tensor to a PIL Image for augmentation
        pil_image = transforms.ToPILImage()(pixel_values)

        # Step 3: Apply the transformation (augmentation)
        if self.transform:
            pil_image = self.transform(pil_image)

        return {"pixel_values": pil_image}


def load_and_process_dataset():
    print('📂 Loading images')
    data_raw = tf.keras.preprocessing.image_dataset_from_directory(
        directory=os.path.join(image_dir),
        #labels='inferred',
        label_mode=None,
        batch_size=params.BATCH_SIZE,
        image_size=(224, 224),
    )
    images = []
    file_names = [os.path.basename(path) for path in data_raw.file_paths]
    print(f"File names: {file_names}")

    for image_batch in data_raw:
        images.append(image_batch.numpy() / 255.)  # Convert from TensorFlow tensor to NumPy array


    # Stack them into single arrays
    images = np.concatenate(images, axis=0)

    # Convert images to PyTorch tensors
    images = torch.tensor(images).permute(0, 3, 1, 2)  # Convert shape to [batch_size, channels, height, width]
    print("✅ Dataset loaded. Turn into CustomDataset.")
    return CustomDataset(images, app.state.image_processor, transform=app.state.transform_augmentation), file_names


def predict_on_dataset(dataset):
    """Evaluate the model on a given dataset and print metrics."""
    app.state.model.eval()  # Set to evaluation mode
    predictions = []

    for batch in dataset:
        pixel_values = batch["pixel_values"].unsqueeze(0)  # Ensure batch shape

        with torch.no_grad():
            outputs = app.state.model(pixel_values)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=-1).item()
        predictions.append(predicted_class)
    return predictions




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

    # Model Prediction
    dataset, file_names = load_and_process_dataset()
    predicted_classes = predict_on_dataset(dataset)
    predictions = []
    for i, pred in enumerate(predicted_classes):
        predictions.append({"filename": file_names[i], "prediction": pred})
        os.remove(os.path.join(image_dir, file_names[i]))

    print(f"Predictions: {predictions}")
    # print(f"Files in image dir after pred: {os.listdir(image_dir)}")
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
    print(f"File {file.filename} saved to {os.path.join(image_dir, file.filename)}")
    return {
        "shape": image.shape
    }
