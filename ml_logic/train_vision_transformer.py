from transformers import Trainer, TrainingArguments, AutoImageProcessor
from transformers import AutoModelForImageClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import Dataset
from torchvision import transforms
import tensorflow as tf
import os
import torch
import numpy as np
import params
from PIL import Image
import pandas as pd
from google.cloud import storage
from matplotlib import pyplot as plt
import seaborn as sns


print("🤗 Load AutoImageProcessor")
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
print("✅ Image processor loaded.")

print("🔧 Define image transformations")
transform_augmentation = transforms.Compose([
    transforms.RandomCrop(224),  # Random crop
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.RandomRotation(10),  # Random rotation (adjust as needed)
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Random resized crop
    transforms.RandomVerticalFlip(),  # Random vertical flip
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])
print("✅ Image transformations defined.")


class CustomDataset(Dataset):
    def __init__(self, image_batch, label_batch, image_processor, transform=None):
        self.image_batch = image_batch
        self.label_batch = label_batch
        self.image_processor = image_processor
        self.transform = transform
        print("Custom Dataset created.")

    def __len__(self):
        return len(self.image_batch)

    def __getitem__(self, idx):
        image = self.image_batch[idx]
        label = self.label_batch[idx]

        # Step 1: Preprocess image using the Hugging Face processor (resize, normalization)
        inputs = self.image_processor(image, return_tensors="pt", do_rescale=False)
        pixel_values = inputs["pixel_values"].squeeze(0)  # Remove the batch dimension (since we have a single image)

        # Step 2: Convert the tensor to a PIL Image for augmentation
        pil_image = transforms.ToPILImage()(pixel_values)

        # Step 3: Apply the transformation (augmentation)
        if self.transform:
            pil_image = self.transform(pil_image)

        return {"pixel_values": pil_image, "labels": torch.tensor(label, dtype=torch.long)}


# num labels = len(data_train_raw.class_names)
def create_vision_transformer(data_train, data_val, num_labels):
    model = AutoModelForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=num_labels
    )

    # Step 4: Define training arguments
    training_args = TrainingArguments(
        output_dir="./models",  # Output directory for saved models
        evaluation_strategy="epoch",  # Evaluate after each epoch
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=params.EPOCHS,  # Set the number of epochs you want
        weight_decay=0.01,
        logging_dir="./logs",  # Directory for logs
        save_steps=500,
        logging_steps=100,
        save_total_limit=2,  # Save only the last 2 models to avoid excessive disk usage
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_train,
        eval_dataset=data_val,
    )

    return trainer


def load_and_process_dataset(dataset_name='train'):
    print('📂 Loading dataset:', dataset_name)
    data_raw = tf.keras.preprocessing.image_dataset_from_directory(
        directory=os.path.join(params.DATA_DIR, dataset_name),
        labels='inferred',
        label_mode='categorical',
        batch_size=params.BATCH_SIZE,
        image_size=(224, 224),
    )
    images = []
    labels = []
    num_labels = len(data_raw.class_names)

    for image_batch, label_batch in data_raw:
        images.append(image_batch.numpy() / 255.)  # Convert from TensorFlow tensor to NumPy array
        labels.append(label_batch.numpy())  # Convert from TensorFlow tensor to NumPy array

    # Stack them into single arrays
    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)

    # Convert images to PyTorch tensors
    images = torch.tensor(images).permute(0, 3, 1, 2)  # Convert shape to [batch_size, channels, height, width]
    labels = torch.tensor(np.argmax(labels, axis=1), dtype=torch.long)
    print("✅ Dataset loaded. Turn into CustomDataset.")
    return CustomDataset(images, labels, image_processor, transform=transform_augmentation), num_labels


def train_vision_transformer():
    data_train, num_labels_train = load_and_process_dataset('train')
    data_val, num_labels_val = load_and_process_dataset('validation')
    print("✅ Data loaded and processed.")
    assert num_labels_train == num_labels_val, "The number of labels in the training and validation datasets must be the same."
    trainer = create_vision_transformer(data_train, data_val, num_labels=num_labels_train)
    print("✅ Trainer created.")
    print("🚀 Training...")
    res = trainer.train()
    return res


def load_trained_model(checkpoint_path, num_labels):
    """Load a trained model from a checkpoint."""
    model = AutoModelForImageClassification.from_pretrained(checkpoint_path, num_labels=num_labels)
    model.eval()
    print("✅ Model loaded from checkpoint.")
    return model


def predict(image_path, model, image_processor):
    """Predict the class of an image using the trained model."""
    model.eval()  # Set the model to evaluation mode

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    inputs = image_processor(image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]

    # Perform inference
    with torch.no_grad():
        outputs = model(pixel_values)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()

    return predicted_class


def evaluate_model(model, dataset):
    """Evaluate the model on a given dataset and print metrics."""
    model.eval()  # Set to evaluation mode
    predictions, true_labels = [], []

    for batch in dataset:
        pixel_values = batch["pixel_values"].unsqueeze(0)  # Ensure batch shape
        labels = batch["labels"].item()

        with torch.no_grad():
            outputs = model(pixel_values)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=-1).item()

        predictions.append(predicted_class)
        true_labels.append(labels)
        print(f"Predicted: {predicted_class}")

    # Compute evaluation metrics
    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, zero_division=0, output_dict=True)

    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", report)

    return accuracy, report, predictions, true_labels


if __name__ == '__main__':
    # Evaluate model
    print("🔍 Load the trained model")
    model = load_trained_model("models/checkpoint-64240", num_labels=15)
    print("✅ Model loaded.")
    print("🔍 Load and process the validation dataset")
    data_val, num_labels_val = load_and_process_dataset('predict')
    print("✅ Validation dataset loaded and processed.")
    print("📊 Evaluate the model")
    accuracy, report, predictions, true_labels = evaluate_model(model, data_val)

    # Create confusion matrix
    cm = confusion_matrix(true_labels, predictions, normalize='true')  # Normalize rows to sum to 1
    cm_percentage = cm * 100  # Convert to percentage

    csv_filename = f"transformer_classification_report.csv"
    png_filename = f"transformer_confusion_matrix.png"

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percentage, annot=True, fmt=".2f", cmap="Blues", cbar=True)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (Percentage)")
    plt.savefig(png_filename)
    plt.close()
    print(f"Confusion matrix saved to {png_filename}")

    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(csv_filename, index=True)

    # Predict
