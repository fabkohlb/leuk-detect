from transformers import Trainer, TrainingArguments, AutoImageProcessor
from transformers import AutoModelForImageClassification
from torch.utils.data import Dataset
from torchvision import transforms
import tensorflow as tf
import os
import torch
import numpy as np
import params

print("🤗 Load AutoImageProcessor")
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
print("✅ Image processor loaded.")


transform_augmentation = transforms.Compose([
    transforms.RandomCrop(224),  # Random crop
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.RandomRotation(10),  # Random rotation (adjust as needed)
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Random resized crop
    transforms.RandomVerticalFlip(),  # Random vertical flip
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])


class CustomDataset(Dataset):
    def __init__(self, image_batch, label_batch, image_processor, transform=None):
        self.image_batch = image_batch
        self.label_batch = label_batch
        self.image_processor = image_processor
        self.transform = transform

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
        per_device_train_batch_size=params.BATCH_SIZE,
        per_device_eval_batch_size=params.BATCH_SIZE,
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


if __name__ == '__main__':
    train_vision_transformer()
