import registry
import params
import os
from keras.utils import image_dataset_from_directory
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from google.cloud import storage



def evaluate_model(model_name):
    print("Start evaluation of the model")
    model = registry.load_model(model_name)
    _eval_model(model)


def _eval_model(model):
    # Load evaluation data
    data = image_dataset_from_directory(
        directory=os.path.join(params.DATA_DIR, 'validation'),
        labels='inferred',
        label_mode='categorical',
        batch_size=params.BATCH_SIZE,
        image_size=(224, 224),
        shuffle=False  # Ensure order consistency
    )
    print("✅ Evaluation data loaded.")

    # Get class names
    class_names = data.class_names

    # Extract true labels
    y_true = np.concatenate([y.numpy() for _, y in data])
    y_true = np.argmax(y_true, axis=1)  # Convert one-hot encoding to class indices

    # Get predictions
    y_pred_probs = model.predict(data)
    y_pred = np.argmax(y_pred_probs, axis=1)  # Get predicted class indices

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized_float = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.round(cm_normalized_float * 100).astype(int)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    file_name = f"{params.EVALUATION_MODEL_NAME}_eval_plot.png"
    plt.savefig(file_name)

    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    pd.DataFrame(report_dict).to_csv(f"{params.EVALUATION_MODEL_NAME}_classification_report.csv")

    # Save model to GCS
    client = storage.Client()
    bucket = client.bucket(params.BUCKET_NAME)
    blob = bucket.blob(f"evaluation/{file_name}")
    blob.upload_from_filename(filename=file_name)

    blob2 = bucket.blob(f"evaluation/{params.EVALUATION_MODEL_NAME}_classification_report.csv")
    blob2.upload_from_filename(filename=f"{params.EVALUATION_MODEL_NAME}_classification_report.csv")

    os.remove(file_name)
    os.remove(f"{params.EVALUATION_MODEL_NAME}_classification_report.csv")

    # Evaluate model (optional)
    result = model.evaluate(data, batch_size=params.BATCH_SIZE, verbose="auto")
    print("Evaluation Metrics:", result)

def _eval_model_fredi(model):
    data = image_dataset_from_directory(
        directory=os.path.join(params.DATA_DIR, 'evaluation'),
        labels='inferred',
        label_mode='categorical',
        batch_size=params.BATCH_SIZE,
        image_size=(224, 224),
    )
    print("✅ Evaluation data loaded.")

    result = model.evaluate(
        data,
        batch_size=params.BATCH_SIZE,
        verbose="auto",
    )
    print(result)



if __name__ == '__main__':
    evaluate_model(params.EVALUATION_MODEL_NAME)
