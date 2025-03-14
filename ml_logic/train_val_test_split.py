import os
import shutil
import random
import math
import params

TRAIN_DIR = os.path.join(params.DATA_DIR, "train")
TEST_DIR = os.path.join(params.DATA_DIR, "test")
VAL_DIR = os.path.join(params.DATA_DIR, "validation")

# Set random seed to have the same test set every time
random.seed(42)
# Process each class folder
for folder in os.listdir(TRAIN_DIR):
    folder_path = os.path.join(TRAIN_DIR, folder)

    if os.path.isdir(folder_path):  # Ensure it's a folder
        files = os.listdir(folder_path)
        random.shuffle(files)  # Shuffle files

        # Compute ceil(10% of files)
        num_test_files = math.ceil(params.TEST_SPLIT * len(files))
        test_files = files[:num_test_files]  # Select first N shuffled files

        # Create a corresponding class folder in the test directory
        test_folder_path = os.path.join(TEST_DIR, folder)
        os.makedirs(test_folder_path, exist_ok=True)

        # Move test files to the new test folder
        for file in test_files:
            src_path = os.path.join(folder_path, file)
            dst_path = os.path.join(test_folder_path, file)
            shutil.move(src_path, dst_path)


# Relax random seed to have different validation set every time
random.seed(None)
# Process each class folder
for folder in os.listdir(TRAIN_DIR):
    folder_path = os.path.join(TRAIN_DIR, folder)

    if os.path.isdir(folder_path):  # Ensure it's a folder
        files = os.listdir(folder_path)
        random.shuffle(files)  # Shuffle files

        # Compute ceil(10% of files)
        num_val_files = math.ceil(params.VALIDATION_SPLIT / (1 - params.TEST_SPLIT) * len(files))
        val_files = files[:num_val_files]  # Select first N shuffled files

        # Create a corresponding class folder in the val directory
        val_folder_path = os.path.join(VAL_DIR, folder)
        os.makedirs(val_folder_path, exist_ok=True)

        # Move val files to the new val folder
        for file in val_files:
            src_path = os.path.join(folder_path, file)
            dst_path = os.path.join(val_folder_path, file)
            shutil.move(src_path, dst_path)
