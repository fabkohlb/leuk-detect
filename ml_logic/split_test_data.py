import os
import shutil
import random
import math
import params

# Set random seed to fix the test set
random.seed(42)

# Process each class folder
print(params.DATA_DIR)
TRAIN_DIR = os.path.join(params.DATA_DIR, "train")
TEST_DIR = os.path.join(params.DATA_DIR, "test")

for folder in os.listdir(TRAIN_DIR):
    folder_path = os.path.join(TRAIN_DIR, folder)

    if os.path.isdir(folder_path):  # Ensure it's a folder
        files = os.listdir(folder_path)
        random.shuffle(files)  # Shuffle files

        # Compute ceil(10% of files)
        num_test_files = math.ceil(0.1 * len(files))
        test_files = files[:num_test_files]  # Select first N shuffled files

        # Create a corresponding class folder in the test directory
        test_folder_path = os.path.join(TEST_DIR, folder)
        os.makedirs(test_folder_path, exist_ok=True)

        # Move test files to the new test folder
        for file in test_files:
            src_path = os.path.join(folder_path, file)
            dst_path = os.path.join(test_folder_path, file)
            shutil.move(src_path, dst_path)
