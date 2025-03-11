import os
import cv2
import numpy as np
import ml_logic.params as params


def load_full_data_set(data_dir, limit_num_img_per_class=10000):
    """
    This function loads the whole dataset and returns it in the format X, y
    Loading the full dataset takes around 35 seconds.
    """
    file_names = load_all_filenames(data_dir)

    X = []
    y = []
    for k in file_names.keys():
        for img_file_name in file_names[k][:limit_num_img_per_class]:
            X.append(cv2.imread(os.path.join(data_dir, k, img_file_name)))
        y.append(list(file_names.keys()).index(k))
    X = np.asarray(X)
    y = np.asarray(y)
    return X, y

def load_all_filenames(data_dir):
    # Get classes
    class_folder_list = sorted(os.listdir(data_dir))

    # Get all file_names
    file_names = {}
    for folder in class_folder_list:
        if os.path.isdir(os.path.join(data_dir, folder)):
            # Check if folder name has 3 letters (like the class labels)
            if len(folder) == 3:
                file_list = os.listdir(os.path.join(data_dir, folder))
                # Check if folder contains files
                if len(file_list) > 0:
                    file_names[folder] = file_list
    return file_names



if __name__ == '__main__':
    X, y = load_full_data_set(params.DATA_DIR)
    print(f"Shape X: {X.shape}")
    print(f"Shape y: {y.shape}")
