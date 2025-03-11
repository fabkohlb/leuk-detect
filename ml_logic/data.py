import os
import cv2
import numpy as np
import ml_logic.params as params


def load_full_data_set(data_dir):
    """
    This function loads the whole dataset and returns it in the format X, y
    """
    file_names = load_all_filenames(data_dir)
    return None

def load_all_filenames(data_dir):
    # Get classes
    class_list = os.listdir(params.DATA_DIR)
    class_list.remove('.DS_Store')
    print(f"Number of classes: {len(class_list)}")

    # Get all file_names
    file_names = {}
    for cl in class_list:
        file_names[cl] = os.listdir(os.path.join(params.DATA_DIR, cl))
    return file_names



if __name__ == '__main__':
    load_full_data_set(params.DATA_DIR)
