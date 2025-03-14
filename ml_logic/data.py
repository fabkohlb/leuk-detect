import os
import cv2
import numpy as np
import ml_logic.params # i changed it to from . so i could acces it from my notebook
#import ml_logic.params as params
import tensorflow as tf

def load_full_data_set_Freddi(data_dir, limit_num_img_per_class=10000):
    """
    This function loads the whole dataset and returns it in the format X, y
    Loading the full dataset takes around 35 seconds.
    """
    file_names = load_all_filenames_Freddi(data_dir)

    X = []
    y = []
    for k in file_names.keys():
        for img_file_name in file_names[k][:limit_num_img_per_class]:
            X.append(cv2.imread(os.path.join(data_dir, k, img_file_name)))
        y.append(list(file_names.keys()).index(k))
    X = np.asarray(X)
    y = np.asarray(y)
    return X, y

def load_all_filenames_Freddi(data_dir):
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

def load_dataset(val_split=0.3):
    """Returns the train_data, validation_data and test_data => you can input these directly in each method as they contain X, y"""
    #load train_data and validation_data
    train_data = tf.keras.preprocessing.image_dataset_from_directory(params.DATA_SET, image_size=(224,224), label_mode='categorical', subset='training', validation_split=val_split, seed=1)
    validation_data = tf.keras.preprocessing.image_dataset_from_directory(params.DATA_SET, image_size=(224,224), label_mode='categorical', subset='validation', validation_split=val_split, seed=1)

    #split validation data into val and test data
    # n = tf.data.experimental.cardinality(val_data)
    # test_data = val_data.take((2*n) // 3)
    # validation_data = val_data.skip((2*n) // 3)
    return train_data, validation_data

if __name__ == '__main__':
    # X, y = load_full_data_set(params.DATA_DIR)
    # print(f"Shape X: {X.shape}")
    # print(f"Shape y: {y.shape}")
    #print(load_all_filenames_Freddi(params.DATA_DIR))
    print(load_dataset())
