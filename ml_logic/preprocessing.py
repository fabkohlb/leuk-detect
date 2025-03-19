import tensorflow as tf
import numpy as np
import random
import os
import cv2
from tensorflow.image import central_crop, stateless_random_flip_left_right, stateless_random_flip_up_down, stateless_random_brightness
import params

def augmented_image(img, NUM_AUGMENTATIONS, img_name, brightness=0.5, flip_left_right=0.5, flip_up_down=0.5, cropping=0, gaussian_noise=0,
                    hue=0, saturation=0, contrast=0):
    """
    Augments a given image with different transformation methods applied based on their given probability.
    """

    # Store augmented images
    new_images = {}

    for i in range(NUM_AUGMENTATIONS):
        seed = (i, 0)  # Seed for reproducibility
        augmented_img = img  # Start with original image

        if random.random() < brightness:
            augmented_img = stateless_random_brightness(augmented_img, max_delta=0.35, seed=seed)

        if random.random() < flip_left_right:
            augmented_img = stateless_random_flip_left_right(augmented_img, seed=seed)

        if random.random() < flip_up_down:
            augmented_img = stateless_random_flip_up_down(augmented_img, seed=seed)

        if random.random() < cropping:
            central_fraction = random.uniform(0.8, 0.99)
            augmented_img = central_crop(augmented_img, central_fraction=central_fraction)

        if random.random() < gaussian_noise:
            augmented_img = tf.image.convert_image_dtype(augmented_img, tf.float32)  # Convert to float32
            noise = tf.random.normal(shape=tf.shape(augmented_img), mean=0.0, stddev=0.05)
            augmented_img = tf.add(augmented_img, noise)
            augmented_img = tf.clip_by_value(augmented_img, 0.0, 1.0)  # Keep values in range
            augmented_img = tf.image.convert_image_dtype(augmented_img, tf.uint8)

        if random.random() < hue:
            hue_delta = random.uniform(-0.08, 0.08)
            augmented_img = tf.image.adjust_hue(augmented_img, hue_delta)

        if random.random() < saturation:
            saturation_factor = random.uniform(0.7, 2)
            augmented_img = tf.image.adjust_saturation(augmented_img, saturation_factor)

        if random.random() < contrast:
            contrast_factor = random.uniform(0, 2.5)
            augmented_img = tf.image.adjust_contrast(augmented_img, contrast_factor)

        # Store the final augmented image
        aug_img_name = f"augm_img_{i}_{img_name}"
        new_images[aug_img_name] = augmented_img

    return new_images


def augment_img_class(cell_type: str, num_augm_pics_per_pic: int, brightness=0.5, flip_left_right=0.5, flip_up_down=0.5, cropping=0, gaussian_noise=0,
                      hue=0, saturation=0, contrast=0):
    """
    Augments images of a given cell type and saves them in the corresponding folder.

    Parameters:
    - cell_type (str): The cell category to augment.
    - num_augm_pics_per_pic (int): Number of augmented images to generate per original image.
    - brightness (float): Probability of adjusting brightness (default: 0.5).
    - flip_left_right (float): Probability of flipping left-right (default: 0.5).
    - flip_up_down (float): Probability of flipping up-down (default: 0.5).
    - cropping (float): Probability of cropping (default: 0).
    - gaussian_noise (float): Probability of adding Gaussian noise (default: 0).
    - hue (float): Probability of adjusting hue (randomly slightly shifts the color on the color spectrum) (default: 0).
    - saturation (float): Probability of adjusting saturation (default: 0).
    - contrast (float): Probability of adjusting contrast (default: 0).
    """

    # Try to construct the relative path
    data_dir = os.path.dirname(__file__)  # Get current script directory
    relative_path = os.path.join(data_dir, '..', 'data', 'train')

    # Check if the relative path exists; if not, use the absolute path from params
    if os.path.exists(relative_path):
        cell_dir = os.path.join(relative_path, cell_type)  # Use relative path
    else:
        cell_dir = os.path.join(params.DATA_DIR, 'train', cell_type)  # Use absolute path

    if not os.path.exists(cell_dir):
        print(f"❌ Folder '{cell_type}' not found!")
        return

    img_names = os.listdir(cell_dir)

    for img_name in img_names:
        print(img_name)
        if img_name.endswith('.png'):
            img = cv2.imread(os.path.join(cell_dir, img_name))
            if img is None:
                print(f"⚠ Skipping unreadable image: {img_name}")
                continue

            # Generate augmented images
            augmented_images = augmented_image(img, num_augm_pics_per_pic, img_name, brightness, flip_left_right, flip_up_down,
                                               cropping, gaussian_noise, hue, saturation, contrast)
            for aug_name in augmented_images:
                aug_path = os.path.join(cell_dir, aug_name)  # Save in the same folder
                aug_img_np = tf.cast(augmented_images[aug_name], tf.uint8).numpy()  # Convert tensor to NumPy
                cv2.imwrite(aug_path, aug_img_np)  # Save image

    print(f"✅ Augmentation completed for '{cell_type}'. Images saved in the same folder.\n{len(img_names)*num_augm_pics_per_pic} new images were generated!")


def preprocess_images(images, output_img_size=(224, 224)):
    """
    Preprocesses a batch of images by resizing them.
    """
    new_img = tf.image.resize(images, size=output_img_size, method=tf.image.ResizeMethod.BICUBIC) / 255.
    return new_img.numpy()


if __name__ == '__main__':
    # Test the preprocessing
    augment_img_class('BAS', 10, brightness=0.3, flip_left_right=1.0, flip_up_down=1.0, cropping=0.3,
                  gaussian_noise=0.05, hue=0.05, saturation=0.0, contrast=0.3)

    augment_img_class('EBO', 10, brightness=0.3, flip_left_right=1.0, flip_up_down=1.0, cropping=0.3,
                  gaussian_noise=0.05, hue=0.05, saturation=0.0, contrast=0.3)

    augment_img_class('KSC', 10, brightness=0.3, flip_left_right=1.0, flip_up_down=1.0, cropping=0.3,
                  gaussian_noise=0.05, hue=0.05, saturation=0.0, contrast=0.3)

    augment_img_class('LYA', 10, brightness=0.3, flip_left_right=1.0, flip_up_down=1.0, cropping=0.3,
                  gaussian_noise=0.05, hue=0.05, saturation=0.05, contrast=0.3)

    augment_img_class('MMZ', 10, brightness=0.3, flip_left_right=1.0, flip_up_down=1.0, cropping=0.3,
                  gaussian_noise=0.05, hue=0.05, saturation=0.05, contrast=0.3)

    augment_img_class('MOB', 10, brightness=0.3, flip_left_right=1.0, flip_up_down=1.0, cropping=0.3,
                  gaussian_noise=0.05, hue=0.05, saturation=0.0, contrast=0.3)

    augment_img_class('MYB', 10, brightness=0.3, flip_left_right=1.0, flip_up_down=1.0, cropping=0.3,
                  gaussian_noise=0.05, hue=0.05, saturation=0.0, contrast=0.3)

    augment_img_class('NGB', 10, brightness=0.3, flip_left_right=1.0, flip_up_down=1.0, cropping=0.3,
                  gaussian_noise=0.05, hue=0.05, saturation=0.0, contrast=0.3)

    augment_img_class('PMB', 10, brightness=0.3, flip_left_right=1.0, flip_up_down=1.0, cropping=0.3,
                  gaussian_noise=0.05, hue=0.05, saturation=0.0, contrast=0.3)

    augment_img_class('PMO', 10, brightness=0.3, flip_left_right=1.0, flip_up_down=1.0, cropping=0.3,
                  gaussian_noise=0.05, hue=0.05, saturation=0.0, contrast=0.3)

    print('Done')
