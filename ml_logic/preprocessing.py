import tensorflow as tf
import numpy as np


def preprocess_images(images, output_img_size=(224, 224)):
    """
    This preprocessing function takes a batch of images (expected to be all in the same shape)
    and preprocesses all images.
    """
    new_img = tf.image.resize(images, size=output_img_size, method=tf.image.ResizeMethod.BICUBIC) / 255.
    return new_img.numpy()



if __name__ == '__main__':
    # Test the preprocessing
    print('### Test the preprocessing routine')
