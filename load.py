import os
from datetime import time

import numpy as np
import tensorflow as tf
from PIL import Image

from v2 import CifarModel


def save_img(img):
    img_final = img.numpy() if isinstance(img, tf.Tensor) else img

    # Normalize the image if needed (ensure values are between 0-255 for saving)
    img_final = np.clip(img_final[0] * 255.0, 0, 255).astype(np.uint8)  # assuming batch size of 1

    # Create timestamped filename
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"generated_image_{timestamp}.png"

    # Save the image
    image = Image.fromarray(img_final)
    image.save(os.path.join("generated_images", filename))


def generate_img(model):
    img_final = model.diffusion.p_sample_loop(denoise_fn=model.unet, shape=(1, 32, 32, 3))
    save_img(img_final)


def main():
    model = CifarModel.CifarModel()  # Replace this with your actual model definition
    model.build(input_shape=(None, (32, 32, 3)))

    # Load the saved weights
    model.load_weights('model_weights.weights.h5')

    generate_img(model)


if __name__ == '__main__':
    main()
