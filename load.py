import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from PIL import Image

from v2 import CifarModel


def save_img(img):
    img_final = img.numpy() if isinstance(img, tf.Tensor) else img

    # Normalize the image if needed (ensure values are between 0-255 for saving)
    img_final = np.clip(img_final[0] * 255.0, 0, 255).astype(np.uint8)  # assuming batch size of 1

    # Create timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    filename = f"generated_image_{timestamp}.png"

    # Save the image
    image = Image.fromarray(img_final)
    image.save(os.path.join("generated_images", filename))


def generate_img(model, num_images=1):
    # 1. Randomly sample noise (starting point for reverse process)
    samples = tf.random.normal(
        shape=(num_images, 32, 32, 3), dtype=tf.float32
    )
    # 2. Sample from the model iteratively
    for t in reversed(range(0, model.diffusion.num_timesteps)):
        tt = tf.cast(tf.fill(num_images, t), dtype=tf.int64)
        pred_noise = model.ema_network.predict(
            [samples, tt], verbose=0, batch_size=num_images
        )

        samples = model.diffusion.p_sample_v2(
            pred_noise, samples, tt, clip_denoised=True
        )
        save_img(samples)
    # 3. Return generated samples
    save_img(samples)


def main():
    model = CifarModel.CifarModel()  # Replace this with your actual model definition
    model.build(input_shape=(None, (32, 32, 3)))

    # Load the saved weights
    model.load_weights('model_weights.weights.h5')

    samples = generate_img(model)

if __name__ == '__main__':
    main()
