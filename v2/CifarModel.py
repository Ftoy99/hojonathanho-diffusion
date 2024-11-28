import keras
import tensorflow as tf

from diffusion_tf.diffusion_utils_2 import GaussianDiffusion2, get_beta_schedule
from v2.Unet import build_model


class CifarModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.randflip = True
        self.betas = get_beta_schedule("linear", beta_start=0.0001, beta_end=0.02, num_diffusion_timesteps=1000)
        self.diffusion = GaussianDiffusion2(
            betas=self.betas, model_mean_type="eps", model_var_type="fixedlarge", loss_type="mse")
        self.dropout = 0.1
        self.unet = build_model(
            img_size=32,
            img_channels=3,
            widths=[64 * mult for mult in [1, 2, 4, 8]],
            has_attention=[False, False, True, True],
            num_res_blocks=2,
            norm_groups=8,
            activation_fn=keras.activations.swish)
        self.ema = 0.999
        self.ema_network = build_model(
            img_size=32,
            img_channels=3,
            widths=[64 * mult for mult in [1, 2, 4, 8]],
            has_attention=[False, False, True, True],
            num_res_blocks=2,
            norm_groups=8,
            activation_fn=keras.activations.swish)
        self.ema_network.set_weights(self.unet.get_weights())

    def train_step(self, data):
        x, y = data
        # show_images(images=x, title="Images og")

        # B, H, W, C = tf.shape(x)  # Get sizes batch , height ,width , channels

        # ? how much noise basically in that timestep we progressively will have less noise
        t = tf.random.uniform([tf.shape(x)[0]], minval=0, maxval=self.diffusion.num_timesteps,
                              dtype=tf.int32)  # uniform (get around the same amount of step->1 and step->self.diffusion.num_timesteps)

        with tf.GradientTape() as tape:
            noise = tf.random.normal(shape=tf.shape(x), dtype=x.dtype)

            # show_images(images=noise, title="Noise we will add")

            images_t = self.diffusion.q_sample(x, t, noise)  # Add noise with gausiandiffusion

            # plot them
            # show_images(images=images_t, title="Images with noise")

            # 5. Pass the diffused images and time steps to the network
            pred_noise = self.unet([images_t, t], training=True)

            # 6. Calculate the loss
            loss = self.loss(noise, pred_noise)

        # 7. Get the gradients
        gradients = tape.gradient(loss, self.unet.trainable_weights)

        # update wieghts of unet
        self.optimizer.apply_gradients(zip(gradients, self.unet.trainable_weights))

        # 9. Updates the weight values for the network with EMA weights
        for weight, ema_weight in zip(self.unet.weights, self.ema_network.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        # 10. Return loss values
        return {"loss": loss}

