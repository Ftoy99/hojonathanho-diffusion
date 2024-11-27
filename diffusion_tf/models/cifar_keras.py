import functools
import warnings

import keras
import numpy as np
import tensorflow as tf
from keras.src import tree

from keras.src.trainers.data_adapters import data_adapter_utils
from diffusion_tf.models import unet
from diffusion_tf.diffusion_utils_2 import GaussianDiffusion2


class CifarKerasModel(keras.Model):
    def __init__(self, *args, model_name, betas: np.ndarray, model_mean_type: str, model_var_type: str, loss_type: str,
                 num_classes: int, dropout: float, randflip, **kwargs):
        super().__init__()
        self.model_name = model_name
        self.diffusion = GaussianDiffusion2(
            betas=betas, model_mean_type=model_mean_type, model_var_type=model_var_type, loss_type=loss_type)
        self.num_classes = num_classes
        self.dropout = dropout
        self.randflip = randflip
        self.random_flip = keras.layers.RandomFlip()
        self.dense1 = keras.layers.Dense(32, activation="relu")
        self.dense2 = keras.layers.Dense(10, activation="softmax")
        self.dropout = keras.layers.Dropout(0.5)

    def train_step(self, data):
        '''
        Originally train takes input and step count to noise/denoise must addjust this method for call
        '''
        print(data)
        x, y, sample_weight = data_adapter_utils.unpack_x_y_sample_weight(data)

        # Forward pass
        with tf.GradientTape() as tape:
            if self._call_has_training_arg:
                y_pred = self(x, y, training=True)
            else:
                y_pred = self(x, y)
            loss = self._compute_loss(
                x=x,
                y=y,
                y_pred=y_pred,
                sample_weight=sample_weight,
                training=True,
            )
            self._loss_tracker.update_state(
                loss, sample_weight=tf.shape(tree.flatten(x)[0])[0]
            )
            if self.optimizer is not None:
                loss = self.optimizer.scale_loss(loss)

        # Compute gradients
        if self.trainable_weights:
            trainable_weights = self.trainable_weights
            gradients = tape.gradient(loss, trainable_weights)

            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_weights))
        else:
            warnings.warn("The model does not have any trainable weights.")

        return self.compute_metrics(x, y, y_pred, sample_weight=sample_weight)

    def call(self, inputs, label, training=False, **kwargs):
        B, H, W, C = inputs.shape
        # To breakpoint and debug in here you must have eager execution on
        if self.randflip:
            inputs = self.random_flip(inputs)

        t = tf.random.uniform([B], 0, self.diffusion.num_timesteps, dtype=tf.int32)

        losses = self.diffusion.training_losses(
            denoise_fn=functools.partial(self._denoise, y=label, dropout=self.dropout), x_start=inputs, t=t)

        assert losses.shape == t.shape == [B]
        return {'loss': tf.reduce_mean(losses)}

        # inputs = self.dense1(inputs)
        # inputs = self.dropout(inputs, training=training)  # This is to prevent overfitting , dont run when inference
        # return self.dense2(inputs)

    def _denoise(self, x, t, y, dropout):
        B, H, W, C = x.shape.as_list()
        assert x.dtype == tf.float32
        assert t.shape == [B] and t.dtype in [tf.int32, tf.int64]
        assert y.shape == [B] and y.dtype in [tf.int32, tf.int64]
        out_ch = (C * 2) if self.diffusion.model_var_type == 'learned' else C
        y = None
        if self.model_name == 'unet2d16b2':  # 35.7M
            # return unet.model(
            #     x, t=t, y=y, name='model', ch=128, ch_mult=(1, 2, 2, 2), num_res_blocks=2, attn_resolutions=(16,),
            #     out_ch=out_ch, num_classes=self.num_classes, dropout=dropout
            # )
            return unet.model(
                x, t=t, y=y, name='model', ch=128, ch_mult=(1, 2, 2, 2), num_res_blocks=2, attn_resolutions=(16,),
                out_ch=out_ch, num_classes=1, dropout=dropout
            )
        raise NotImplementedError(self.model_name)

    def train_fn(self, x, y):
        B, H, W, C = x.shape
        if self.randflip:
            x = tf.image.random_flip_left_right(x)
            assert x.shape == [B, H, W, C]
        t = tf.random_uniform([B], 0, self.diffusion.num_timesteps, dtype=tf.int32)
        losses = self.diffusion.training_losses(
            denoise_fn=functools.partial(self._denoise, y=y, dropout=self.dropout), x_start=x, t=t)
        assert losses.shape == t.shape == [B]
        return {'loss': tf.reduce_mean(losses)}

    def samples_fn(self, dummy_noise, y):
        return {
            'samples': self.diffusion.p_sample_loop(
                denoise_fn=functools.partial(self._denoise, y=y, dropout=0),
                shape=dummy_noise.shape.as_list(),
                noise_fn=tf.random_normal
            )
        }

    def progressive_samples_fn(self, dummy_noise, y):
        samples, progressive_samples = self.diffusion.p_sample_loop_progressive(
            denoise_fn=functools.partial(self._denoise, y=y, dropout=0),
            shape=dummy_noise.shape.as_list(),
            noise_fn=tf.random_normal
        )
        return {'samples': samples, 'progressive_samples': progressive_samples}

    def bpd_fn(self, x, y):
        total_bpd_b, terms_bpd_bt, prior_bpd_b, mse_bt = self.diffusion.calc_bpd_loop(
            denoise_fn=functools.partial(self._denoise, y=y, dropout=0),
            x_start=x
        )
        return {
            'total_bpd': total_bpd_b,
            'terms_bpd': terms_bpd_bt,
            'prior_bpd': prior_bpd_b,
            'mse': mse_bt
        }
