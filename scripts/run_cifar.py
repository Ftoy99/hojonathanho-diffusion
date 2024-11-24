"""
Unconditional CIFAR10

python3 scripts/run_cifar.py train --bucket_name_prefix $BUCKET_PREFIX --exp_name $EXPERIMENT_NAME --tpu_name $TPU_NAME
python3 scripts/run_cifar.py evaluation --bucket_name_prefix $BUCKET_PREFIX --tpu_name $EVAL_TPU_NAME --model_dir $MODEL_DIR
"""
import functools

import numpy as np
import PIL as pil
import tensorflow as tf
from PIL import Image

from diffusion_tf import utils
from diffusion_tf.diffusion_utils_2 import get_beta_schedule
from diffusion_tf.models.cifar_keras import CifarKerasModel
from diffusion_tf.models.cifar_og_model import Model

from diffusion_tf.tpu_utils import tpu_utils, datasets, simple_eval_worker


def _load_model(kwargs, ds):
    return Model(
        model_name=kwargs['model_name'],
        betas=get_beta_schedule(
            kwargs['beta_schedule'], beta_start=kwargs['beta_start'], beta_end=kwargs['beta_end'],
            num_diffusion_timesteps=kwargs['num_diffusion_timesteps']
        ),
        model_mean_type=kwargs['model_mean_type'],
        model_var_type=kwargs['model_var_type'],
        loss_type=kwargs['loss_type'],
        num_classes=ds.num_classes,
        dropout=kwargs['dropout'],
        randflip=kwargs['randflip']
    )


def simple_eval(model_dir, tpu_name, bucket_name_prefix, mode, load_ckpt, total_bs=256,
                tfds_data_dir='tensorflow_datasets'):
    # region = utils.get_gcp_region()
    # tfds_data_dir = 'gs://{}-{}/{}'.format(bucket_name_prefix, region, tfds_data_dir)
    kwargs = tpu_utils.load_train_kwargs(model_dir)
    print('loaded kwargs:', kwargs)
    ds = datasets.get_dataset(kwargs['dataset'], tfds_data_dir=tfds_data_dir)
    worker = simple_eval_worker.SimpleEvalWorker(
        tpu_name=tpu_name, model_constructor=functools.partial(_load_model, kwargs=kwargs, ds=ds),
        total_bs=total_bs, dataset=ds)
    worker.run(mode=mode, logdir=model_dir, load_ckpt=load_ckpt)


def evaluation(  # evaluation loop for use during training
        model_dir, tpu_name, bucket_name_prefix, once=False, dump_samples_only=False, total_bs=256,
        tfds_data_dir='tensorflow_datasets', load_ckpt=None
):
    # region = utils.get_gcp_region()
    # tfds_data_dir = 'gs://{}-{}/{}'.format(bucket_name_prefix, region, tfds_data_dir)
    kwargs = tpu_utils.load_train_kwargs(model_dir)
    print('loaded kwargs:', kwargs)
    ds = datasets.get_dataset(kwargs['dataset'], tfds_data_dir=tfds_data_dir)
    worker = tpu_utils.EvalWorker(
        tpu_name=tpu_name,
        model_constructor=functools.partial(_load_model, kwargs=kwargs, ds=ds),
        total_bs=total_bs, inception_bs=total_bs, num_inception_samples=50000,
        dataset=ds,
    )
    worker.run(
        logdir=model_dir, once=once, skip_non_ema_pass=True, dump_samples_only=dump_samples_only, load_ckpt=load_ckpt)


def train(
        model_name='unet2d16b2', dataset='cifar10',
        optimizer='adam', total_bs=128, grad_clip=1., lr=2e-4, warmup=5000,
        num_diffusion_timesteps=1000, beta_start=0.0001, beta_end=0.02, beta_schedule='linear',
        model_mean_type='eps', model_var_type='fixedlarge', loss_type='mse',
        dropout=0.1, randflip=1,
        tfds_data_dir='tensorflow_datasets', log_dir='logs', keep_checkpoint_max=2):
    kwargs = dict(locals())

    # Get dataset
    ds, ds_labels = datasets.get_dataset("cifar10")

    # model_constructor = lambda: Model(
    #     model_name=model_name,
    #     betas=get_beta_schedule(
    #         beta_schedule, beta_start=beta_start, beta_end=beta_end, num_diffusion_timesteps=num_diffusion_timesteps
    #     ),
    #     model_mean_type=model_mean_type,
    #     model_var_type=model_var_type,
    #     loss_type=loss_type,
    #     num_classes=ds.num_classes,
    #     dropout=dropout,
    #     randflip=randflip
    # )

    # tpu_utils.run_training(
    #     exp_name='{dataset}_{model_name}_{optimizer}_bs{total_bs}_lr{lr}w{warmup}_beta{beta_start}-{beta_end}-{beta_schedule}_t{num_diffusion_timesteps}_{model_mean_type}-{model_var_type}-{loss_type}_dropout{dropout}_randflip{randflip}'.format(
    #         **kwargs),
    #     model_constructor=model_constructor,
    #     optimizer=optimizer, total_bs=total_bs, lr=lr, warmup=warmup, grad_clip=grad_clip,
    #     train_input_fn=ds.train_input_fn,
    #     dump_kwargs=kwargs, iterations_per_loop=2000,
    #     keep_checkpoint_max=keep_checkpoint_max
    # )

    # this is to check if images are ok and flat it
    images = []
    labels = []
    for batch in ds:
        for img_bytes in batch[b'data']:
            # Convert img_bytes (a 1D array of 3072) into a 32x32x3 image
            r = img_bytes[:1024].reshape(32, 32)
            g = img_bytes[1024:2048].reshape(32, 32)
            b = img_bytes[2048:].reshape(32, 32)

            # Stack channels into a single 32x32x3 array
            img_array = np.stack([r, g, b], axis=-1).astype(np.uint8)
            images.append(img_array)
        for label in batch[b'labels']:
            labels.append(label)

    # Convert images and labels to NumPy arrays
    images = np.array(images)
    labels = np.array(labels)

    # Define the split ratio
    split_ratio = 0.8  # 80% training, 20% testing
    num_samples = len(images)
    split_index = int(num_samples * split_ratio)

    # Split the data
    train_images, test_images = images[:split_index], images[split_index:]
    train_labels, test_labels = labels[:split_index], labels[split_index:]

    # Create TensorFlow datasets
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

    # Shuffle, batch, and prefetch
    train_ds = train_ds.shuffle(1000).batch(100).prefetch(tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.batch(100).prefetch(tf.data.experimental.AUTOTUNE)

    # Make model
    cifar_keras_model = CifarKerasModel(
        model_name=model_name,
        betas=get_beta_schedule(
            beta_schedule, beta_start=beta_start, beta_end=beta_end, num_diffusion_timesteps=num_diffusion_timesteps
        ),
        model_mean_type=model_mean_type,
        model_var_type=model_var_type,
        loss_type=loss_type,
        num_classes=len(ds_labels),
        dropout=dropout,
        randflip=randflip)

    cifar_keras_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',  # Suitable for integer labels
        metrics=['accuracy'])  # maybe add optimizer loss and stuff

    # Train the model
    history = cifar_keras_model.fit(train_ds, epochs=20)

    # Evaluate the model
    test_loss, test_accuracy = cifar_keras_model.evaluate(test_ds)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
