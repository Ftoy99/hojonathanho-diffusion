import keras
import tensorflow as tf

from keras.losses import MeanSquaredError
from keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint

from v2 import CifarModel


def main():
    # tf.config.run_functions_eagerly(True)  # This is to debug
    # tf.data.experimental.enable_debug_mode()  # This is to debug
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))  # show detected gpu number

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0  # make this from 0-1 and dtype float32 instead of img uint8
    x_test = x_test.astype('float32') / 255.0  # make this from 0-1 and dtype float32 instead of img uint8

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # make the model
    model = CifarModel.CifarModel()

    # Optimizer and loss type
    optimizer = Adam(learning_rate=2e-4, epsilon=1e-8)
    loss_type = MeanSquaredError()  # mse

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_type)  # Adam optimize

    # checkpoint_callback = ModelCheckpoint(
    #     'model_epoch_{epoch:02d}.h5',  # Filename pattern, using epoch number in filename
    #     save_freq='epoch',  # Save the model after every epoch
    #     save_best_only=False,  # Set to True to save only the best model based on a monitored metric
    #     verbose=1,  # Verbosity level (optional)
    #     save_format="tf"
    # )

    model.fit(x_train, y_train, batch_size=100, epochs=10)


if __name__ == '__main__':
    main()
