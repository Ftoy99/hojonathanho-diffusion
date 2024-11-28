import keras
import tensorflow as tf


def main():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    print("ready")



if __name__ == '__main__':
  main()