"""
Title: Simple MNIST convnet
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2015/06/19
Last modified: 2020/04/21
Description: A simple convnet that achieves ~99% test accuracy on MNIST.
"""

"""
## Setup
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import keras_interoperability
from n2d2.solver import Adam
from time import time
from os.path import exists
import n2d2
import argparse

# ARGUMENTS PARSING
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, help='Path to the MNIST Dataset')
args = parser.parse_args()

"""
## Prepare the data
"""
start_time = time()
# training parameters
# batch_size = 2
batch_size = 128
epochs = 10
# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)

print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print("y_train shape:", y_train.shape)

"""
## Build the model
"""

tf_model = tf.keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        # layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

# Asking N2D2 to use GPU 0
n2d2.global_variables.cuda_device = 0
n2d2.global_variables.default_model = 'Frame_CUDA'

model = keras_interoperability.wrap(tf_model, batch_size=batch_size, for_export=True)


"""
## Train the model
"""

model.summary()
path_saved_param="./param_lenet"
if exists(path_saved_param):
    print(f"Importing model parameters from {path_saved_param}")
    model.get_deepnet_cell().import_free_parameters(path_saved_param)
else:
    model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    model.get_deepnet_cell().export_free_parameters(path_saved_param)
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

# Importing data for calibration.
database = n2d2.database.MNIST(data_path=args.data_path, validation=0.1)
provider = n2d2.provider.DataProvider(database, [28, 28, 1], batch_size=batch_size)
provider.add_transformation(n2d2.transform.Rescale(width=28, height=28))
print(provider)


# Generating C export
n2d2.export.export_c(
    model.get_deepnet_cell(),
    provider=provider,
    nb_bits=8,
    export_nb_stimuli_max=1,
    calibration=1)

print(f"Script time : {time()-start_time}s")