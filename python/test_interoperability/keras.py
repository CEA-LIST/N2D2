"""
Source : https://keras.io/examples/vision/mnist_convnet/
"""
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from tensorflow.compat.v1 import enable_eager_execution

import n2d2
import N2D2
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow


class custom_Conv(layers.Layer):

    _initialized = False
    def __init__(self, size_out, kernelDims=[1, 1], strideDims=[1,1], paddingDims=[1,1]):
        super(custom_Conv, self).__init__()
        net = N2D2.Network()
        deepNet = N2D2.DeepNet(net)
        cell = N2D2.ConvCell_Frame_float(deepNet, "name", kernelDims, size_out, strideDims=strideDims, paddingDims=paddingDims)
        self._n2d2 = cell

        
    def call(self, inputs):
        # TODO : Need to change the shape on the input so that the first pass give the good shape
        if tensorflow.executing_eagerly():
            data_type = inputs.dtype.name
            # Ugly ?
            if data_type == "int32" or data_type == "int64":
                data_type = int
            elif data_type == "float32" or data_type == "float64":
                data_type = float
            else:
                raise TypeError("Unknown type :", data_type)
            numpy_tensor = tensorflow.convert_to_tensor(inputs).numpy()
            n2d2_tensor = n2d2.tensor.Tensor([3, 3], DefaultDataType=float)
            n2d2_tensor.from_numpy(numpy_tensor)
            print("InputShape :", n2d2_tensor.shape())
            OutputDims = n2d2_tensor.copy()
            self._n2d2.clearInputs()
            self._n2d2.addInputBis(n2d2_tensor.N2D2(), OutputDims.N2D2())
            if not self._initialized:
                self._n2d2.initialize()
                
            self._n2d2.propagate()
            outputs = self._n2d2.getOutputs()
            t_outputs = n2d2.tensor.Tensor([3, 3], DefaultDataType=data_type)
            t_outputs.from_N2D2(outputs)
            print("OutputShape :", t_outputs.shape())
            inputs = tensorflow.convert_to_tensor(t_outputs.to_numpy())
        return inputs

    def get_config(self):
        return {}

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

# TODO: Here the eager execution is disabled (why ?)
enable_eager_execution()
model = keras.Sequential()
model.run_eagerly= True
model.add(keras.Input(shape=input_shape))
model.add(layers.ConvDepthWise(32, kernel_size=(3, 3), activation="tanh"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(custom_Conv(64, kernelDims=[3, 3]))
# model.add(layers.ConvDepthWise(64, kernel_size=(3, 3), activation="tanh"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation="softmax"))


model.summary()

# input("Press Enter to train.")

batch_size = 64
epochs = 15

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"], run_eagerly=True)

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])