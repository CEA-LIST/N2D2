"""
    (C) Copyright 2021 CEA LIST. All Rights Reserved.
    Contributor(s): Cyril MOINEAU (cyril.moineau@cea.fr) 
                    Johannes THIELE (johannes.thiele@cea.fr)

    This software is governed by the CeCILL-C license under French law and
    abiding by the rules of distribution of free software.  You can  use,
    modify and/ or redistribute the software under the terms of the CeCILL-C
    license as circulated by CEA, CNRS and INRIA at the following URL
    "http://www.cecill.info".

    As a counterpart to the access to the source code and  rights to copy,
    modify and redistribute granted by the license, users are provided only
    with a limited warranty  and the software's author,  the holder of the
    economic rights,  and the successive licensors  have only  limited
    liability.

    The fact that you are presently reading this means that you have had
    knowledge of the CeCILL-C license and that you accept its terms.
"""

import unittest
import n2d2
import tensorflow as tf
from keras_interoperability import wrap
from tensorflow.keras.layers import MaxPooling2D, Conv2D, Dense, Flatten
from tensorflow.keras import Input
import tensorflow.keras as keras



class test_keras(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self) -> None:
        n2d2.global_variables.cuda_device = 0
        n2d2.global_variables.default_model = 'Frame'

    @unittest.skip("This test is deprecated due to N2D2 not supporting a Pool Layer as an input.")
    def test_propagation(self):
        tf_model = keras.Sequential([
            Input(shape=[3, 3, 2]),
            MaxPooling2D(pool_size=(1, 1))
        ])
        self.model = wrap(tf_model, batch_size=5)
        self.model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])
        self.x = tf.random.uniform([4,3,3,2])
        y = self.model.call(self.x)
        for predicted, truth in zip(y.numpy().flatten(), self.x.numpy().flatten()):
            self.assertEqual(predicted, truth)

    def test_propagation_conv(self):
        tf_model = keras.Sequential([
            Input(shape=[1, 3, 3]),
            Conv2D(3, kernel_size=(1, 1))
        ])
        self.model = wrap(tf_model, batch_size=5)
        self.x = tf.random.uniform([5,1,3,3])
        n2d2_y = self.model.call(self.x)
        tf_y = tf_model.call(self.x)
        print("N2D2 output : ")
        print(n2d2_y)
        print("TF output : ")
        print(tf_y)
        for predicted, truth in zip(n2d2_y.numpy().flatten(), tf_y.numpy().flatten()):
            self.assertTrue((abs(float(predicted) - float(truth)) < (0.01 * (abs(truth)+ 0.0001))))


    def test_backpropagation_conv(self):
        tf_model = keras.Sequential([
            Input(shape=[3, 3, 2]),
            Conv2D(3, kernel_size=(1, 1), use_bias=False)
        ])
        self.model = wrap(tf_model, batch_size=5)
        sgd_opt = tf.keras.optimizers.SGD(
            learning_rate=0.01, momentum=0.0, nesterov=False, name='SGD')
        self.model.compile(loss="categorical_crossentropy", optimizer=sgd_opt, metrics=["accuracy"])
        tf_model.compile(loss="categorical_crossentropy", optimizer=sgd_opt, metrics=["accuracy"])
        self.x = tf.random.uniform([5,3,3,2])
        self.y = tf.random.uniform([5,3,3,3])
        self.xn = tf.identity(self.x)
        self.yn = tf.identity(self.y)
        # for i in tf_model.layers[0].weights:
        #     print(i)
        # for i in self.model._deepnet_cell[0].get_biases():
        #     print(i)
        tf_model.fit(x=self.x, y=self.y, batch_size=5, validation_split=0, epochs=1)
        self.model.fit(x=self.xn, y=self.yn, batch_size=5, validation_split=0, epochs=1)
      
        # for i in tf_model.layers[0].weights:
        #     print(i)
        # for i in self.model._deepnet_cell[0].get_biases():
        #     print(i)
        self.x = tf.random.uniform([25,3,3,2])
        n2d2_y = self.model.call(self.x)
        tf_y = tf_model.call(self.x)
        # print("N2D2 output : ")
        # print(n2d2_y)
        # print("TF output : ")
        # print(tf_y)
        for predicted, truth in zip(n2d2_y.numpy().flatten(), tf_y.numpy().flatten()):
            self.assertTrue((abs(float(predicted) - float(truth)) < (0.01 * (abs(truth)+ 0.0001))), 
            f"After training, N2D2 and Keras weights diverge.")

    def test_propagation_fc(self):
        tf_model = keras.Sequential([
            Input(shape=[3,3,1]),
            Flatten(),
            Dense(9)
        ])
        self.model = wrap(tf_model, batch_size=5)
        self.model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])
        self.x = tf.random.uniform([5,3,3,1])
        n2d2_y = self.model.call(self.x)
        tf_y = tf_model.call(self.x)
        print("N2D2 output : ")
        print(n2d2_y)
        print("TF output : ")
        print(tf_y)
        for predicted, truth in zip(n2d2_y.numpy().flatten(), tf_y.numpy().flatten()):
            self.assertTrue((abs(float(predicted) - float(truth)) < (0.01 * (abs(truth)+ 0.0001))))

if __name__ == '__main__':
    unittest.main()