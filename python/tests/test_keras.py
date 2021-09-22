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
import tensorflow as tf
import n2d2
import N2D2
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras import Input

class test_keras(unittest.TestCase):
    def setUp(self):
        self.model = n2d2.keras.CustomSequential([
            Input(shape=[2, 2, 1]),
            MaxPooling2D(pool_size=(1, 1))
        ], batch_size=1)
        self.model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])
        self.x = tf.random.uniform([1,2,2,1])
    def test_propagation_is_working(self):
        y = self.model.call(self.x)
        print("Predicted :")
        print(y)
        print("Expected :")
        print(self.x)
        for predicted, truth in zip(y.numpy().flatten(), self.x.numpy().flatten()):
            self.assertEqual(predicted, truth)

if __name__ == '__main__':
    unittest.main()