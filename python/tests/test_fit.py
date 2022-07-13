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
from n2d2.cells import Sequence
from random import randint
from os import getenv

DATA_PATH = getenv("N2D2_DATA")
if DATA_PATH is None:
    DATA_PATH="/local/DATABASE/"

class fit_FRAME(unittest.TestCase):
    """This test the method fit and checks if it produces the same result as a learning loop we can manually do with the API.
    """
    def setUp(self):
        self.seed = randint(1, 1000)
        print(f"Random seed : {self.seed}")
        n2d2.global_variables.seed = self.seed
        n2d2.global_variables.default_model = "Frame"

    def tearDown(self):
        pass

    def test_weights_equality(self):
        batch_size = 256
        epochs = 1
        database = n2d2.database.MNIST(data_path=f"{DATA_PATH}mnist")
        provider = n2d2.provider.DataProvider(database, [28, 28, 1], batch_size=batch_size)
        n2d2.global_variables.seed = self.seed # reset seed before defining each model to have the same weights !
        model = Sequence([n2d2.cells.Fc(28*28, 50, activation=n2d2.activation.Rectifier(), no_bias=True, weights_filler=n2d2.filler.Constant(value=0.01)),
                            n2d2.cells.Fc(50, 10, no_bias=True, weights_filler=n2d2.filler.Constant(value=0.01)),
                            n2d2.cells.nn.Softmax(with_loss=True)        
                            ], "model1")

        n2d2.global_variables.seed = self.seed
        model_f = Sequence([n2d2.cells.Fc(28*28, 50, activation=n2d2.activation.Rectifier(), no_bias=True, weights_filler=n2d2.filler.Constant(value=0.01)),
                            n2d2.cells.Fc(50, 10, no_bias=True, weights_filler=n2d2.filler.Constant(value=0.01)),
                            n2d2.cells.nn.Softmax(with_loss=True)        
                            ], "model2")

        # Training using fit method !
        # converting sequence into Deepnet
        modelf = model_f.to_deepnet_cell(provider)
        n2d2.global_variables.seed = self.seed # reset seed for random batch reading !
        modelf.fit(learn_epoch=epochs, valid_metric='Accuracy')
        fit_fc1_weights =  modelf[0].get_weights()

        target = n2d2.target.Score(provider)

        provider.set_partition("Learn")
        model.learn()
        n2d2.global_variables.seed = self.seed # reset seed for random batch reading !
        provider.set_batch()
        for epoch in range(epochs):

            print("\n### Train Epoch: " + str(epoch) + " ###")

            i = 0
            while not provider.all_batchs_provided():
                x = provider.read_batch()
                x = model(x)
                x = target(x)
                x.back_propagate()
                x.update()

                print("Example: " + str(i) + ", loss: "
                    + "{0:.3f}".format(target.loss()), end='\r')
                i += 1
        print("\n")
        python_loop_fc1_weights =  model[0].get_weights()

        for i,j in zip(python_loop_fc1_weights, fit_fc1_weights):
            for tensor1, tensor2 in zip(i, j):
                for value1, value2 in zip(tensor1, tensor2):
                    print(f"LOOP {value1} | FIT {value2}")
                    self.assertTrue(abs(value1-value2) < (0.01 * abs(value2)) + 0.001, f"Different weights value found :\nLOOP {value1} | FIT {value2}")

class fit_FRAME_CUDA(unittest.TestCase):
    """This test the method fit and checks if it produces the same result as a learning loop we can manually do with the API.
    """
    def setUp(self):
        self.seed = randint(1, 1000)
        print(f"Random seed : {self.seed}")
        n2d2.global_variables.seed = self.seed
        n2d2.global_variables.default_model = "Frame_CUDA"

    def tearDown(self):
        n2d2.global_variables.default_model = "Frame"

    def test_weights_equality(self):
        batch_size = 256
        epochs = 1
        database = n2d2.database.MNIST(data_path=f"{DATA_PATH}mnist")
        provider = n2d2.provider.DataProvider(database, [28, 28, 1], batch_size=batch_size)
        n2d2.global_variables.seed = self.seed # reset seed before defining each model to have the same weights !
        model = Sequence([n2d2.cells.Fc(28*28, 50, activation=n2d2.activation.Rectifier(), no_bias=True, weights_filler=n2d2.filler.Constant(value=0.01)),
                            n2d2.cells.Fc(50, 10, weights_filler=n2d2.filler.Constant(value=0.01)),
                            n2d2.cells.nn.Softmax(with_loss=True)        
                            ], "model1")
        n2d2.global_variables.seed = self.seed
        model_f = Sequence([n2d2.cells.Fc(28*28, 50, activation=n2d2.activation.Rectifier(), no_bias=True, weights_filler=n2d2.filler.Constant(value=0.01)),
                            n2d2.cells.Fc(50, 10, no_bias=True, weights_filler=n2d2.filler.Constant(value=0.01)),
                            n2d2.cells.nn.Softmax(with_loss=True)        
                            ], "model2")
        # Training using fit method !
        # converting sequence into Deepnet
        modelf = model_f.to_deepnet_cell(provider)
        n2d2.global_variables.seed = self.seed # reset seed for random batch reading !
        modelf.fit(learn_epoch=epochs, valid_metric='Accuracy')
        fit_fc1_weights =  modelf[0].get_weights()

        target = n2d2.target.Score(provider)

        provider.set_partition("Learn")
        model.learn()
        n2d2.global_variables.seed = self.seed # reset seed for random batch reading !
        provider.set_batch()
        for epoch in range(epochs):

            print("\n### Train Epoch: " + str(epoch) + " ###")

            i = 0
            while not provider.all_batchs_provided():
                x = provider.read_batch()
                x = model(x)
                x = target(x)
                x.back_propagate()
                x.update()

                print("Example: " + str(i) + ", loss: "
                    + "{0:.3f}".format(target.loss()), end='\r')
                i += 1
        python_loop_fc1_weights =  model[0].get_weights()

        for i,j in zip(python_loop_fc1_weights, fit_fc1_weights):
            for tensor1, tensor2 in zip(i, j):
                for value1, value2 in zip(tensor1, tensor2):
                    print(f"LOOP {value1} | FIT {value2}")
                    self.assertTrue(abs(value1-value2) < (0.01 * abs(value2)) + 0.001, f"Different weights value found :\nLOOP {value1} | FIT {value2}")

if __name__ == '__main__':
    """
    You need to add this line for the tests to be run.
    """
    unittest.main()