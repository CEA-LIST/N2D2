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
from math import ceil
from n2d2.cells import Sequence
from random import randint

class fit_FRAME(unittest.TestCase):
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
        database = n2d2.database.MNIST(data_path="/nvme0/DATABASE/MNIST/raw")
        provider = n2d2.provider.DataProvider(database, [28, 28, 1], batch_size=batch_size)
        n2d2.global_variables.seed = self.seed # reset seed before defining each model to have the same weights !
        model = Sequence([n2d2.cells.Fc(28*28, 50, activation=n2d2.activation.Rectifier()),
                            n2d2.cells.Fc(50, 10),
                            n2d2.cells.nn.Softmax(with_loss=True)        
                            ], "model1")
        n2d2.global_variables.seed = self.seed
        model_f = Sequence([n2d2.cells.Fc(28*28, 50, activation=n2d2.activation.Rectifier()),
                            n2d2.cells.Fc(50, 10),
                            n2d2.cells.nn.Softmax(with_loss=True)        
                            ], "model2")

        # Training using fit method !
        # converting sequence into Deepnet
        modelf = model_f.to_deepnet_cell(provider)
        modelf.fit(learn_epoch=epochs, valid_metric='Accuracy')
        fit_fc1_weights =  modelf[0].get_weights()

        target = n2d2.target.Score(provider)

        provider.set_partition("Learn")
        model.learn()
        for epoch in range(epochs):

            print("\n### Train Epoch: " + str(epoch) + " ###")

            for i in range(ceil(database.get_nb_stimuli('Learn')/batch_size)):

                x = provider.read_random_batch()
                x = model(x)
                x = target(x)
                x.back_propagate()
                x.update()
                print("Example: " + str(i*batch_size) + ", loss: "
                    + "{0:.3f}".format(target.loss()), end='\r')
        python_loop_fc1_weights =  model[0].get_weights()

        for i,j in zip(python_loop_fc1_weights, fit_fc1_weights):
            for tensor1, tensor2 in zip(i, j):
                for value1, value2 in zip(tensor1, tensor2):
                    self.assertTrue(abs(value1-value2) < (0.01 * abs(value2)) + 0.001, f"Different weights value found :\nLOOP {value1} | FIT {value2}")

class fit_FRAME_CUDA(unittest.TestCase):
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
        database = n2d2.database.MNIST(data_path="/nvme0/DATABASE/MNIST/raw")
        provider = n2d2.provider.DataProvider(database, [28, 28, 1], batch_size=batch_size)
        n2d2.global_variables.seed = self.seed # reset seed before defining each model to have the same weights !
        model = Sequence([n2d2.cells.Fc(28*28, 50, activation=n2d2.activation.Rectifier()),
                            n2d2.cells.Fc(50, 10),
                            n2d2.cells.nn.Softmax(with_loss=True)        
                            ], "model1")
        n2d2.global_variables.seed = self.seed
        model_f = Sequence([n2d2.cells.Fc(28*28, 50, activation=n2d2.activation.Rectifier()),
                            n2d2.cells.Fc(50, 10),
                            n2d2.cells.nn.Softmax(with_loss=True)        
                            ], "model2")

        # Training using fit method !
        # converting sequence into Deepnet
        modelf = model_f.to_deepnet_cell(provider)
        modelf.fit(learn_epoch=epochs, valid_metric='Accuracy')
        fit_fc1_weights =  modelf[0].get_weights()

        target = n2d2.target.Score(provider)

        provider.set_partition("Learn")
        model.learn()
        for epoch in range(epochs):

            print("\n### Train Epoch: " + str(epoch) + " ###")

            for i in range(ceil(database.get_nb_stimuli('Learn')/batch_size)):

                x = provider.read_random_batch()
                x = model(x)
                x = target(x)
                x.back_propagate()
                x.update()

                print("Example: " + str(i*batch_size) + ", loss: "
                    + "{0:.3f}".format(target.loss()), end='\r')
        python_loop_fc1_weights =  model[0].get_weights()

        for i,j in zip(python_loop_fc1_weights, fit_fc1_weights):
            for tensor1, tensor2 in zip(i, j):
                for value1, value2 in zip(tensor1, tensor2):
                    self.assertTrue(abs(value1-value2) < (0.01 * abs(value2)) + 0.001, f"Different weights value found :\nLOOP {value1} | FIT {value2}")

if __name__ == '__main__':
    """
    You need to add this line for the tests to be run.
    """
    unittest.main()