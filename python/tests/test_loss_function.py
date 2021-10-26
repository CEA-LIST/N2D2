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

'''
Guide line for testing

The file test file should be placed in the test folder and should be named following the convention : "test_*.py".

The command to test the library is (you have to be in N2D2/python) :

python -m unittest discover -s test -v

The option discovery check for test_*.py files, so for example this file will note be caught !


If you need more information please check : https://docs.python.org/3/library/unittest.html
'''

import unittest
import n2d2
import math

class test_loss_function(unittest.TestCase):
    """
    The class needs to inherit unittest.TestCase, the name doesn't matter and the class doesn't need to be instantiated.
    """
    database = n2d2.database.MNIST(data_path="/nvme0/DATABASE/MNIST/raw", validation=0.1)

    provider = n2d2.provider.DataProvider(database, [32, 32, 1], batch_size=256)
    provider.add_transformation(n2d2.transform.Rescale(width=32, height=32))
    provider.set_partition("Learn")

    def setUp(self):
        val = -1.0
        class test_loss(n2d2.loss_function.LossFunction):
            def __init__(self) -> None:
                super().__init__()
            def compute_loss(self, inputs: n2d2.Tensor, target: n2d2.Tensor, **kwargs) -> n2d2.Tensor:
                return n2d2.Tensor(inputs.shape(), value=val)
        self.loss_function = test_loss()
        self.value = val       
        self.model = n2d2.models.lenet.LeNet(10)

    def tearDown(self):
        pass

    def test_diffInputs_update(self):
        x = self.provider.read_random_batch()
        model = self.model
        x = model(x)
        self.loss_function(x, self.provider.get_labels())
        diffInputs = n2d2.Tensor.from_N2D2(model[-1].N2D2().getDiffInputs())
        for i in diffInputs:
            self.assertEqual(i, self.value)

    def test_diffOutputs_update(self):
        x = self.provider.read_random_batch()
        model = self.model
        x = model(x)
        x = self.loss_function(x, self.provider.get_labels())
        old = [i for i in n2d2.Tensor.from_N2D2(model[-2].N2D2().getDiffOutputs())]
        x.back_propagate()
        new = [i for i in n2d2.Tensor.from_N2D2(model[-2].N2D2().getDiffOutputs())]
        flag = False
        for i,j in zip(old, new):
            if i != j :
                flag= True
                break
        self.assertTrue(flag)

    def test_weights_update(self):
        x = self.provider.read_random_batch()
        model = self.model
        x = model(x)
        x = self.loss_function(x, self.provider.get_labels())
        x.back_propagate()
        old_weights = [w for a in model[-2].get_weights()[0] for w in a]
        x.update()
        weights = [w for a in model[-2].get_weights()[0] for w in a]
        flag = False
        for i, j in zip(weights, old_weights):
            if i != j:
                flag=True
                break
        self.assertTrue(flag)

if __name__ == '__main__':
    """
    You need to add this line for the tests to be run.
    """
    unittest.main()