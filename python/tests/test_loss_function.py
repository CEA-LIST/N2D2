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

class test_loss_function(unittest.TestCase):
    """
    The class needs to inherit unittest.TestCase, the name doesn't matter and the class doesn't need to be instantiated.
    """

    def setUp(self):
        val = -1.0
        class test_loss(n2d2.loss_function.LossFunction):
            def __init__(self) -> None:
                super().__init__()

            def compute_loss(self, inputs: n2d2.Tensor, target: n2d2.Tensor, **kwargs) -> n2d2.Tensor:
                return n2d2.Tensor(inputs.shape(), value=val)
        self.input = n2d2.Tensor([1, 1, 2, 2], cuda=n2d2.cuda_compiled)
        self.fc = n2d2.cells.Fc(1, 5, weights_solver = n2d2.solver.SGD(learning_rate=0.05, momentum=0.9, decay=0.0005, learning_rate_decay=0.993))

        shape = self.input.dims()
        # Note : It's important to set diffOutputs as an attribute else when exiting this method
        # Python garbage collector will erase this variable while Cpp will still use it resulting in a SegFault
        self.diffOutputs = n2d2.Tensor(shape, value=0, dim_format="N2D2")

        self.fc.N2D2().clearInputs() 
        self.fc.N2D2().addInputBis(self.input.N2D2(), self.diffOutputs.N2D2())
        
        self.loss_function = test_loss()
        self.value = val


    def tearDown(self):
        pass

    def test_diffInputs_update(self):
        x = self.input
        fc = self.fc

        x = fc(x)

        self.loss_function(x, n2d2.Tensor([1, 5, 1, 1]))

        diffInputs = n2d2.Tensor.from_N2D2(fc.N2D2().getDiffInputs())
        for i in diffInputs:
            self.assertEqual(i, self.value)

    def test_diffOutputs_update(self):
        x = self.input
        fc = self.fc
        
        x = fc(x)
        fc.N2D2().clearInputs() 
        fc.N2D2().addInputBis(self.input.N2D2(), self.diffOutputs.N2D2())

        x = self.loss_function(x, n2d2.Tensor([1, 5, 1, 1]))
        old = [i for i in n2d2.Tensor.from_N2D2(fc.N2D2().getDiffOutputs())]
        x.back_propagate()
        new = [i for i in n2d2.Tensor.from_N2D2(fc.N2D2().getDiffOutputs())]
        flag = False
        for i,j in zip(old, new):
            if i != j :
                flag= True
                break
        self.assertTrue(flag)

    def test_weights_update(self):
        x = self.input
        fc = self.fc
        
        x = fc(x)
        fc.N2D2().clearInputs() 
        fc.N2D2().addInputBis(self.input.N2D2(), self.diffOutputs.N2D2())
        x = self.loss_function(x, n2d2.Tensor([1, 5, 1, 1]))
        x.back_propagate()
        old_weights = [w for a in fc.get_weights()[0] for w in a]
        x.update()
        weights = [w for a in fc.get_weights()[0] for w in a]
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