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
import N2D2
from math import tanh

class test_Linear(unittest.TestCase):
    def setUp(self):
        self.activation = n2d2.activation.Linear()
        
    def tearDown(self):
        pass

    def test_output(self):
        
        ActivationCell = n2d2.cell.Activation(activationFunction=self.activation)
        input_tensor = n2d2.Tensor([1, 1, 2, 2], value=1.0, cuda=True) 
        input_tensor[0] = 0
        input_tensor[1] = 1
        input_tensor[2] = 2
        input_tensor[3] = 3
        inputs = n2d2.GraphTensor(input_tensor)
        outputs = ActivationCell(inputs)
        self.assertEqual(outputs[0], 0)
        self.assertEqual(outputs[1], 1)
        self.assertEqual(outputs[2], 2)
        self.assertEqual(outputs[3], 3)

class test_Tanh(unittest.TestCase):
    def setUp(self):
        self.activation = n2d2.activation.Tanh()
        
    def tearDown(self):
        pass

    def test_output(self):
        
        ActivationCell = n2d2.cell.Activation(activationFunction=self.activation)
        input_tensor = n2d2.Tensor([1, 1, 2, 2], value=1.0, cuda=True) 
        input_tensor[0] = 0
        input_tensor[1] = 1
        input_tensor[2] = 2
        input_tensor[3] = 3
        inputs = n2d2.GraphTensor(input_tensor)
        outputs = ActivationCell(inputs)
        self.assertEqual(round(outputs[0], 5), round(tanh(0), 5))
        self.assertEqual(round(outputs[1], 5), round(tanh(1), 5))
        self.assertEqual(round(outputs[2], 5), round(tanh(2), 5))
        self.assertEqual(round(outputs[3], 5), round(tanh(3), 5))

class test_Rectifier(unittest.TestCase):
    def setUp(self):
        self.activation = n2d2.activation.Rectifier()
        
    def tearDown(self):
        pass

    def test_output(self):
        
        ActivationCell = n2d2.cell.Activation(activationFunction=self.activation)
        input_tensor = n2d2.Tensor([1, 1, 2, 2], value=1.0, cuda=True) 
        input_tensor[0] = -1
        input_tensor[1] = -2
        input_tensor[2] = 2
        input_tensor[3] = 3
        inputs = n2d2.GraphTensor(input_tensor)
        outputs = ActivationCell(inputs)
        self.assertEqual(outputs[0], 0)
        self.assertEqual(outputs[1], 0)
        self.assertEqual(outputs[2], 2)
        self.assertEqual(outputs[3], 3)
if __name__ == '__main__':
    unittest.main()