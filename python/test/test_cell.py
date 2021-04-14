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

class test_ConvCell(unittest.TestCase):
    """
    The class needs to inherite unittest.TestCase, the name doesn't matter and the class doesn't need to be instanciated.
    """

    def setUp(self):
        self.cell = n2d2.cells.cell.Conv(1, 10, kernel_dims=[2, 2])

    def tearDown(self):
        pass

    def test_get_set_weight(self):
        weight = n2d2.Tensor([2, 2])
        weight[0] = 1
        weight[1] = 2
        self.cell.set_weight(0, 0, weight)
        self.assertTrue(weight == self.cell.get_weight(0, 0))
        
    def test_get_weights(self):
        print(self.cell.get_weights())

    def test_getOutput(self):
        self.cell.get_outputs()

class test_FcCell(unittest.TestCase):
    """
    The class needs to inherit unittest.TestCase, the name doesn't matter and the class doesn't need to be instanciated.
    """

    def setUp(self):
        self.cell = n2d2.cells.cell.Fc(10, 5)
    def tearDown(self):
        pass

    def test_get_set_weight(self):
        weight = n2d2.Tensor([1])
        weight[0] = 1
        print(weight)
        self.cell.set_weight(0, 0, weight)
        tensor = N2D2.Tensor_float([])
        self.assertTrue(weight == self.cell.get_weight(0, 0))
    
    def test_get_weights(self):
        print(self.cell.get_weights())
    
    def test_getOutput(self):
        self.cell.get_outputs()

if __name__ == '__main__':
    unittest.main()