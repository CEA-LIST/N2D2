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
"""
Test if instantiating an abstract class is possible.
"""

class test_abstract_init(unittest.TestCase):

    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_cell(self):
        with self.assertRaises(TypeError):
            n2d2.cells.Cell()
    def test_neural_network_cell(self):
        with self.assertRaises(TypeError):
            n2d2.cells.NeuralNetworkCell()
    def test_target(self):
        with self.assertRaises(TypeError):
            n2d2.target.Target()
    def test_solver(self):
        with self.assertRaises(TypeError):
            n2d2.solver.Solver()
    def test_filler(self):
        with self.assertRaises(TypeError):
            n2d2.filler.Filler()
    def test_transform(self):
        with self.assertRaises(TypeError):
            n2d2.transform.Transformation()
    def test_provider(self):
        with self.assertRaises(TypeError):
            n2d2.provider.Provider()
if __name__ == '__main__':
    unittest.main()