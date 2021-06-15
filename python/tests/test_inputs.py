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
from n2d2 import activation

"""
TODO
Here we test if the errors for main parameters are well set.

Optional parameters (N2D2) type are checked using N2D2 in n2d2_interface._set_N2D2_parameter()
Do we need to test if an error is send for each object and each parameters ?
The test will be very redondant and provide very little  
"""

class test_cells(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_Fc(self):
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.cells.Fc("1", 1)
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.cells.Fc(1, "1")
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.cells.Fc(1, 1, activation=1)
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.cells.Fc(1, 1, weights_solver=1)
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.cells.Fc(1, 1, bias_solver=1)
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.cells.Fc(1, 1, weights_filler=1)
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.cells.Fc(1, 1, bias_filler=1)
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.cells.Fc(1, 1, quantizer=1)

    def test_Conv(self):
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.cells.Conv("1", 1, [1, 1])
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.cells.Conv(1, "1", [1, 1])
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.cells.Conv(1, 1, 1)
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.cells.Conv(1, 1, [1, 1], activation=1)
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.cells.Conv(1, 1, [1, 1], weights_solver=1)
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.cells.Conv(1, 1, [1, 1], bias_solver=1)
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.cells.Conv(1, 1, [1, 1], weights_filler=1)
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.cells.Conv(1, 1, [1, 1], bias_filler=1)
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.cells.Conv(1, 1, [1, 1], quantizer=1)


    def test_Softmax(self):
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.cells.Softmax(activation=1)
    
    def test_Pool(self):
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.cells.Pool(1)
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.cells.Pool([1, 1], activation=1)

    def test_Pool(self):
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.cells.Pool(1)
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.cells.Pool([1, 1], activation=1)        
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.cells.Pool([1, 1], pooling=1)  


    def test_Deconv(self):
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.cells.Deconv("1", 1, [1, 1])
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.cells.Deconv(1, "1", [1, 1])
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.cells.Deconv(1, 1, 1)
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.cells.Deconv(1, 1, [1, 1], activation=1)
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.cells.Deconv(1, 1, [1, 1], weights_solver=1)
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.cells.Deconv(1, 1, [1, 1], bias_solver=1)
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.cells.Deconv(1, 1, [1, 1], weights_filler=1)
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.cells.Deconv(1, 1, [1, 1], bias_filler=1)
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.cells.Deconv(1, 1, [1, 1], quantizer=1)

    def test_ElemWise(self):
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.cells.ElemWise(operation=1)
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.cells.ElemWise(mode=1)
        with self.assertRaises(n2d2.error_handler.WrongValue):
            n2d2.cells.ElemWise(mode='a')
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.cells.ElemWise(weights="aa") 
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.cells.ElemWise(shifts="aa") 

    def test_Dropout(self):
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.cells.Dropout(dropout="a")
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.cells.Dropout(activation=1)
          
    def test_Padding(self):
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.cells.Padding("1",1,1,1)
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.cells.Padding(1,"1",1,1)
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.cells.Padding(1,1,"1",1)
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.cells.Padding(1,1,1,"1")
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.cells.Padding(1,1,1,1, activation=1)

    def test_Reshape(self):
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.cells.Reshape(1)
        
    def test_BatchNorm2d(self):
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.cells.BatchNorm2d(1, scale_solver="a")
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.cells.BatchNorm2d(1, bias_solver="a")
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.cells.BatchNorm2d(1, moving_average_momentum=1)
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.cells.BatchNorm2d(1, epsilon=1)
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.cells.BatchNorm2d(1, activation=1)

class test_activation(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_Linear(self):
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.activation.Linear(quantizer=1)
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.activation.Linear(clipping="a")
            
    def test_Rectifier(self):
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.activation.Rectifier(quantizer=1)
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.activation.Rectifier(leak_slope="a")
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.activation.Rectifier(clipping="a")

class test_filler(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_He(self):
        with self.assertRaises(n2d2.error_handler.WrongValue):
            n2d2.filler.He(variance_norm=1)
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.filler.He(mean_norm="a")
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.filler.He(scaling="a")

    def test_Normal(self):
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.filler.Normal(mean="a")
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.filler.Normal(std_dev="a")      

    def test_Xavier(self):
        with self.assertRaises(n2d2.error_handler.WrongValue):
            n2d2.filler.Xavier(distribution=1)
        with self.assertRaises(n2d2.error_handler.WrongValue):
            n2d2.filler.Xavier(distribution=1)
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.filler.Xavier(scaling="a")  
    
    def test_Constant(self):
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.filler.Constant(value="a")  

class test_solver(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_SGD(self):
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.solver.SGD(learning_rate='a')
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.solver.SGD(momentum="a")
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.solver.SGD(decay="a")
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.solver.SGD(min_decay="a")
        with self.assertRaises(n2d2.error_handler.WrongValue):
            n2d2.solver.SGD(learning_rate_policy=1)
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.solver.SGD(learning_rate_decay='a')
        with self.assertRaises(n2d2.error_handler.WrongValue):
            n2d2.solver.SGD(clamping=1)

    def test_Adam(self):
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.solver.Adam(learning_rate='a')
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.solver.Adam(beta1="a")
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.solver.Adam(beta2="a")
        with self.assertRaises(n2d2.error_handler.WrongInputType):
            n2d2.solver.Adam(epsilon='a')
        with self.assertRaises(n2d2.error_handler.WrongValue):
            n2d2.solver.Adam(clamping=1)


if __name__ == '__main__':
    """
    You need to add this line for the tests to be run.
    """
    unittest.main()