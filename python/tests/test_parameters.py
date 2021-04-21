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
import N2D2
import n2d2
import unittest
from n2d2.n2d2_interface import N2D2_Interface as N2D2_Interface




class test_params(unittest.TestCase):
    def setUp(self):
        self.parameters = {
            
        }
        self.object = None

    def test_parameters(self):
        # /!\ You want to always put this at the end of the test /!\

        if self.object: # We don't do test if it's the dummy class
            parameters = self.object.N2D2().getParameters()
            for param in self.parameters.keys():
                if N2D2_Interface.python_to_n2d2_convention(param) in parameters:
                    param_name = N2D2_Interface.python_to_n2d2_convention(param)
                    N2D2_param, N2D2_type = self.object.N2D2().getParameterAndType(param_name)
                    N2D2_param = N2D2_Interface._N2D2_type_map[N2D2_type](N2D2_param)
                    self.assertEqual(self.parameters[param], N2D2_param)
            #         self.parameters.pop(param)
            
            # # We check if we have tested every parameters !
            # self.assertTrue(self.parameters.keys() == [])


### TEST CELLS ###


class test_Fc(test_params):
    def setUp(self):
        self.parameters = {
            "name": "test",
            "activation_function": n2d2.activation.Tanh(),
            "weights_solver": n2d2.solver.SGD(),
            "bias_solver": n2d2.solver.SGD(),
            "weights_filler": n2d2.filler.Normal(),
            "bias_filler": n2d2.filler.Normal(),
            'no_bias': True,
        }
        self.object = n2d2.cells.Fc(10, 5, **self.parameters)

    def test_parameters(self):
        self.assertEqual(self.parameters["name"], self.object.N2D2().getName())
        self.assertIs(self.parameters["activation_function"].N2D2(), self.object.N2D2().getActivation())
        self.assertIs(self.parameters["weights_solver"].N2D2(), self.object.N2D2().getWeightsSolver())
        self.assertIs(self.parameters["bias_solver"].N2D2(), self.object.N2D2().getBiasSolver())
        self.assertIs(self.parameters["weights_filler"].N2D2(), self.object.N2D2().getWeightsFiller())
        self.assertIs(self.parameters["bias_filler"].N2D2(), self.object.N2D2().getBiasFiller())
        super().test_parameters()


class test_Conv(test_params):
    def setUp(self):
        self.parameters = {
            "name": "test",
            "activation_function": n2d2.activation.Tanh(),
            "weights_solver": n2d2.solver.SGD(),
            "sub_sample_dims": [2, 2],
            "stride_dims": [2, 2],
            "dilation_dims": [1, 1],
            "padding_dims": [2, 2],
            "bias_solver": n2d2.solver.SGD(),
            "weights_filler": n2d2.filler.Normal(),
            "bias_filler": n2d2.filler.Normal(),
            "no_bias": True,
            # "quantizer": n2d2.quantizer.CellQuantizer(), # TODO 
            "back_propagate": True,
        }
        self.object = n2d2.cells.Conv(10, 5, [2, 2], **self.parameters)

    def test_parameters(self):
        self.assertEqual(self.parameters["name"], self.object.N2D2().getName())
        self.assertIs(self.parameters["activation_function"].N2D2(), self.object.N2D2().getActivation())
        self.assertIs(self.parameters["weights_solver"].N2D2(), self.object.N2D2().getWeightsSolver())
        self.assertIs(self.parameters["bias_solver"].N2D2(), self.object.N2D2().getBiasSolver())
        self.assertIs(self.parameters["weights_filler"].N2D2(), self.object.N2D2().getWeightsFiller())
        self.assertIs(self.parameters["bias_filler"].N2D2(), self.object.N2D2().getBiasFiller())
        self.assertEqual(self.parameters["sub_sample_dims"], [self.object.N2D2().getSubSampleX(), self.object.N2D2().getSubSampleY()])
        self.assertEqual(self.parameters["stride_dims"], [self.object.N2D2().getStrideX(), self.object.N2D2().getStrideY()])
        self.assertEqual(self.parameters["padding_dims"], [self.object.N2D2().getPaddingX(), self.object.N2D2().getPaddingY()])
        self.assertEqual(self.parameters["dilation_dims"], [self.object.N2D2().getDilationX(), self.object.N2D2().getDilationY()])
  
        super().test_parameters()

if __name__ == '__main__':
    unittest.main()