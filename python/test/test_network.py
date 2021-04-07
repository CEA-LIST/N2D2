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
import N2D2

@unittest.skip("Waiting for deepnet refactor !")
class test_modibleNet(unittest.TestCase):
    def setUp(self):
        n2d2.global_variables.set_cuda_device(4)
        net = N2D2.Network(1)
        self.N2D2_deepNet = N2D2.DeepNetGenerator.generate(net, "../models/MobileNetv1.ini")
        self.N2D2_deepNet.initialize()

    def test_Output(self):
        
        db = n2d2.database.ILSVRC2012(0.01)
        provider = n2d2.provider.DataProvider(db, [160, 160, 3], batchSize=256)
        n2d2_deepNet = n2d2.model.MobileNetv1(provider)._deepnet

        n2d2_first_cell = n2d2_deepNet.getCells()[n2d2_deepNet.getLayers()[1][0]] # The first layer is the env, so we get the second.
        n2d2_last_cell  = n2d2_deepNet.getCells()[n2d2_deepNet.getLayers()[-1][-1]]
        shape = [256, 3, 160, 160]
        input_tensor = tensor.Tensor(shape, value=1)
        print("Input0", input_tensor)
        diffOutputs = tensor.Tensor(shape, value=0)

        n2d2_first_cell.clearInputs()
        n2d2_first_cell.addInputBis(input_tensor.N2D2(), diffOutputs.N2D2())
        print("Input1", input_tensor)
        n2d2_output = n2d2_last_cell.getOutputs() 


        N2D2_first_cell = N2D2_deepNet.getCells()[N2D2_deepNet.getLayers()[1][0]] # The first layer is the env, so we get the second.
        N2D2_last_cell  = N2D2_deepNet.getCells()[N2D2_deepNet.getLayers()[-1][-1]]
        
        N2D2_first_cell.clearInputs()
        N2D2_first_cell.addInputBis(input_tensor.N2D2(), diffOutputs.N2D2())
        N2D2_output = N2D2_last_cell.getOutputs() 
        print("Input2", input_tensor)

        print("Output n2d2:", n2d2_output)
        print("Output N2D2:", N2D2_output)


if __name__ == '__main__':
    unittest.main()
    