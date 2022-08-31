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
import random
import math 
from n2d2.quantizer import SATCell

class test_SATQuant(unittest.TestCase):
    """
    The class needs to inherit unittest.TestCase, the name doesn't matter and the class doesn't need to be instantiated.
    """

    def setUp(self):
        """
        Method called before each test
        """
        n2d2.global_variables.set_cuda_device = 0


    def tearDown(self):
        """
        Method called after a test, even if it failed.
        Can be used to clean variables
        """
        pass

    def test(self):
        k_width = 2
        k_height = 2
        nb_channel = 1
        nb_outputs = 10
        nb_bits = 32

        weights = n2d2.Tensor([k_width,k_height, nb_channel, nb_outputs])
        weightsQ = n2d2.Tensor([k_width,k_height, nb_channel, nb_outputs])
        weightsDiff = n2d2.Tensor([k_width,k_height, nb_channel, nb_outputs])
        biases = n2d2.Tensor([nb_outputs])
        biasesQ = n2d2.Tensor([nb_outputs])
        biasesDiff = n2d2.Tensor([nb_outputs])

        max_W_abs = 0.0
        v_range = 2**nb_bits
        for i in range(weights.N2D2().size()):
            weights[i] = random.uniform(-1, 1)
            weightsQ[i] = math.tanh(weights[i])
            abs_w = abs(weightsQ[i])
            if(abs_w > max_W_abs):
                max_W_abs = abs_w

        for i in range(weightsQ.N2D2().size()):
            weightsQ[i] = 2 * ((1.0/v_range))*(0.5*((weightsQ[i]/max_W_abs)+1.0)*v_range)-1.0

        wSum = weightsQ.N2D2().sum()
        wMean = weightsQ.N2D2().mean()
        wVariance = 0.0

        for i in range(weightsQ.N2D2().size()):
            wVariance += ((weightsQ[i] - wMean)**2)

        wVariance = wVariance / (weightsQ.N2D2().size() - 1.0)
        wSATscaling = math.sqrt(wVariance*weightsQ.dimB()*weightsQ.dimY()*weightsQ.dimX())

        for i in range(weightsQ.N2D2().size()):
            weightsQ[i] = weightsQ[i] / wSATscaling

        for i in range(weightsDiff.N2D2().size()):
            weightsDiff[i] = weights[i]*((1/math.cosh(weights[i]))*(1/math.cosh(weights[i])) / (max_W_abs*wSATscaling))

        for i in range(biases.N2D2().size()):
            val = random.uniform(-1,1)
            biases[i] = val
            biasesQ[i] = val
            biasesDiff[i] = val

        weights.N2D2().synchronizeHToD()
        biases.N2D2().synchronizeHToD()

        quant = SATCell()

        quant.add_weights(weights, weights)
        quant.add_biases(biases, biases)

        quant.set_range(v_range)
        quant.set_scaling(True)
        quant.set_quantization(True)

        quant.N2D2().initialize()
        quant.N2D2().propagate()

        weights_estimated = quant.N2D2().getQuantizedWeights(0)
        weights_estimated.synchronizeDToH()

        for i in range(weightsQ.N2D2().size()):
            self.assertTrue(abs(weightsQ[i]-weights_estimated[i]) < 0.001 )

        biases_estimated = quant.N2D2().getQuantizedBiases()
        biases_estimated.synchronizeDToH()
        for i in range(biasesQ.N2D2().size()):
            self.assertTrue(abs(biasesQ[i]-biases_estimated[i]) < 0.001 )

        quant.N2D2().getQuantizedBiases().synchronizeDToH()
        quant.N2D2().back_propagate()

        weightsDiffEstimated = quant.N2D2().getDiffFullPrecisionWeights(0)
        weightsDiffEstimated.synchronizeDToH()
        for i in range(weightsDiff.N2D2().size()):
            self.assertTrue(abs((weightsDiff[i]-weightsDiffEstimated[i])) < 0.001)

        biasDiffEstimated = quant.N2D2().getDiffFullPrecisionBiases()
        biasDiffEstimated.synchronizeDToH()
        print(biasesDiff, biases_estimated, biasDiffEstimated)
        for i in range(biasesDiff.N2D2().size()):
            self.assertTrue(abs((biasesDiff[i]-biases_estimated[i])) < 0.001)

if __name__ == '__main__':
    """
    You need to add this line for the tests to be run.
    """
    unittest.main()
