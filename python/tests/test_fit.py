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
Run a fit and a test on a simple layer.
This test is not truly unitary as the feature mostly come from the CPP side.
'''

import unittest
import n2d2

class test_fit_test(unittest.TestCase):


    def test(self):
        # Change default model
        n2d2.global_variables.default_model = "Frame_CUDA"
        # Change cuda device (default 0)
        n2d2.global_variables.set_cuda_device = 2


        batch_size = 256

        print("\n### Create database ###")
        database = n2d2.database.MNIST(data_path="/nvme0/DATABASE/MNIST/raw/", validation=0.1)
        print(database)

        print("\n### Create Provider ###")
        provider = n2d2.provider.DataProvider(database, [32, 32, 1], batch_size=batch_size)
        provider.add_transformation(n2d2.transform.Rescale(width=32, height=32))
        print(provider)

        print("\n### Loading Model ###")
        model = n2d2.models.lenet.LeNet(10)
        print(model)

        deepnet_cell = model.to_deepnet_cell(provider)

        deepnet_cell.fit(1)

if __name__ == '__main__':
    """
    You need to add this line for the tests to be run.
    """
    unittest.main()