"""
    (C) Copyright 2022 CEA LIST. All Rights Reserved.
    Contributor(s): Cyril MOINEAU (cyril.moineau@cea.fr) 

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
import keras_interoperability

from tensorflow.random import uniform
from os.path import exists
from os import remove

from tiny_ml_keras.anomaly_model import get_model as get_anomaly_model
from tiny_ml_keras.kws_model import kws_dscnn as get_kws_model
from tiny_ml_keras.resnet_model import resnet_v1_eembc as get_resnet_model
from tiny_ml_keras.vww_model import mobilenet_v1 as get_mobilenet_model


DATA_PATH="/local/DATABASE/"


class test_keras_export(unittest.TestCase):
    """
    The class needs to inherit unittest.TestCase, the name doesn't matter and the class doesn't need to be instantiated.
    """
    absolute_precision = 0.0001
    relative_precision = 0.001

    def setUp(self):
        n2d2.global_variables.cuda_device = 0
        n2d2.global_variables.default_model = 'Frame_CUDA'

    def tearDown(self):
        n2d2.global_variables.cuda_device = 0
        n2d2.global_variables.default_model = 'Frame'

    def check_tensor_equality(self, x, y):
        for i,j in zip(x.numpy().flatten(), y.numpy().flatten()):
            self.assertTrue(abs(i-j) < self.relative_precision * abs(j) + self.absolute_precision,
                "N2D2 and Keras give different output tensor !")

    def test_anomaly_CPP(self):

        net_test=get_anomaly_model(640)
        n2d2_net_test = keras_interoperability.wrap(net_test, batch_size=5, for_export=True)
        input_test= uniform([5, 640])
        keras_out = net_test(input_test)
        n2d2_out = n2d2_net_test(input_test)
        print("Keras output :")
        print(keras_out)
        print("N2D2 output :")
        print(n2d2_out)
        self.check_tensor_equality(keras_out, n2d2_out)

        print("Model have been wrapped !")
        # Importing data for calibration.
        db = n2d2.database.DIR(DATA_PATH+"tif_database_ToyCar",
                            learn=0.8, validation=0.2, random_partitioning=True)

        provider = n2d2.provider.DataProvider(db,[640, 1, 1], batch_size=5)

        # Generating CPP export
        n2d2.export.export_cpp(
            n2d2_net_test.get_deepnet_cell(),
            provider=provider,
            nb_bits=8,
            calibration=1,
            export_nb_stimuli_max=0)
        export_generated = exists("./export_CPP_int8")
        self.assertTrue(export_generated)
        if not export_generated:
            remove("./export_CPP_int8")

    def test_kws_CPP(self):

        net_test=get_kws_model(49,10,12, for_tflite=True, BNorm=True)
        n2d2_net_test = keras_interoperability.wrap(net_test, batch_size=5, for_export=True)
        input_test= uniform([5, 49, 10, 1])
        keras_out = net_test(input_test)
        n2d2_out = n2d2_net_test(input_test)
        print("Keras output :")
        print(keras_out)
        print("N2D2 output :")
        print(n2d2_out)
        self.check_tensor_equality(keras_out, n2d2_out)

        print("Model have been wrapped !")
        # Importing data for calibration.
        db = n2d2.database.DIR(DATA_PATH+"speech_commands_v0.02_mfcc_10words",
                        learn=0.8, validation=0.2, depth=1, 
                        ignore_mask=["*/_background_noise_"], valid_extensions=["tiff"])

        provider = n2d2.provider.DataProvider(db, [10, 49, 1], batch_size=5)
        deepnet_cell = n2d2_net_test.get_deepnet_cell()
        # remove SoftMax
        deepnet_cell.remove(deepnet_cell[-1].get_name())
        # Generating CPP export
        n2d2.export.export_cpp(
            deepnet_cell,
            provider=provider,
            nb_bits=8,
            calibration=1,
            export_nb_stimuli_max=0)
        export_generated = exists("./export_CPP_int8")

        self.assertTrue(export_generated)
        if not export_generated:
            remove("./export_CPP_int8")

    def test_resnet_CPP(self):

        net_test=get_resnet_model()
        n2d2_net_test = keras_interoperability.wrap(net_test, batch_size=5, for_export=True)
        input_test= uniform([5, 32, 32, 3])
        keras_out = net_test(input_test)
        n2d2_out = n2d2_net_test(input_test)
        print("Keras output :")
        print(keras_out)
        print("N2D2 output :")
        print(n2d2_out)
        self.check_tensor_equality(keras_out, n2d2_out)

        print("Model have been wrapped !")
        # Importing data for calibration.
        db = n2d2.database.DIR(DATA_PATH+"CIFAR-10-images/test",
            learn=0.4,
            validation=0.2,
            random_partitioning=True,
            depth=1,
            valid_extensions=["jpg"])

        provider = n2d2.provider.DataProvider(db, [32, 32, 3], batch_size=5)
        deepnet_cell = n2d2_net_test.get_deepnet_cell()
        # remove SoftMax
        deepnet_cell.remove(deepnet_cell[-1].get_name())
        # Generating CPP export
        n2d2.export.export_cpp(
            deepnet_cell,
            provider=provider,
            nb_bits=8,
            calibration=1,
            export_no_unsigned=True,
            export_nb_stimuli_max=0
            )
        export_generated = exists("./export_CPP_int8")

        self.assertTrue(export_generated)
        if not export_generated:
            remove("./export_CPP_int8")

    def test_mobilenet_CPP(self):

        net_test=get_mobilenet_model()
        n2d2_net_test = keras_interoperability.wrap(net_test, batch_size=5, for_export=True)
        input_test= uniform([5, 96, 96, 3])
        keras_out = net_test(input_test)
        n2d2_out = n2d2_net_test(input_test)
        print("Keras output :")
        print(keras_out)
        print("N2D2 output :")
        print(n2d2_out)
        self.check_tensor_equality(keras_out, n2d2_out)

        print("Model have been wrapped !")
        # Importing data for calibration.
        db = n2d2.database.DIR(DATA_PATH+"vw_coco2014_96",
            learn=0.8, validation=0.2, random_partitioning=True)

        provider = n2d2.provider.DataProvider(db, [96, 96, 3], batch_size=5)

        deepnet_cell = n2d2_net_test.get_deepnet_cell()

        # remove SoftMax
        deepnet_cell.remove(deepnet_cell[-1].get_name())

        # Generating CPP export
        n2d2.export.export_cpp(
            deepnet_cell,
            provider=provider,
            nb_bits=8,
            calibration=1,
            export_nb_stimuli_max=0)
        export_generated = exists("./export_CPP_int8")

        self.assertTrue(export_generated)
        if not export_generated:
            remove("./export_CPP_int8")

if __name__ == '__main__':
    """
    You need to add this line for the tests to be run.
    """
    unittest.main()