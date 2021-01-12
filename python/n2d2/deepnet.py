"""
    (C) Copyright 2020 CEA LIST. All Rights Reserved.
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
import n2d2.cell
import n2d2.converter
from n2d2.n2d2_interface import N2D2_Interface


class DeepNet(N2D2_Interface):

    def __init__(self, network, Model, DataType, **config_parameters):

        N2D2_Interface.__init__(self, **config_parameters)

        self._Model = Model
        self._DataType = DataType

        self._N2D2_object = N2D2.DeepNet(network)


    def __str__(self):
        output = "Deepnet" + "(" + self._Model + "<" + self._DataType + ">" + ")"
        output += N2D2_Interface.__str__(self)

    def get_model(self):
        return self._Model

    def get_datatype(self):
        return self._DataType


def load_from_ONNX(model_path, database, stimuliProvider):
    # TODO : add this code into a deepNet_converter function
    network = N2D2.Network(1)
    deepNet = N2D2.DeepNet(network)
    iniParser = N2D2.IniParser()
    deepNet.setDatabase(database)
    deepNet.setStimuliProvider(stimuliProvider)
    deepNet = N2D2.DeepNetGenerator.generateFromONNX(network, model_path, iniParser, deepNet)
    cells = deepNet.getCells()
    for cell in cells.values():
        # TODO : Need to work on cell converter
        n2d2.converter.cell_converter(cell)