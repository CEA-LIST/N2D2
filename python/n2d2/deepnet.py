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
import collections


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


def load_from_ONNX(model_path, dims, batch_size=1):
    """
    :param model_path: Path to the model.
    :type model_path: str
    :param dims:
    :type dims: list
    :param batch_size:
    :type batch_size: unsigned int
    Load a deepnet from an ONNX file given its input dimensions.
    """
    network = N2D2.Network(n2d2.global_variables.default_seed)
    deepNet = N2D2.DeepNet(network)
    provider = n2d2.provider.DataProvider(n2d2.database.Database(), dims, batchSize=batch_size)
    deepNet.setDatabase(provider.get_database().N2D2())
    deepNet.setStimuliProvider(provider.N2D2())
    N2D2.CellGenerator.defaultModel = n2d2.global_variables.default_deepNet.get_model()
    deepNet = N2D2.DeepNetGenerator.generateFromONNX(network, model_path, N2D2.IniParser(), deepNet)
    model = n2d2.converter.deepNet_converter(deepNet)
    model.clear_input(unlink_N2D2_input=True)     # Remove dummy stimuli provider
    return model

def load_from_INI(path):
    """
    :param model_path: Path to the ini file.
    :type model_path: str
    Load a deepnet from an INI file.
    """
    network = N2D2.Network(1)
    deepNet = N2D2.DeepNetGenerator.generateFromINI(network, path)
    return n2d2.converter.deepNet_converter(deepNet)
"""
We should be able to create cells and sequences of cell incrementally
We should be able to extract cell and sequences and run these subnetworks easily
"""


"""
Structure that is organised sequentially. 
"""
class Sequence:
    def __init__(self, sequences, inputs=None, name=""):
        assert isinstance(name, str)
        self._name = name
        assert isinstance(sequences, list)
        if not sequences:
            raise ValueError("Got empty list as input. List must contain at least one element")
        self._sequences = sequences


        previous = None
        for elem in self._sequences:
            if previous is not None:
                #elem.clear_input()
                elem.add_input(previous)
            previous = elem

        if inputs is not None:
            if isinstance(inputs, list):
                for cell in inputs:
                    self.add_input(cell)
            else:
                self.add_input(inputs)

    def get_cells(self):
        cells = {}
        self._get_cells(cells)
        return cells

    def _get_cells(self, cells):
        for elem in self._sequences:
            if isinstance(elem, Sequence) or isinstance(elem, Layer):
                elem._get_cells(cells)
            else:
                cells[elem.get_name()] = elem

    # TODO: Method that converts sequential representation into corresponding N2D2 deepnet

    def add_input(self, inputs, link_N2D2_input=False):
        self.get_first().add_input(inputs, link_N2D2_input)

    def get_inputs(self):
        return self.get_first().get_inputs()

    def clear_input(self, unlink_N2D2_input=False):
        self.get_first().clear_input(unlink_N2D2_input)

    def initialize(self):
        for cell in self._sequences:
            cell.initialize()

    def propagate(self, inference=False):
        for cell in self._sequences:
            cell.propagate(inference)

    def back_propagate(self):
        for cell in reversed(self._sequences):
            cell.back_propagate()

    def update(self):
        for cell in self._sequences:
            cell.update()

    def import_free_parameters(self, dirName, ignoreNotExists=False):
        print("Importing weights from directory '" + dirName + "'")
        for name, cell in self.get_cells().items():
            path = dirName + name + ".syntxt"
            cell.import_free_parameters(path, ignoreNotExists=ignoreNotExists)

    def get_subsequence(self, id):
        if isinstance(id, int):
            return self._sequences[id]
        else:
            for elem in self._sequences:
                if elem.get_name() == id:
                    return elem
            raise RuntimeError("No subsequence with name: \'" + id + "\'")

    def get_name(self):
        return self._name

    def set_name(self, name):
        self._name = name

    def get_sequences(self):
        return self._sequences

    def get_last(self):
        return self._sequences[-1].get_last()

    def get_first(self):
        return self._sequences[0].get_first()
    """
    def get_cells_sequence(self):
        return self._cells_sequence
    """

    """
    def convert_to_INI_section(self):
        output = ""
        for cell in self._cells_sequence:
            output += cell.convert_to_INI_section()
            output += "\n"
        return output
    """

    def __str__(self):
        return self._generate_str(1)

    def _generate_str(self, indent_level):
        if not self.get_name() == "":
            output = "\'" + self.get_name() + "\' " + "Sequence("
        else:
            output = "Sequence("

        for idx, value in enumerate(self._sequences):
            output += "\n" + (indent_level * "\t") + "(" + str(idx) + ")"
            if isinstance(value, n2d2.deepnet.Sequence):
                output += ": " + value._generate_str(indent_level + 1)
            elif isinstance(value, n2d2.deepnet.Layer):
                output += ": " + value._generate_str(indent_level + 1)
            else:
                output += ": " + value.__str__()
        output += "\n" + ((indent_level-1) * "\t") + ")"
        return output




class Layer:
    def __init__(self, layer, Inputs=None, name=""):
        assert isinstance(name, str)
        self._name = name
        assert isinstance(layer, list)
        if not layer:
            raise ValueError("Got empty list as input. List must contain at least one element")
        self._layer = layer

        if Inputs is not None:
            if isinstance(Inputs, list):
                for cell in Inputs:
                    self.add_input(cell)
            else:
                self.add_input(Inputs)

    def get_cells(self):
        cells = {}
        self._get_cells(cells)
        return cells

    def _get_cells(self, cells):
        for elem in self._layer:
            if isinstance(elem, Sequence):
                elem._get_cells(cells)
            else:
                cells[elem.get_name()] = elem

    # TODO: Method that converts layer representation into corresponding N2D2 deepnet

    def add_input(self, input, link_N2D2_input=False):
        for cell in self._layer:
            cell.add_input(input, link_N2D2_input)

    def clear_input(self, unlink_N2D2_input=False):
        for cell in self._layer:
            cell.clear_input(unlink_N2D2_input)

    def get_last(self):
        return self

    def get_first(self):
        return self

    def get_elements(self):
        return self._layer

    def initialize(self):
        for cell in self._layer:
            cell.initialize()

    def propagate(self, inference=False):
        for cell in self._layer:
            cell.propagate(inference)

    def back_propagate(self):
        for cell in reversed(self._layer):
            cell.back_propagate()

    def update(self):
        for cell in self._layer:
            cell.update()

    def get_name(self):
        return self._name

    def __str__(self):
        return self._generate_str(0)

    """
    def convert_to_INI_section(self):
        output = ""
        for cell in self._layer:
            output += cell.convert_to_INI_section()
            output += "\n"
        return output
    """

    def _generate_str(self, indent_level):
        if not self.get_name() == "":
            output = "\'" + self.get_name() + "\' " + "Layer(\n"
        else:
            output = "Layer(\n"
        for idx, elem in enumerate(self._layer):
            if isinstance(elem, n2d2.cell.Cell):
                output += (indent_level * "\t") + "[" + str(idx) + "]: " + elem.__str__() + "\n"
            else:
                output += (indent_level * "\t") + "[" + str(idx) + "]: " + elem._generate_str(indent_level+1) + "\n"
        output += ((indent_level-1) * "\t") + ")"
        return output
        