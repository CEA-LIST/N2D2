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


def load_from_ONNX(model_path, provider):
    """
    :param model_path: Path to the model.
    :type model_path: str
    :param provider: 
    :type provider: :py:class:`n2d2.provider.DataProvider`

    Load a deepnet from an ONNX file and a database.
    """
    network = N2D2.Network(1)
    deepNet = N2D2.DeepNet(network)
    iniParser = N2D2.IniParser()
    deepNet.setDatabase(provider.get_database().N2D2())
    deepNet.setStimuliProvider(provider.N2D2())
    deepNet = N2D2.DeepNetGenerator.generateFromONNX(network, model_path, iniParser, deepNet)
    return n2d2.converter.deepNet_converter(deepNet)

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
    def __init__(self, sequences, Name=None):
        if Name is not None:
            assert isinstance(Name, str)
        self._Name = Name
        assert isinstance(sequences, list)
        if not sequences:
            raise ValueError("Got empty list as input. List must contain at least one element")
        #self._sequence_dict = collections.OrderedDict()
        self._sequences = sequences
        # TODO: This is currently not really used
        #self._cells_sequence = []

        #self._generate_graph(self, '')

        #print(self._sequence_dict)

        """
        previous = None
        for key, value in self._sequence_dict.items():
            if isinstance(value, n2d2.cell.Cell) or isinstance(value, n2d2.deepnet.Layer):
                if previous is not None:
                    value.clear_input()
                    value.add_input(previous)
                previous = value
        """
        previous = None
        for elem in self._sequences:
            if previous is not None:
                elem.clear_input()
                elem.add_input(previous)
            previous = elem


    """Goes recursively through sequences"""

    def _generate_graph(self, sequence, sequence_idx):

        if not sequence_idx == '':
            self._sequence_dict[sequence_idx] = sequence
        #sequence.set_sequence_idx(sequence_idx)

        if isinstance(sequence, Sequence):
            if not sequence_idx == '':
                sequence_idx += '.'
            for idx, sub_sequence in enumerate(sequence.get_sequences()):
                self._generate_graph(sub_sequence, sequence_idx + str(idx))
        else:
            self._cells_sequence.append(sequence)

    # TODO: Method that converts sequential representation into corresponding N2D2 deepnet

    def add_input(self, inputs):
        self.get_first().add_input(inputs)

    def get_inputs(self):
        return self.get_first().get_intputs()

    def clear_input(self):
        self.get_first().clear_input()

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

    def get_subsequence(self, idx):
        return self._sequences[idx]

    def get_name(self):
        return self._Name

    def set_name(self, Name):
        self._Name = Name

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
    def __init__(self, layer, Name=None):
        if Name is not None:
            assert isinstance(Name, str)
        self._Name = Name
        assert isinstance(layer, list)
        if not layer:
            raise ValueError("Got empty list as input. List must contain at least one element")
        self._layer_dict = collections.OrderedDict()
        self._layer = layer

        for idx, elem in enumerate(self._layer):
            self._layer_dict[str(idx)] = elem

    # TODO: Method that converts layer representation into corresponding N2D2 deepnet

    def add_input(self, input):
        for cell in self._layer:
            cell.add_input(input)

    def clear_input(self):
        for cell in self._layer:
            cell.clear_input()

    def get_last(self):
        output = []
        for cell in self._layer:
            output.append(cell)
        return output

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
        output = "Layer(\n"
        for key, value in self._layer_dict.items():
            output += (indent_level * "\t") + "[" + key + "]: " + value.__str__() + "\n"
        return output
        