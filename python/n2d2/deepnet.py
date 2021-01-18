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


def load_from_ONNX(model_path, database, stimuliProvider):
    # TODO : need to change the enrty to use n2d2 objects
    network = N2D2.Network(1)
    deepNet = N2D2.DeepNet(network)
    iniParser = N2D2.IniParser()
    deepNet.setDatabase(database)
    deepNet.setStimuliProvider(stimuliProvider)
    deepNet = N2D2.DeepNetGenerator.generateFromONNX(network, model_path, iniParser, deepNet)
    return n2d2.converter.deepNet_converter(deepNet)

def load_from_INI(path):
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
        self._sequence_dict = collections.OrderedDict()
        self._sequences = sequences
        self._cells_sequence = []

        self._generate_graph(self, '')

        print(self._sequence_dict)

        previous = None
        for key, value in self._sequence_dict.items():
            if isinstance(value, n2d2.cell.Cell):
                if previous is not None:
                    value.clear_input()
                    value.add_input(previous)
                previous = value


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

    # TODO: Method that converts sequencial representation into correspondind N2D2 deepnet

    def add_provider(self, provider):
        self._cells_sequence[0].add_input(provider)

    def clear_provider(self):
        self._cells_sequence[0].clear_input()

    def initialize(self):
        for cell in self._cells_sequence:
            cell.initialize()

    def propagate(self, inference=False):
        for cell in self._cells_sequence:
            cell.N2D2().propagate(inference=inference)

    def back_propagate(self):
        for cell in reversed(self._cells_sequence):
            cell.N2D2().backPropagate()

    def update(self):
        for cell in self._cells_sequence:
            cell.N2D2().update()

    def get_subsequence(self, name):
        return self._sequence_dict[name]

    def get_name(self):
        return self._Name

    def set_name(self, Name):
        self._Name = Name

    #def get_sequence_idx(self):
    #    return self._sequence_idx

    #def set_sequence_idx(self, idx):
    #    self._sequence_idx = idx

    def get_sequences(self):
        return self._sequences

    def get_output_cell(self):
        return self._sequences[-1].get_output_cell()

    def get_input_cell(self):
        return self._sequences[0].get_input_cell()

    def get_cells(self):
        return self._cells_sequence

    def __str__(self):
        output = "n2d2.cell.Sequence("
        output += self._generate_str()
        output += "\n)"
        return output

    def convert_to_INI_section(self):
        output = ""
        for cell in self._cells_sequence:
            output += cell.convert_to_INI_section()
            output += "\n"
        return output

    def _generate_str(self):
        output = ""
        for key, value in self._sequence_dict.items():
            indent_level = key.count('.')
            output += "\n" + (indent_level * "\t") + "(" + key + ")"
            #if sequence.get_name() is not None:
            #    output += " \'" + sequence.get_name() + "\'"
            #output += ": n2d2.cell.Sequence("
            if isinstance(value, n2d2.cell.Cell):
                output += ": " + value.__str__()
        return output
        