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
import n2d2.global_variables
import n2d2.cell
import n2d2.converter
from n2d2.n2d2_interface import N2D2_Interface


class DeepNet(N2D2_Interface):

    def __init__(self, N2D2_object=None, **config_parameters):


        if 'model' in config_parameters:
            self._model = config_parameters.pop('model')
        else:
            self._model = n2d2.global_variables.default_model
        if 'dataType' in config_parameters:
            self._datatype = config_parameters.pop('dataType')
        else:
            self._datatype = n2d2.global_variables.default_dataType

        if N2D2_object is None:
            self._create_from_parameters(**config_parameters)
        else:
            if len(config_parameters) > 0:
                raise RuntimeError("N2D2_object given but len(config_parameters) > 0")
            self._create_from_N2D2_object(N2D2_object)

        # Even though a deepnet object does not require a provider, some methods using the deepnet
        # expect it to have one. For these cases we have to add a dummy provider
        self._provider = None

    def _create_from_parameters(self, **config_parameters):
        N2D2_Interface.__init__(self, **config_parameters)
        self._network = n2d2.global_variables.default_net #N2D2.Network(n2d2.global_variables.default_seed)
        self._N2D2_object = N2D2.DeepNet(self._network)
        self._set_N2D2_parameters(self._config_parameters)


    def _create_from_N2D2_object(self, N2D2_object):
        N2D2_Interface.__init__(self,  **self._load_N2D2_parameters(N2D2_object))
        self._N2D2_object = N2D2_object
        self._network = self._N2D2_object.getNetwork()



    def add_provider(self, provider):
        self._provider = provider
        self._N2D2_object.setStimuliProvider(provider.N2D2())

    def __str__(self):
        output = "Deepnet" + "(" + self._model + "<" + self._datatype + ">" + ")"
        output += N2D2_Interface.__str__(self)
        return output

    def get_model(self):
        return self._model

    def get_datatype(self):
        return self._datatype

    def draw(self, filename):
        N2D2.DrawNet.draw(self._N2D2_object, filename)

    def draw_graph(self, filename):
        N2D2.DrawNet.drawGraph(self._N2D2_object, filename)


def load_from_ONNX(model_path, dims, batch_size=1, ini_file=None):
    """
    :param model_path: Path to the model.
    :type model_path: str
    :param dims:
    :type dims: list
    :param batch_size:
    :type batch_size: unsigned int
    Load a deepnet from an ONNX file given its input dimensions.
    """
    deepNet = DeepNet()
    provider = n2d2.provider.DataProvider(n2d2.database.Database(), dims, batchSize=batch_size)
    deepNet.N2D2().setDatabase(provider.get_database().N2D2())
    print(provider.get_database().N2D2())
    deepNet.N2D2().setStimuliProvider(provider.N2D2())
    N2D2.CellGenerator.defaultModel = "Frame_CUDA" #deepNet.get_model()
    print(N2D2.CellGenerator.defaultModel)
    print("Launch generator")
    ini_parser = N2D2.IniParser()
    if ini_file is not None:
        ini_parser.load(ini_file)
    ini_parser.currentSection("onnx", True)
    deepNet = N2D2.DeepNetGenerator.generateFromONNX(n2d2.global_variables.default_net, model_path, ini_parser, deepNet.N2D2())
    print("Launch converter")
    model = n2d2.converter.deepNet_converter(deepNet)
    model.get_first().clear_input() #Remove dummy stimuli provider
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
Structure that is organised sequentially. 
"""
class Sequence:
    def __init__(self, sequence, name=""):
        assert isinstance(name, str)
        self._name = name
        assert isinstance(sequence, list)
        self._sequence = sequence

    def add(self, cell):
        self._sequence.append(cell)

    # TODO: Implement in layer
    # TODO: At the moment this does not release memory of deleted cells
    def remove_subsequence(self, idx, reconnect=True):
        cell = self._sequence[idx]
        print("Removing cell: " + cell.get_name())
        if isinstance(cell, Sequence) or isinstance(cell, Layer):
            for _ in cell.get_elements():
                cell.remove_subsequence(0, reconnect)
            print("delete: " + self._sequence[idx].get_name())
            del self._sequence[idx]
        elif isinstance(cell, n2d2.cell.Cell):
            cells = self.get_cells()
            children = cell.N2D2().getChildrenCells()
            parents = cell.N2D2().getParentsCells()
            for child in children:
                if child.getName() in cells:
                    n2d2_child = cells[child.getName()]
                    print("Child: " + n2d2_child.get_name())
                    for idx, ipt in enumerate(n2d2_child.get_inputs()):
                        if ipt.get_name() == cell.get_name():
                            del n2d2_child.get_inputs()[idx]
                    if reconnect:
                        for parent in parents:
                            if parent.getName() in cells:
                                n2d2_child.add_input(cells[parent.getName()])
                            else:
                                print("Warning: parent '" + parent.getName() + "' of removed cell '" + cell.get_name() +
                                "' not found in same sequence as removed cell. If the parent is part of another sequence, "
                                "please reconnect it manually.")
                else:
                    print("Warning: child '" + child.getName() + "' of removed cell '" + cell.get_name() +
                          "' not found in same sequence as removed cell. If the child is part of another sequence, "
                          "please remove the corresponding parent cell manually.")
            cell.get_deepnet().N2D2().removeCell(cell.N2D2(), reconnect)
            del self._sequence[idx]
        else:
            raise RuntimeError("Unknown object at index: " + str(idx))


    def get_cells(self):
        cells = {}
        self._get_cells(cells)
        return cells

    def _get_cells(self, cells):
        for elem in self._sequence:
            if isinstance(elem, Sequence) or isinstance(elem, Layer):
                elem._get_cells(cells)
            else:
                cells[elem.get_name()] = elem
    """
    def add_input(self, inputs):
        self.get_first().add_input(inputs)
    
    def get_inputs(self):
        return self.get_first().get_inputs()

    def clear_input(self):
        self.get_first().clear_input()
    """

    # TODO: Is this really used?
    def initialize(self):
        for cell in self._sequence:
            cell.initialize()

    def propagate(self, inference=False):
        for cell in self._sequence:
            cell.propagate(inference)

    def back_propagate(self):
        for cell in reversed(self._sequence):
            cell.back_propagate()

    def update(self):
        for cell in self._sequence:
            cell.update()

    def import_free_parameters(self, dirName, ignoreNotExists=False):
        print("Importing weights from directory '" + dirName + "'")
        for name, cell in self.get_cells().items():
            path = dirName + "/" + name + ".syntxt"
            cell.import_free_parameters(path, ignoreNotExists=ignoreNotExists)
            cell.import_activation_parameters(dirName, ignoreNotExists=ignoreNotExists)

    def get_subsequence(self, id):
        if isinstance(id, int):
            return self._sequence[id]
        else:
            for elem in self._sequence:
                if elem.get_name() == id:
                    return elem
            raise RuntimeError("No subsequence with name: \'" + id + "\'")

    def get_name(self):
        return self._name

    def set_name(self, name):
        self._name = name

    def get_elements(self):
        return self._sequence

    def get_last(self):
        return self._sequence[-1].get_last()

    def get_first(self):
        return self._sequence[0].get_first()

    def __str__(self):
        return self._generate_str(1)

    def _generate_str(self, indent_level):
        if not self.get_name() == "":
            output = "\'" + self.get_name() + "\' " + "Sequence("
        else:
            output = "Sequence("

        for idx, value in enumerate(self._sequence):
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
    def __init__(self, layer, name=""):
        assert isinstance(name, str)
        self._name = name
        assert isinstance(layer, list)
        if not layer:
            raise ValueError("Got empty list as input. List must contain at least one element")
        self._layer = layer

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

    """
    def add_input(self, input):
        for cell in self._layer:
            cell.add_input(input)

    def clear_input(self):
        for cell in self._layer:
            cell.clear_input()
    """

    def get_last(self):
        return self

    def get_first(self):
        return self

    def get_elements(self):
        return self._layer

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

