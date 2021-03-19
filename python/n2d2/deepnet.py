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


    def propagate(self, inference=False):
        self._N2D2_object.propagate(inference)

    def back_propagate(self):
        self._N2D2_object.backPropagate()

    def update(self):
        self._N2D2_object.update()

    def import_free_parameters(self, dirName, ignoreNotExists=False):
        print("Importing weights from directory '" + dirName + "'")
        self._N2D2_object.import_free_parameters(dirName, ignoreNotExists=ignoreNotExists)


    def add_provider(self, provider):
        self._provider = provider
        self._N2D2_object.setStimuliProvider(provider.N2D2())

    """
    def __str__(self):
        output = "Deepnet" + "(" + self._model + "<" + self._datatype + ">" + ")"
        output += N2D2_Interface.__str__(self)
        return output
    """

    def get_model(self):
        return self._model

    def get_datatype(self):
        return self._datatype

    def draw(self, filename):
        N2D2.DrawNet.draw(self._N2D2_object, filename)

    def draw_graph(self, filename):
        N2D2.DrawNet.drawGraph(self._N2D2_object, filename)


    @classmethod
    def load_from_ONNX(cls, model_path, dims, batch_size=1, ini_file=None):
        """
        :param model_path: Path to the model.
        :type model_path: str
        :param dims:
        :type dims: list
        :param batch_size:
        :type batch_size: unsigned int
        Load a deepnet from an ONNX file given its input dimensions.
        """
        deepNet = cls()
        provider = n2d2.provider.DataProvider(n2d2.database.Database(), dims, batchSize=batch_size)
        deepNet.N2D2().setDatabase(provider.get_database().N2D2())
        deepNet.N2D2().setStimuliProvider(provider.N2D2())
        N2D2.CellGenerator.defaultModel = "Frame_CUDA" #deepNet.get_model()
        ini_parser = N2D2.IniParser()
        if ini_file is not None:
            ini_parser.load(ini_file)
        ini_parser.currentSection("onnx", True)
        N2D2_deepNet = N2D2.DeepNetGenerator.generateFromONNX(n2d2.global_variables.default_net, model_path, ini_parser, deepNet.N2D2())
        model = n2d2.converter.deepnet_converter(N2D2_deepNet)
        model.get_first().clear_input() #Remove dummy stimuli provider
        return model

    @classmethod
    def load_from_INI(cls, path):
        """
        :param model_path: Path to the ini file.
        :type model_path: str
        Load a deepnet from an INI file.
        """
        deepNet = cls(N2D2.DeepNetGenerator.generateFromINI(n2d2.global_variables.default_net, path))
        return n2d2.converter.deepnet_converter(deepNet)



"""
Structure that is organised sequentially. 
"""

# TODO: Mark the first and last cell in the print
class Group:
    def __init__(self, sequence, name=""):
        assert isinstance(name, str)
        self._name = name
        assert isinstance(sequence, list)
        self._sequence = sequence
        last_deepnet = None
        for ipt in self._sequence:
            deepnet = ipt.get_last().get_deepnet()
            if last_deepnet is not None:
                if not id(deepnet) == id(last_deepnet):
                    print(id(deepnet))
                    print(id(last_deepnet))
                    raise RuntimeError("Cells of group have different deepnets")
            last_deepnet = deepnet
        self._deepnet = last_deepnet

        # By default the input and output cells are the first and last element of the sequence
        # To change this, modify these members
        self._first = None
        self._last = None

    def add(self, cell):
        if self._deepnet is not None:
            if not id(cell.get_deepnet()) == id(self.get_deepnet()):
                raise RuntimeError("Deepnet of cell is different than group deepnet")
        else:
            self._deepnet = cell.get_deepnet()
        self._sequence.append(cell)


    def get_deepnet(self):
        return self._deepnet

    # TODO: At the moment this does not release memory of deleted cells
    def remove(self, idx, reconnect=True):
        cell = self._sequence[idx]
        print("Removing cell: " + cell.get_name())
        if isinstance(cell, Group):
            for _ in cell.get_elements():
                cell.remove(0, reconnect)
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
            if isinstance(elem, Group):
                elem._get_cells(cells)
            else:
                cells[elem.get_name()] = elem

    def add_input(self, inputs):
        self.get_first().add_input(inputs)

    def clear_input(self):
        self.get_first().clear_input()

    def get_outputs(self):
        return self.get_last().get_outputs()

    def dims(self):
        return self.get_last().get_outputs().dims()

    def get_nb_outputs(self):
        return self.get_last().get_nb_outputs()

    def __getitem__(self, item):
        return self.get_cells()[item]


    """
    def import_free_parameters(self, dirName, ignoreNotExists=False):
        print("Importing weights from directory '" + dirName + "'")
        for name, cell in self.get_cells().items():
            path = dirName + "/" + name + ".syntxt"
            cell.import_free_parameters(path, ignoreNotExists=ignoreNotExists)
            cell.import_activation_parameters(dirName, ignoreNotExists=ignoreNotExists)
    """


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
        if self._last is not None:
            return self._last
        else:
            return self._sequence[-1].get_last()

    def get_first(self):
        if self._first is not None:
            return self._first
        else:
            return self._sequence[0].get_first()

    def __str__(self):
        return self._generate_str(1)

    def _generate_str(self, indent_level):
        if not self.get_name() == "":
            output = "\'" + self.get_name() + "\' " + "Group("
        else:
            output = "Group("

        for idx, value in enumerate(self._sequence):
            output += "\n" + (indent_level * "\t") + "(" + str(idx) + ")"
            if isinstance(value, n2d2.deepnet.Group):
                output += ": " + value._generate_str(indent_level + 1)
            else:
                output += ": " + value.__str__()
        output += "\n" + ((indent_level-1) * "\t") + ")"
        return output


"""
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
            if isinstance(elem, Group):
                elem._get_cells(cells)
            else:
                cells[elem.get_name()] = elem

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
"""
