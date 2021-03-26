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

    def __init__(self, from_parameters=True, **config_parameters):
        N2D2_Interface.__init__(self, **config_parameters)

        if not 'name' in self._config_parameters:
            self._config_parameters['name'] = "DeepNet" + str(n2d2.global_variables.deepNet_counter)
            n2d2.global_variables.deepNet_counter += 1

        if from_parameters:
            self._create_from_parameters()

        self._groups = Group(self._config_parameters['name'])
        #self.set_provider(provider)
        self._current_group = self._groups
        self._group_dict = {self._config_parameters['name']: self._groups}
        self._group_counter = 1

    def add_to_current_group(self, cell):
        self._current_group.add(cell)

    def begin_group(self, name=None):
        if name is None:
            name = "Group" + str(self._group_counter)
        if name in self._group_dict:
            raise RuntimeError("Group with name '" + name + "' already exists in DeepNet")
        self._current_group = Group(name, self._current_group)
        self._group_dict[name] = self._current_group
        self._group_counter += 1

    def end_group(self):
        self._current_group.get_parent_group().add(self._current_group)
        self._current_group = self._current_group.get_parent_group()


    def _create_from_parameters(self):
        self._network = n2d2.global_variables.default_net 
        self._N2D2_object = N2D2.DeepNet(self._network)
        self._set_N2D2_parameters(self._config_parameters)

    @classmethod
    def create_from_N2D2_object(cls, provider, N2D2_object):
        if not provider.dims() == N2D2_object.getStimuliProvider().getData().dims():
            raise RuntimeError("N2D2 object has input dimensions " + str(N2D2_object.getStimuliProvider().getData().dims()) +
                               " while given provider has dimensions " + str(provider.dims()))
        deepnet = cls(from_parameters=False, **N2D2_Interface._load_N2D2_parameters(N2D2_object))
        deepnet._N2D2_object = N2D2_object
        deepnet.set_provider(provider)
        deepnet._network = deepnet._N2D2_object.getNetwork()

        cells = deepnet._N2D2_object.getCells()
        layers = deepnet._N2D2_object.getLayers()
        if not layers[0][0] == "env":
            print("Is env:" + layers[0][0])
            raise RuntimeError("First layer of N2D2 deepnet is not a StimuliProvider. You may be skipping cells")
        for idx, layer in enumerate(layers[1:]):
            if len(layer) > 1:
                deepnet.begin_group()
                for cell in layer:
                    N2D2_cell = cells[cell]
                    n2d2_cell = n2d2.converter.cell_converter(N2D2_cell, deepnet)
                    if idx == 0:
                        n2d2_cell.clear_input() # Remove old stimuli provider
                        n2d2_cell.add_input(n2d2.GraphTensor(None, provider))
                deepnet.end_group()
            else:
                N2D2_cell = cells[layer[0]]
                n2d2_cell = n2d2.converter.cell_converter(N2D2_cell, deepnet)
                if idx == 0:
                    n2d2_cell.clear_input()  # Remove old stimuli provider
                    n2d2_cell.add_input(n2d2.GraphTensor(None, provider))
        return deepnet


    @classmethod
    def load_from_ONNX(cls, provider, model_path, ini_file=None):
        """
        :param provider: Provider object to base deepnet upon
        :type provider: n2d2.provider.Provider
        :param model_path: Path to the model.
        :type model_path: str
        :param ini_file: Path to an optional .ini file with additional onnx import instructions
        :type model_path: str
        Load a deepnet from an ONNX file given a provider object.
        """
        if not isinstance(provider, n2d2.provider.Provider):
            raise ValueError("Input needs to be of type 'provider'")
        N2D2_deepnet = N2D2.DeepNet(n2d2.global_variables.default_net)
        N2D2_deepnet.setStimuliProvider(provider.N2D2())
        if isinstance(provider, n2d2.provider.DataProvider):
            N2D2_deepnet.setDatabase(provider.get_database().N2D2())
        N2D2.CellGenerator.defaultModel = n2d2.global_variables.default_model
        ini_parser = N2D2.IniParser()
        if ini_file is not None:
            ini_parser.load(ini_file)
        ini_parser.currentSection("onnx", True)
        N2D2_deepnet = N2D2.DeepNetGenerator.generateFromONNX(n2d2.global_variables.default_net, model_path, ini_parser, N2D2_deepnet)
        n2d2_deepnet = DeepNet.create_from_N2D2_object(provider, N2D2_deepnet)
        return n2d2_deepnet

    @classmethod
    def load_from_INI(cls, path):
        """
        :param model_path: Path to the ini file.
        :type model_path: str
        Load a deepnet from an INI file.
        """
        n2d2_deepnet = DeepNet.create_from_N2D2_object(N2D2.DeepNetGenerator.generateFromINI(n2d2.global_variables.default_net, path))
        return n2d2_deepnet
    

    def propagate(self):
        self._N2D2_object.propagate(self._inference)

    def back_propagate(self):
        self._N2D2_object.backPropagate()

    def update(self):
        self._N2D2_object.update()

    def import_free_parameters(self, dirName, ignoreNotExists=False):
        self._N2D2_object.importNetworkFreeParameters(dirName, ignoreNotExists=ignoreNotExists)

    def set_provider(self, provider):
        self._provider = provider
        self._N2D2_object.setStimuliProvider(provider.N2D2())

    def get_provider(self):
        return self._provider

    def get_cells(self):
        return self._groups.get_cells()

    def get_last(self):
        if len(self._groups) == 0:
            return self._provider
        else:
            return self._groups.get_last()

    def get_first(self):
        return self._groups.get_first()

    def get_outputs(self):
        return self.get_last().get_outputs()

    def dims(self):
        return self.get_outputs().dims()

    def get_group(self, group_id):
        return self._groups.get_group(group_id)

    def draw(self, filename):
        N2D2.DrawNet.draw(self._N2D2_object, filename)

    def draw_graph(self, filename):
        N2D2.DrawNet.drawGraph(self._N2D2_object, filename)

    def remove(self, idx, reconnect=True):
        self._groups.remove(idx, reconnect)

    def __str__(self):
        return self._groups.__str__()


# TODO: Merge into DeepNet
class Group:
    def __init__(self, name, parent_group=None):
        self._name = name
        self._sequence = []
        self._parent_group = parent_group

    def add(self, cell):
        if cell.get_name() in self.get_cells():
            raise RuntimeError("Cell with name '" + cell.get_name() + "' already exists in group '" + self._name + "'. "
                               "Are you trying to call the same cell twice? Cyclic graphs are not supported.")
        self._sequence.append(cell)

    def __len__(self):
        return len(self._sequence)

    def get_parent_group(self):
        return self._parent_group


    # TODO: At the moment this does not release memory of deleted cells
    def remove(self, idx, reconnect=True):
        cell = self._sequence[idx]
        print("Removing element: " + cell.get_name())
        if isinstance(cell, Group):
            length = len(cell.get_elements())
            for i in reversed(range(length)):
                cell.remove(i, reconnect)
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

    """
    def import_free_parameters(self, dirName, ignoreNotExists=False):
        print("Importing weights from directory '" + dirName + "'")
        for name, cell in self.get_cells().items():
            path = dirName + "/" + name + ".syntxt"
            cell.import_free_parameters(path, ignoreNotExists=ignoreNotExists)
            cell.import_activation_parameters(dirName, ignoreNotExists=ignoreNotExists)
    """

    def get_group(self, group_id):
        if isinstance(group_id, int):
            return self._sequence[group_id]
        else:
            for elem in self._sequence:
                if elem.get_name() == group_id:
                    return elem
            raise RuntimeError("No group with name: \'" + group_id + "\'")

    def get_name(self):
        return self._name

    def get_elements(self):
        return self._sequence

    def get_last(self):
        return self._sequence[-1].get_last()

    def get_first(self):
        return self._sequence[0].get_first()

    def get_outputs(self):
        return self.get_last().get_outputs()


    def __str__(self):
        return self._generate_str(1)

    def _generate_str(self, indent_level):
        if not self.get_name() == "":
            output = "\'" + self.get_name() + "\' " + "("
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

