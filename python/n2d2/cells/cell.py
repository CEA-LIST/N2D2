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
from n2d2.deepnet import DeepNet
from abc import ABC, abstractmethod

class Cell(ABC):
    @abstractmethod
    def __init__(self, name):
        if not name:
            name = n2d2.generate_name(self)
        else:
            if not isinstance(name, str):
                raise n2d2.error_handler.WrongInputType("name", str(type(name)), ["str"])
        self._name = name

    def __call__(self, x):
        raise RuntimeError("Cell instance without __call__() method")

    def test(self):
        raise RuntimeError("Cell instance without test() method")

    def learn(self):
        raise RuntimeError("Cell instance without learn() method")

    def import_free_parameters(self, dir_name, ignoreNotExists=False):
        raise RuntimeError("Cell instance without import_free_parameters() method")

    def get_name(self):
        return self._name

    def get_type(self):
        return type(self).__name__


class Block(Cell):
    def __init__(self, cells, name=None):
        assert (isinstance(cells, list))
        self._cells = {}
        for cell in cells:
            self._cells[cell.get_name()] = cell
        Cell.__init__(self, name)

    def __getitem__(self, name):
        return self._cells[name]

    def __call__(self, x):
        raise RuntimeError("Block requires custom __call__() method")

    def test(self):
        for cell in self._cells.values():
            cell.test()

    def learn(self):
        for cell in self._cells.values():
            cell.learn()

    def import_free_parameters(self, dir_name, ignoreNotExists=False):
        for cell in self._cells.values():
            cell.import_free_parameters(dir_name, ignoreNotExists=ignoreNotExists)

    """
    def __str__(self):
        output = "\'" + self._name + "\' " + "[\n"
        for cell in self._cells.values():
            output += "\t" + str(cell) + "\n"
        output += "]\n"
        return output
    """

    def __str__(self):
        return self._generate_str(1)

    def _generate_str(self, indent_level):
        output = "\'" + self.get_name() + "\' " + self.get_type() + "("

        for idx, value in enumerate(self._cells.values()):
            output += "\n" + (indent_level * "\t") + "(" + str(idx) + ")"
            if isinstance(value, n2d2.cells.Block):
                output += ": " + value._generate_str(indent_level + 1)
            else:
                output += ": " + value.__str__()
        output += "\n" + ((indent_level - 1) * "\t") + ")"
        return output



class Sequence(Block):
    def __init__(self, cells, name=None):
        Block.__init__(self, cells, name)
        self._seq = cells

    def __call__(self, x):
        x.get_deepnet().begin_group(name=self._name)
        for cell in self._seq:
            x = cell(x)
        x.get_deepnet().end_group()
        return x

    def __getitem__(self, item):
        if isinstance(item, int):
            return list(self._cells.values())[item]
        else:
            return self._cells[item]

    def get_index(self, item):
        for i, cell in enumerate(self._seq):
            if item.get_name() == cell.get_name():
                return i
        raise RuntimeError("Element with name '" + item.get_name() + "' not found in sequence")


    def __len__(self):
        return len(self._seq)

    def insert(self, index, cell):
        self._seq.insert(index, cell)
        self._cells[cell.get_name()] = cell

    def _generate_str(self, indent_level):
        output = "\'" + self.get_name() + "\' " + self.get_type() + "("

        for idx, value in enumerate(self._seq):
            output += "\n" + (indent_level * "\t") + "(" + str(idx) + ")"
            if isinstance(value, n2d2.cells.Block):
                output += ": " + value._generate_str(indent_level + 1)
            else:
                output += ": " + value.__str__()
        output += "\n" + ((indent_level - 1) * "\t") + ")"
        return output


class DeepNetCell(Block):

    def __init__(self, N2D2_object):

        self._embedded_deepnet = DeepNet.create_from_N2D2_object(N2D2_object)

        if not N2D2_object.getName() == "":
            name = N2D2_object.getName()
        else:
            name = None

        #self._cells = self._embedded_deepnet.get_cells()
        Block.__init__(self, list(self._embedded_deepnet.get_cells().values()), name=name)

        self._deepnet = None # TODO : @Johannes : shouldn't this be equal to _embedded_deepnet ? 
        self._inference = False


    @classmethod
    def load_from_ONNX(cls, provider, model_path, ini_file=None):
        """Load a deepnet from an ONNX file given a provider object.
        :param provider: Provider object to base deepnet upon
        :type provider: n2d2.provider.Provider
        :param model_path: Path to the model.
        :type model_path: str
        :param ini_file: Path to an optional .ini file with additional onnx import instructions
        :type model_path: str
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
        N2D2_deepnet = N2D2.DeepNetGenerator.generateFromONNX(n2d2.global_variables.default_net, model_path, ini_parser,
                                            N2D2_deepnet, [None])
        n2d2_deepnet = cls(N2D2_deepnet)
        return n2d2_deepnet

    @classmethod
    def load_from_INI(cls, path):
        """Load a deepnet from an INI file.
        :param model_path: Path to the ini file.
        :type model_path: str
        """
        n2d2_deepnet = DeepNet.create_from_N2D2_object(
            N2D2.DeepNetGenerator.generateFromINI(n2d2.global_variables.default_net, path))
        return n2d2_deepnet

    def __call__(self, inputs):

        # TODO: Not tested for other inputs that provider yet
        if not isinstance(inputs, n2d2.Tensor):
            raise ValueError("Needs tensor with provider as input")

        #self._deepnet = self._infer_deepnet(inputs)

        #print(self._embedded_deepnet)

        # Recreate graph with underlying N2D2 deepnet
        self._deepnet = self.concat_to_deepnet(inputs.cell.get_deepnet())

        #if not provider.dims() == N2D2_object.getStimuliProvider().getData().dims():
        #    raise RuntimeError(
        #        "N2D2 object has input dimensions " + str(N2D2_object.getStimuliProvider().getData().dims()) +
        #        " while given provider has dimensions " + str(provider.dims()))

        #if inputs.nb_dims() != 4:
        #    raise ValueError("Input Tensor should have 4 dimensions, " + str(inputs.nb_dims()), " were given.")

        #self.get_first().set_deepnet(self._deepnet)
        for cell in self.get_input_cells():
            cell.N2D2().clearInputTensors()
            cell._link_N2D2_input(inputs.cell)

        self._deepnet.N2D2().propagate(self._inference)

        outputs = []
        for cell in self.get_output_cells():
            outputs.append(cell.get_outputs())
        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs

    def concat_to_deepnet(self, deepnet):
        new_n2d2_deepnet = deepnet


        #new_n2d2_deepnet.N2D2().setStimuliProvider(self._N2D2_object.getStimuliProvider())
        cells = self._embedded_deepnet.N2D2().getCells()
        layers = self._embedded_deepnet.N2D2().getLayers()
        if not layers[0][0] == "env":
            print("Is env:" + layers[0][0])
            raise RuntimeError("First layer of N2D2 deepnet is not a StimuliProvider. You may be skipping cells")

        # print("copy graph groups")
        # print(new_n2d2_deepnet._groups)

        #new_n2d2_deepnet.begin_group()
        for idx, layer in enumerate(layers[1:]):
            if len(layer) > 1:
                new_n2d2_deepnet.begin_group("layer" + str(idx))

            # print("Layer: " + str(idx))
            # print(layer)

            for cell in layer:
                N2D2_cell = cells[cell]
                # print("Adding cells: " + N2D2_cell.getName())
                parents = self._embedded_deepnet.N2D2().getParentCells(N2D2_cell.getName())
                if len(parents) == 1 and parents[0] is None:
                    parents = []
                new_n2d2_deepnet.N2D2().addCell(N2D2_cell, parents)
                n2d2_cell = self._embedded_deepnet.get_cells()[N2D2_cell.getName()]
                n2d2_cell.set_deepnet(new_n2d2_deepnet)
                new_n2d2_deepnet.add_to_current_group(n2d2_cell)
            if len(layer) > 1:
                new_n2d2_deepnet.end_group()
        #new_n2d2_deepnet.end_group()

        return new_n2d2_deepnet
    
    #def clear_data_tensors(self):
    #    for cell in self._embedded_deepnet.get_cells().values():
    #        cell.clear_data_tensors()

    def update(self):
        self.get_deepnet().update()

    def test(self):
        self._inference = True

    def learn(self):
        self._inference = False

    def import_free_parameters(self, dir_name, ignoreNotExists=False):
        self._deepnet.N2D2().importNetworkFreeParameters(dir_name, ignoreNotExists=ignoreNotExists)

    def remove(self, name):
        cell = self._embedded_deepnet.N2D2().getCells()[name]
        self._embedded_deepnet.N2D2().removeCell(cell, False)
        self._embedded_deepnet = DeepNet.create_from_N2D2_object(self._embedded_deepnet.N2D2())
        self._cells = self._embedded_deepnet.get_cells()

    def get_deepnet(self):
        return self._deepnet

    def get_embedded_deepnet(self):
        return self._embedded_deepnet

    def get_input_cells(self):
        output = []
        cells = self._embedded_deepnet.get_groups().get_elements()[0]
        if isinstance(cells, n2d2.deepnet.Group):
            for name in cells.get_cells():
                output.append(self._embedded_deepnet.get_cells()[name])
        else:
            output.append(cells)
        return output

    def get_output_cells(self):
        output = []
        cells = self._embedded_deepnet.get_groups().get_elements()[-1]
        if isinstance(cells, n2d2.deepnet.Group):
            for name in cells.get_cells():
                output.append(self._embedded_deepnet.get_cells()[name])
        else:
            output.append(cells)
        return output






