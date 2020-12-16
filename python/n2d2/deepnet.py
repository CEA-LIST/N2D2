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

class Deepnet():

    """
    def __init__(self):
        self._cells = None
        net = N2D2.Network(1)
        self._deepnet = N2D2.DeepNet(net)
    """

    # TODO: Proper exeception handling
    def __init__(self, deepnet, cells, model_parameters):
        if not isinstance(cells, list):
            raise TypeError("Error: Deepnet constructor expects a List of Cells, but got " + str(type(cells)) + " instead")
        self._cells = cells
        self._model_parameters = model_parameters

        self._Model = None

        #net = N2D2.Network(1)
        self._deepnet = deepnet #N2D2.DeepNet(net)
    
    # Prepares cells for computation and initializes certain members
    # NOTE: In the current implementation of sequential, this does not work, since the cell see the deepnet
    # but the deepnet does not see the cells
    #def initialize(self):
    #     self._deepnet.initialize()

    def N2D2(self):
        if self._deepnet is None:
            raise n2d2.UndefinedModelError("N2D2 deepnet member has not been created. Did you run generate_model?")
        return self._deepnet

    def __str__(self):
        return ""
    
"""
This should be usable similar to torch.nn.Sequential.
That means also recursive (sequence of sequence etc.).
Allows simple creation of a deepnet based on standard
Python data structures, without using N2D2 binding functions 
    Input:
    * (List of Python Cell objects, Model)
    * (List of Lists, Model)
"""    
class Sequential(Deepnet):
    # cells is typically a python list 
    def __init__(self, deepnet, cells, Model='Frame', DataType='float', **model_parameters):
        super().__init__(deepnet, cells, model_parameters)

        # Non nested representation of cells for easier access
        self._sequence = []
        # Unfold nested network graph
        self._generate_model(self._cells, Model, DataType)

        self._Model = Model
        self._DataType = DataType

        # Check if cell associated to model parameters exists
        # For the moment, sequence is not a dictionary with cell names.
        names = [_.get_name() for _ in self._sequence]
        for key in self._model_parameters:
            if not key.replace('_model', '') in names:
                raise RuntimeError("No matching cell for model parameter: " + key)

        #for cell in self._sequence:
        #    print(cell._Name)

    # TODO: Check that cell names are unique
    def _generate_model(self, cells, Model, DataType):
        if isinstance(cells, list):
            for cell in cells:
                self._generate_model(cell, Model, DataType)
        elif isinstance(cells, n2d2.cell.Cell):
            if cells.get_name() + '_model' in self._model_parameters:
                cells.generate_model(self._deepnet, Model, DataType, **self._model_parameters[cells.get_name() + '_model'])
            else:
                cells.generate_model(self._deepnet, Model, DataType)
            # Normally this should not copy, but only add an additional name
            if len(self._sequence) > 0:
                cells.add_input(self._sequence[-1])
            self._sequence.append(cells)
        else:
            raise TypeError("Error: Expected a Cell, but got " + str(type(cells)) + " instead")

    """ 
        addInput() sets recursively the Tensor dimensions
        of input and output tensors of all cells
    """
    def add_provider(self, provider):
        if len(self._sequence) > 0:
            self._sequence[0].add_input(provider)
        else:
            raise n2d2.UndefinedModelError("No cells in deepnet")


    """ # Redunant when using self._sequence
    def _generate_cell_links(self, cells, previous):
        if isinstance(cells, list):
            for cell in cells:
                previous = self._generate_cell_links(cell, previous)
            return previous
        else:
            cells.N2D2().addInput(previous.N2D2())
            return cells
    """

    def initialize(self):
        for cell in self._sequence:
            cell.initialize()

    def propagate(self, inference=False):
        for cell in self._sequence:
            cell.N2D2().propagate(inference=inference)

    def back_propagate(self):
        for cell in reversed(self._sequence):
            cell.N2D2().backPropagate()

    def update(self):
        for cell in self._sequence:
            cell.N2D2().update()

    def get_output(self):
        return self._sequence[-1]

    def get_cell(self, name):
        for cell in self._sequence:
            if name is cell.get_name():
                return cell
        raise RuntimeError("Cell: " + name + " not found")

    def get_model(self):
        if self._Model is None:
            raise n2d2.UndefinedModelError("Model variable is undefined. Did you run generate_model?")
        else:
            return self._Model


    def __str__(self):
        indent_level = [0]
        output = "n2d2.deepnet.Sequential("
        output += self._generate_str(self._cells, indent_level, [0])
        output += "\n)"
        return output

    # TODO: Do without artificial mutable objects
    def _generate_str(self, cells, indent_level, block_idx):
        output = ""
        if isinstance(cells, list):
            if indent_level[0] > 0:
                output += "\n"+ (indent_level[0]*"\t") + "(" + str(block_idx[0]) + ") " +"'Block'"
            indent_level[0] += 1
            local_block_idx = [0]
            for idx, cell in enumerate(cells):
                output += self._generate_str(cell, indent_level, local_block_idx)
                local_block_idx[0] += 1
            indent_level[0] -= 1
        else:
            output += "\n"+ (indent_level[0]*"\t") + "(" + str(block_idx[0]) + ") " + cells.__str__()
        return output

    def convert_to_INI_section(self):
        output = ""
        for cell in self._sequence:
            output += cell.convert_to_INI_section()
            output += "\n"
        return output
