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
import collections

"""
Abstract class that stores the computation graph and the N2D2 deepnet object
"""
class Deepnet:

    def __init__(self, deepnet, block_descriptor, model_parameters):
        if not isinstance(block_descriptor, n2d2.cell.Block):
            raise TypeError("Error: Deepnet constructor expects a Cell Block, but got " + str(type(block)) + " instead")
        self._block_descriptor = block_descriptor
        self._model_parameters = model_parameters

        #net = N2D2.Network(1)
        self._deepnet = deepnet #N2D2.DeepNet(net)
    
    # Prepares cells for computation and initializes certain members
    # NOTE: In the current implementation of sequential, this does not work, since the cell see the deepnet
    # but the deepnet does not see the cells
    #def initialize(self):
    #     self._deepnet.initialize()

    def N2D2(self):
        return self._deepnet

    def __str__(self):
        return ""


"""
Allows  creation of a deepnet based on a nested Block object. 
Hardware model and datatype are given at construction time
"""    
class Sequential(Deepnet):
    # cells is typically a python list 
    def __init__(self, deepnet, block_descriptor, Model='Frame', DataType='float', **model_parameters):
        super().__init__(deepnet, block_descriptor, model_parameters)

        # Non nested representation of cells for easier access
        # We do not use OrderedDict since names are already members of cells
        # and we want redundancy between dict keys and cell names
        self._sequence = []

        self._blocks = {}
        # Unfold nested network graph
        block_name = ''

        self._generate_model(self._block_descriptor, Model, DataType, block_name)

        print(self._blocks)

        self._Model = Model
        self._DataType = DataType

        names = [_.get_name() for _ in self._sequence]

        # Check if cell associated to model parameters exists
        for key in self._model_parameters:
            if not key.replace('_model', '') in names:
                raise RuntimeError("No matching cell for model parameter: " + key)

        #for cell in self._sequence:
        #    print(cell._Name)

    """Goes recursively through blocks"""
    def _generate_model(self, block, Model, DataType, block_name):

        if block_name is not "" and block.get_name() is None:
            block.set_name(block_name)

        if block.get_name() in self._blocks:
            raise RuntimeError("Block with name \'" + block.get_name() + "\' already exists")
        else:
            self._blocks[block.get_name()] = block

        if isinstance(block.get_blocks(), list):
            if block_name is not "":
                block_name += "."
            for idx, sub_block in enumerate(block.get_blocks()):
                self._generate_model(sub_block, Model, DataType, block_name + str(idx))
        else:
            if block.get_name() + '_model' in self._model_parameters:
                block.generate_model(self._deepnet, Model, DataType, **self._model_parameters[block.get_name() + '_model'])
            else:
                block.generate_model(self._deepnet, Model, DataType)
            # Normally this should not copy, but only add an additional name
            if len(self._sequence) > 0:
                block.add_input(self._sequence[-1])
            self._sequence.append(block)


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
            if name == cell.get_name():
                return cell
        raise RuntimeError("Cell: " + name + " not found")

    def get_sequence(self):
        return self._sequence

    def get_model(self):
        return self._Model


    def __str__(self):
        indent_level = [0]
        output = "n2d2.deepnet.Sequential("
        output += self._generate_str(self._block_descriptor, indent_level, [0])
        output += "\n)"
        return output

    # TODO: Do without artificial mutable objects
    # TODO: This should be moved to Block
    def _generate_str(self, block, indent_level, block_idx):
        output = ""
        if isinstance(block.get_blocks(), list):
            if indent_level[0] > 0:
                output += "\n"+ (indent_level[0]*"\t") + "Block_" + block.get_name() + ": ["
            indent_level[0] += 1
            local_block_idx = [0]
            for idx, block in enumerate(block.get_blocks()):
                output += self._generate_str(block, indent_level, local_block_idx)
                local_block_idx[0] += 1
            indent_level[0] -= 1
            if indent_level[0] > 0:
                output += "\n"+ (indent_level[0]*"\t") + "]"
        else:
            output += "\n"+ (indent_level[0]*"\t") + "Block_" + block.get_name() + ": " + block.__str__()
        return output

    def convert_to_INI_section(self):
        output = ""
        for cell in self._sequence:
            output += cell.convert_to_INI_section()
            output += "\n"
        return output
