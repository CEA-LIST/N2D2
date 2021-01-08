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
import n2d2.activation
import n2d2.solver
from n2d2.n2d2_interface import N2D2_Interface

"""
Structure that is organised sequentially. 
"""
class Block:
    def __init__(self, blocks, Name=None):
        if Name is not None:
            assert isinstance(Name, str)
        self._Name = Name
        assert isinstance(blocks, list)
        if not blocks:
            raise ValueError("Got empty list as input. List must contain at least one element")

        self._block_idx = ''
        self._block_dict = {}
        self._cells = []
        self._blocks = blocks

        self._generate_graph(self, self._block_idx)

        for idx, block in enumerate(self._blocks):
            if idx > 0:
                block.get_input_cell().add_input(previous.get_output_cell())
            previous = block


    """Goes recursively through blocks"""

    def _generate_graph(self, block, block_idx):

        self._block_dict[block_idx] = block
        block.set_block_idx(block_idx)

        if isinstance(block.get_blocks(), list):
            if not block_idx == "":
                block_idx += "."
            for idx, sub_block in enumerate(block.get_blocks()):
                self._generate_graph(sub_block, block_idx + str(idx))
        else:
            self._cells.append(block)

    def get_name(self):
        return self._Name

    def set_name(self, Name):
        self._Name = Name

    def get_block_idx(self):
        return self._block_idx

    def set_block_idx(self, idx):
        self._block_idx = idx

    def get_blocks(self):
        return self._blocks

    def get_output_cell(self):
        return self._blocks[-1].get_output_cell()

    def get_input_cell(self):
        return self._blocks[0].get_input_cell()

    def get_cells(self):
        return self._cells

    def __str__(self):
        indent_level = [0]
        output = "n2d2.cell.Block("
        output += self._generate_str(self, indent_level, [0])
        output += "\n)"
        return output

    # TODO: Do without artificial mutable objects

    def _generate_str(self, block, indent_level, block_idx):
        output = ""
        if isinstance(block.get_blocks(), list):
            if indent_level[0] > 0:
                output += "\n" + (indent_level[0] * "\t") + "(" + str(block.get_block_idx()) + ")"
                if block.get_name() is not None:
                    output += " \'" + block.get_name() + "\'"
                output += ": n2d2.cell.Block("
            indent_level[0] += 1
            local_block_idx = [0]
            for idx, block in enumerate(block.get_blocks()):
                output += self._generate_str(block, indent_level, local_block_idx)
                local_block_idx[0] += 1
            indent_level[0] -= 1
            if indent_level[0] > 0:
                output += "\n" + (indent_level[0] * "\t") + ")"
        else:
            output += "\n" + (indent_level[0] * "\t") + "(" + str(block.get_block_idx()) + ")"
            if block.get_name() is not None:
                output += " \'" + block.get_name() + "\'"
            output += ": " + block.__str__()
        return output


class Cell(Block, N2D2_Interface):

    def __init__(self, NbOutputs, Name, **config_parameters):

        if 'Model' in config_parameters:
            self._Model = config_parameters.pop('Model')
        else:
            self._Model = n2d2.global_variables.default_Model
        if 'DataType' in config_parameters:
            self._DataType = config_parameters.pop('DataType')
        else:
            self._DataType = n2d2.global_variables.default_DataType

        self._model_key = self._Model + '<' + self._DataType + '>'

        #Block.__init__(self, Name)
        N2D2_Interface.__init__(self, **config_parameters)

        self._Name = Name
        self._blocks = self

        self._inputs = []

        self._constructor_arguments.update({
            'NbOutputs': NbOutputs,
        })

        net = N2D2.Network()
        self._deepnet = N2D2.DeepNet(net)


    def get_output_cell(self):
        return self

    def get_input_cell(self):
        return self

    def get_type(self):
        return self._N2D2_object.getType()

    def add_input(self, cell):
        self._inputs.append(cell)

    def initialize(self):
        for cell in self._inputs:
            self._N2D2_object.addInput(cell.N2D2())
        self._N2D2_object.initialize()


    def __str__(self):
        output = self.get_type()+"Cell(" + self._model_key + ")"
        output += N2D2_Interface.__str__(self)
        return output


    def convert_to_INI_section(self):
        output = ""
        """Possible to create section without name"""
        if self._Name is not None:
            output = "[" + self._Name + "]\n"
        output += "Input="
        for idx, cell in enumerate(self._inputs):
            if idx > 0:
                output += ","
            output += cell.get_name()
        output += "\n"
        output += "Type=" + self.get_type() + "\n"
        output += "NbOutputs=" + str(self._constructor_arguments['NbOutputs']) + "\n"
        return output
   

class Fc(Cell):

    _cell_constructors = {
            'Frame<float>': N2D2.FcCell_Frame_float,
            'Frame_CUDA<float>': N2D2.FcCell_Frame_CUDA_float,
    }

    def __init__(self, NbOutputs, Name=None, **config_parameters):
        Cell.__init__(self, NbOutputs, Name, **config_parameters)

        self._N2D2_object = self._cell_constructors[self._model_key](self._deepnet,
                                                            self._Name,
                                                            self._constructor_arguments['NbOutputs'])

        """Set and initialize here all complex cell members"""
        for key, value in self._config_parameters.items():
            if key is 'ActivationFunction':
                self._N2D2_object.setActivation(self._config_parameters['ActivationFunction'].N2D2())
            elif key is 'WeightsSolver':
                self._N2D2_object.setWeightsSolver(self._config_parameters['WeightsSolver'].N2D2())
            elif key is 'BiasSolver':
                self._N2D2_object.setBiasSolver(self._config_parameters['BiasSolver'].N2D2())
            elif key is 'WeightsFiller':
                self._N2D2_object.setWeightsFiller(self._config_parameters['WeightsFiller'].N2D2())
            elif key is 'BiasFiller':
                self._N2D2_object.setBiasFiller(self._config_parameters['BiasFiller'].N2D2())
            else:
                self._set_N2D2_parameter(key, value)



class Conv(Cell):

    _cell_constructors = {
        'Frame<float>': N2D2.ConvCell_Frame_float,
        'Frame_CUDA<float>': N2D2.ConvCell_Frame_CUDA_float,
    }

    def __init__(self,
                 NbOutputs,
                 KernelDims,
                 Name=None,
                 **config_parameters):
        Cell.__init__(self, NbOutputs, Name, **config_parameters)

        self._constructor_arguments.update({
            'KernelDims': KernelDims,
        })

        self._parse_optional_arguments(['SubSampleDims', 'StrideDims', 'PaddingDims', 'DilationDims'])

        self._N2D2_object = self._cell_constructors[self._model_key](self._deepnet,
                                                                     self._Name,
                                                                     self._constructor_arguments['KernelDims'],
                                                                     self._constructor_arguments['NbOutputs'],
                                                                     **self._optional_constructor_arguments)

        """Set and initialize here all complex cell members"""
        for key, value in self._config_parameters.items():
            if key is 'ActivationFunction':
                self._N2D2_object.setActivation(self._config_parameters['ActivationFunction'].N2D2())
            elif key is 'WeightsSolver':
                self._N2D2_object.setWeightsSolver(self._config_parameters['WeightsSolver'].N2D2())
            elif key is 'BiasSolver':
                self._N2D2_object.setBiasSolver(self._config_parameters['BiasSolver'].N2D2())
            elif key is 'WeightsFiller':
                self._N2D2_object.setWeightsFiller(self._config_parameters['WeightsFiller'].N2D2())
            elif key is 'BiasFiller':
                self._N2D2_object.setBiasFiller(self._config_parameters['BiasFiller'].N2D2())
            else:
                self._set_N2D2_parameter(key, value)



class ElemWise(Cell):

    _cell_constructors = {
        'Frame': N2D2.ElemWiseCell_Frame,
        'Frame_CUDA': N2D2.ElemWiseCell_Frame_CUDA,
    }

    def __init__(self, NbOutputs, Name=None, **config_parameters):
        Cell.__init__(self, NbOutputs=NbOutputs, Name=Name, **config_parameters)

        self._parse_optional_arguments(['Operation', 'Weights', 'Shifts'])
        self._N2D2_object = self._cell_constructors[self._Model](self._deepnet,
                                                self._Name,
                                                self._constructor_arguments['NbOutputs'],
                                                **self._optional_constructor_arguments)

        """Set and initialize here all complex cell members"""
        for key, value in self._config_parameters.items():
            if key is 'ActivationFunction':
                self._N2D2_object.setActivation(self._config_parameters['ActivationFunction'].N2D2())
            else:
                self._set_N2D2_parameter(key, value)



class Softmax(Cell):

    _cell_constructors = {
        'Frame<float>': N2D2.SoftmaxCell_Frame_float,
        'Frame_CUDA<float>': N2D2.SoftmaxCell_Frame_CUDA_float
    }

    def __init__(self, NbOutputs, Name=None, **config_parameters):
        Cell.__init__(self, NbOutputs=NbOutputs, Name=Name, **config_parameters)

        self._parse_optional_arguments(['WithLoss', 'GroupSize'])
        self._N2D2_object = self._cell_constructors[self._model_key](self._deepnet,
                                                                     self._Name,
                                                                     self._constructor_arguments['NbOutputs'],
                                                                     **self._optional_constructor_arguments)
        self._set_N2D2_parameters(self._config_parameters)


