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



class Cell(N2D2_Interface):

    def __init__(self, NbOutputs, **config_parameters):

        if 'Name' in config_parameters:
            Name = config_parameters.pop('Name')
        else:
            Name = ''

        if 'DeepNet' in config_parameters:
            self._deepnet = config_parameters.pop('DeepNet')
        else:
            self._deepnet = n2d2.global_variables.default_DeepNet

        self._Model = self._deepnet.get_model()
        self._DataType = self._deepnet.get_datatype()

        self._model_key = self._Model + '<' + self._DataType + '>'

        N2D2_Interface.__init__(self, **config_parameters)

        self._blocks = self

        self._inputs = []

        self._constructor_arguments.update({
            'Name': Name,
            'NbOutputs': NbOutputs,
        })

        self._initialized = False

    def getName(self):
        return self._Name

    def getOutputs(self):
        return self._N2D2_object.getOutputs()

    def get_first(self):
        return self

    def get_last(self):
        return self

    def get_type(self):
        return self._N2D2_object.getType()

    def add_input(self, inputs):
        if isinstance(inputs, list):
            for cell in inputs:
                self.add_input(cell)
        elif isinstance(inputs, n2d2.deepnet.Sequence):
            self.add_input(inputs.get_last())
        elif isinstance(inputs, n2d2.deepnet.Layer):
            for cell in inputs.get_last():
                self.add_input(cell)
        elif isinstance(inputs, Cell) or isinstance(inputs, n2d2.provider.DataProvider):
            self._inputs.append(inputs)
        else:
            raise TypeError("Cannot add object of type " + str(type(inputs)))

    def get_inputs(self):
        return self._inputs

    def clear_input(self):
        self._inputs = []

    def initialize(self):
        self._N2D2_object.clearInputs()
        for cell in self._inputs:
            self._N2D2_object.addInput(cell.N2D2())
        #if not self._initialized:
        self._N2D2_object.initialize()
        self._initialized = True

    def propagate(self, inference=False):
        self._N2D2_object.propagate(inference)

    def back_propagate(self):
        self._N2D2_object.backPropagate()

    def update(self):
        self._N2D2_object.update()

    def get_name(self):
        return self._N2D2_object.getName()

    def __str__(self):
        output = self.get_type()+"Cell(" + self._model_key + ")"
        output += N2D2_Interface.__str__(self)
        if len(self.get_inputs()) > 0:
            output += "[Inputs="
            for idx, cell in enumerate(self.get_inputs()):
                if idx > 0:
                    output += ","
                output += cell.get_name()
            output += "]"
        return output


    def convert_to_INI_section(self):
        """Possible to create section without name"""
        #if self._constructor_arguments['Name'] is not None:
        output = "[" + self._constructor_arguments['Name'] + "]\n"
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

    def __init__(self, NbOutputs, **config_parameters):
        # TODO : Add description for filler and solver.
        """
        :param NbOutputs: Number of outputs of the cell.
        :type NbOutputs: int
        :param Name: Name fo the cell.
        :type Name: str
        :param ActivationFunction: Activation function used by the cell.
        :type ActivationFunction: :py:class:`n2d2.activation.Activation`, optional
        :param WeightsSolver: TODO
        :type WeightsSolver: :py:class:`n2d2.solver.Solver`, optional
        :param BiasSolver: TODO
        :type BiasSolver: :py:class:`n2d2.solver.Solver`, optional
        :param WeightsFiller: TODO
        :type WeightsFiller: :py:class:`n2d2.filler.Filler`, optional
        :param BiasFiller: TODO
        :type BiasFiller: :py:class:`n2d2.filler.Filler`, optional     
        """
        Cell.__init__(self, NbOutputs, **config_parameters)

        self._N2D2_object = self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                            self._constructor_arguments['Name'],
                                                            self._constructor_arguments['NbOutputs'])

        """Set and initialize here all complex cell members"""
        for key, value in self._config_parameters.items():
            if key is 'ActivationFunction':
                self._N2D2_object.setActivation(value.N2D2())
            elif key is 'WeightsSolver':
                self._N2D2_object.setWeightsSolver(value.N2D2())
            elif key is 'BiasSolver':
                self._N2D2_object.setBiasSolver(value.N2D2())
            elif key is 'WeightsFiller':
                self._N2D2_object.setWeightsFiller(value.N2D2())
            elif key is 'BiasFiller':
                self._N2D2_object.setBiasFiller(value.N2D2())
            else:
                self._set_N2D2_parameter(key, value)


# TODO: This is less powerful as the generator, in the sense that it does not accept several formats for the stride, conv, etc.
class Conv(Cell):

    _cell_constructors = {
        'Frame<float>': N2D2.ConvCell_Frame_float,
        'Frame_CUDA<float>': N2D2.ConvCell_Frame_CUDA_float,
        'Frame<double>': N2D2.ConvCell_Frame_double,
        'Frame_CUDA<double>': N2D2.ConvCell_Frame_CUDA_double,
    }

    def __init__(self,
                 NbOutputs,
                 KernelDims,
                 **config_parameters):
        """
        :param NbOutputs: Number of outputs of the cell.
        :type NbOutputs: int
        :param Name: Name fo the cell.
        :type Name: str
        :param KernelDims: Kernel dimension.
        :type KernelDims: list
        :param subSampleDims: TODO
        :type subSampleDims: list, optional
        :param strideDims: TODO
        :type strideDims: list, optional
        :param paddingDims: TODO
        :type paddingDims: list, optional
        :param dilationDims: TODO
        :type dilationDims: list, optional     
        """
        Cell.__init__(self, NbOutputs, **config_parameters)

        self._constructor_arguments.update({
            'KernelDims': KernelDims,
        })

        self._parse_optional_arguments(['SubSampleDims', 'StrideDims', 'PaddingDims', 'DilationDims'])

        self._N2D2_object = self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                                     self._constructor_arguments['Name'],
                                                                     self._constructor_arguments['KernelDims'],
                                                                     self._constructor_arguments['NbOutputs'],
                                                                     **self._optional_constructor_arguments)

        """Set and initialize here all complex cell members"""
        for key, value in self._config_parameters.items():
            if key is 'ActivationFunction':
                self._N2D2_object.setActivation(value.N2D2())
            elif key is 'WeightsSolver':
                self._N2D2_object.setWeightsSolver(value.N2D2())
            elif key is 'BiasSolver':
                self._N2D2_object.setBiasSolver(value.N2D2())
            elif key is 'WeightsFiller':
                self._N2D2_object.setWeightsFiller(value.N2D2())
            elif key is 'BiasFiller':
                self._N2D2_object.setBiasFiller(value.N2D2())
            else:
                self._set_N2D2_parameter(key, value)



class ElemWise(Cell):

    _cell_constructors = {
        'Frame': N2D2.ElemWiseCell_Frame,
        'Frame_CUDA': N2D2.ElemWiseCell_Frame_CUDA,
    }

    def __init__(self, NbOutputs, **config_parameters):
        Cell.__init__(self, NbOutputs=NbOutputs, **config_parameters)

        self._parse_optional_arguments(['Operation', 'Weights', 'Shifts'])
            
        operation = {
            "Sum": N2D2.ElemWiseCell.Operation.Sum,
            "AbsSum": N2D2.ElemWiseCell.Operation.AbsSum,
            "EuclideanSum": N2D2.ElemWiseCell.Operation.EuclideanSum,
            "Prod": N2D2.ElemWiseCell.Operation.Prod,
            "Max": N2D2.ElemWiseCell.Operation.Max
        }
        
        # I think the best would be to ask for a string and then convert it to the good N2D2 object
        if self._optional_constructor_arguments['operation'] in operation:
            self._optional_constructor_arguments['operation'] = operation[self._optional_constructor_arguments['operation']]
        else:
            raise n2d2.ParameterNotInListError(self._optional_constructor_arguments['operation'], [key for key in operation])

        self._N2D2_object = self._cell_constructors[self._Model](self._deepnet.N2D2(),
                                                self._constructor_arguments['Name'],
                                                self._constructor_arguments['NbOutputs'],
                                                **self._optional_constructor_arguments)

        """Set and initialize here all complex cell members"""
        for key, value in self._config_parameters.items():
            if key is 'ActivationFunction':
                self._N2D2_object.setActivation(value.N2D2())
            else:
                self._set_N2D2_parameter(key, value)



class Softmax(Cell):

    _cell_constructors = {
        'Frame<float>': N2D2.SoftmaxCell_Frame_float,
        'Frame_CUDA<float>': N2D2.SoftmaxCell_Frame_CUDA_float,
        'Frame<double>': N2D2.SoftmaxCell_Frame_double,
        'Frame_CUDA<double>': N2D2.SoftmaxCell_Frame_CUDA_double,
    }

    def __init__(self, NbOutputs, **config_parameters):
        Cell.__init__(self, NbOutputs=NbOutputs, **config_parameters)

        self._parse_optional_arguments(['WithLoss', 'GroupSize'])
        self._N2D2_object = self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                                     self._constructor_arguments['Name'],
                                                                     self._constructor_arguments['NbOutputs'],
                                                                     **self._optional_constructor_arguments)
        self._set_N2D2_parameters(self._config_parameters)

class Dropout(Cell):
    _cell_constructors = {
        'Frame<float>': N2D2.DropoutCell_Frame_float,
        'Frame_CUDA<float>': N2D2.DropoutCell_Frame_CUDA_float,
        'Frame<double>': N2D2.DropoutCell_Frame_double,
        'Frame_CUDA<double>': N2D2.DropoutCell_Frame_CUDA_double,
    }
    def __init__(self, NbOutputs, **config_parameters):
        Cell.__init__(self, NbOutputs=NbOutputs, **config_parameters)
        # No optionnal arg ?
        self._parse_optional_arguments([])
        self._N2D2_object = self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                        self._constructor_arguments['Name'],
                                                        self._constructor_arguments['NbOutputs'],
                                                        **self._optional_constructor_arguments)
        self._set_N2D2_parameters(self._config_parameters)

class Padding(Cell):

    _cell_constructors = {
        'Frame': N2D2.PaddingCell_Frame,
        'Frame_CUDA': N2D2.PaddingCell_Frame_CUDA,
    }

    def __init__(self,
                 nbOutputs,
                 topPad,
                 botPad,
                 leftPad,
                 rightPad,
                 **config_parameters):
        Cell.__init__(self, nbOutputs, **config_parameters)

        self._constructor_arguments.update({
                 "TopPad": topPad,
                 "BotPad": botPad,
                 "LeftPad": leftPad,
                 "RightPad": rightPad
        })

        self._parse_optional_arguments([])

        self._N2D2_object = self._cell_constructors[self._Model](self._deepnet.N2D2(),
                                                                     self._constructor_arguments['Name'],
                                                                     self._constructor_arguments['NbOutputs'],
                                                                     self._constructor_arguments['TopPad'],
                                                                     self._constructor_arguments['BotPad'],
                                                                     self._constructor_arguments['LeftPad'],
                                                                     self._constructor_arguments['RightPad'],
                                                                     **self._optional_constructor_arguments)
        self._set_N2D2_parameters(self._config_parameters)


class Pool(Cell):

    _cell_constructors = {
        'Frame<float>': N2D2.PoolCell_Frame_float,
        'Frame_CUDA<float>': N2D2.PoolCell_Frame_CUDA_float,
    }
    
    def __init__(self,
                 NbOutputs,
                 PoolDims,
                 **config_parameters):

        Cell.__init__(self, NbOutputs, **config_parameters)

        self._constructor_arguments.update({
            'PoolDims': PoolDims,
        })
        pooling = {
            "Average": N2D2.PoolCell.Pooling.Average,
            "Max": N2D2.PoolCell.Pooling.Max
        }

        # Note: Removed Pooling
        self._parse_optional_arguments(['StrideDims', 'PaddingDims', 'Pooling'])

        # I think the best would be to ask for a string and then convert it to the good N2D2 object rather than creatin 
        if self._optional_constructor_arguments['pooling'] in pooling:
            self._optional_constructor_arguments['pooling'] = pooling[self._optional_constructor_arguments['pooling']]
        else:
            raise n2d2.ParameterNotInListError(self._optional_constructor_arguments['pooling'], [key for key in pooling])


        self._N2D2_object = self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                                     self._constructor_arguments['Name'],
                                                                     self._constructor_arguments['PoolDims'],
                                                                     self._constructor_arguments['NbOutputs'],
                                                                     **self._optional_constructor_arguments)

        self._set_N2D2_parameters(self._config_parameters)



class LRN(Cell):
    _cell_constructors = {
        'Frame<float>': N2D2.LRNCell_Frame_float,
        'Frame_CUDA<float>': N2D2.LRNCell_Frame_CUDA_float,
    }


    def __init__(self, NbOutputs, **config_parameters):
        Cell.__init__(self, NbOutputs=NbOutputs, **config_parameters)

        # No optional parameter
        self._parse_optional_arguments([])
        self._N2D2_object = self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                self._constructor_arguments['Name'],
                                                self._constructor_arguments['NbOutputs'],
                                                **self._optional_constructor_arguments)

class BatchNorm(Cell):
    _cell_constructors = {
        'Frame<float>': N2D2.BatchNormCell_Frame_float,
        'Frame_CUDA<float>': N2D2.BatchNormCell_Frame_CUDA_float,
    }
    def __init__(self, NbOutputs, **config_parameters):
        Cell.__init__(self, NbOutputs=NbOutputs, **config_parameters)
        # No optional parameter
        self._parse_optional_arguments([])
        self._N2D2_object = self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                self._constructor_arguments['Name'],
                                                self._constructor_arguments['NbOutputs'],
                                                **self._optional_constructor_arguments)
