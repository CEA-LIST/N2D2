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

    def __init__(self, NbOutputs, Name, **config_parameters):

        if 'DeepNet' in config_parameters:
            self._deepnet = config_parameters.pop('DeepNet')
        else:
            self._deepnet = n2d2.global_variables.default_DeepNet

        self._Model = self._deepnet.get_model()
        self._DataType = self._deepnet.get_datatype()

        self._model_key = self._Model + '<' + self._DataType + '>'

        N2D2_Interface.__init__(self, **config_parameters)

        self._Name = Name
        self._blocks = self

        self._inputs = []

        self._constructor_arguments.update({
            'NbOutputs': NbOutputs,
        })

        self._initialized = False



    def get_output_cell(self):
        return self

    def get_input_cell(self):
        return self

    def get_type(self):
        return self._N2D2_object.getType()

    def add_input(self, cell):
        self._inputs.append(cell)

    def clear_input(self):
        self._inputs = []

    def initialize(self):
        self._N2D2_object.clearInputs()
        for cell in self._inputs:
            self._N2D2_object.addInput(cell.N2D2())
        #if not self._initialized:
        self._N2D2_object.initialize()
        self._initialized = True


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

        self._N2D2_object = self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                            self._Name,
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
                 Name=None,
                 **config_parameters):
        Cell.__init__(self, NbOutputs, Name, **config_parameters)

        self._constructor_arguments.update({
            'KernelDims': KernelDims,
        })

        self._parse_optional_arguments(['SubSampleDims', 'StrideDims', 'PaddingDims', 'DilationDims'])

        self._N2D2_object = self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                                     self._Name,
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

    def __init__(self, NbOutputs, Name=None, **config_parameters):
        Cell.__init__(self, NbOutputs=NbOutputs, Name=Name, **config_parameters)

        self._parse_optional_arguments(['Operation', 'Weights', 'Shifts'])
        self._N2D2_object = self._cell_constructors[self._Model](self._deepnet.N2D2(),
                                                self._Name,
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

    def __init__(self, NbOutputs, Name=None, **config_parameters):
        Cell.__init__(self, NbOutputs=NbOutputs, Name=Name, **config_parameters)

        self._parse_optional_arguments(['WithLoss', 'GroupSize'])
        self._N2D2_object = self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                                     self._Name,
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
    def __init__(self, NbOutputs, Name=None, **config_parameters):
        Cell.__init__(self, NbOutputs=NbOutputs, Name=Name, **config_parameters)
        # No optionnal arg ?
        self._parse_optional_arguments([])
        self._N2D2_object = self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                        self._Name,
                                                        self._constructor_arguments['NbOutputs'],
                                                        **self._optional_constructor_arguments)
        self._set_N2D2_parameters(self._config_parameters)

class Padding(Cell):

    _cell_constructors = {
        'Frame': N2D2.PaddingCell_Frame,
        'Frame_CUDA': N2D2.PaddingCell_Frame_CUDA,
    }

    def __init__(self,
                 NbOutputs,
                 topPad,
                 botPad,
                 leftPad,
                 rightPad,
                 Name=None,
                 **config_parameters):
        Cell.__init__(self, NbOutputs, Name, **config_parameters)

        self._constructor_arguments.update({
                 "topPad": topPad,
                 "botPad": botPad,
                 "leftPad": leftPad,
                 "rightPad": rightPad
        })

        self._parse_optional_arguments([])

        self._N2D2_object = self._cell_constructors[self._Model](self._deepnet.N2D2(),
                                                                     self._Name,
                                                                     self._constructor_arguments['topPad'],
                                                                     self._constructor_arguments['botPad'],
                                                                     self._constructor_arguments['leftPad'],
                                                                     self._constructor_arguments['rightPad'],
                                                                     **self._optional_constructor_arguments)

class Pool(Cell):

    _cell_constructors = {
        'Frame<float>': N2D2.PoolCell_Frame_float,
        'Frame_CUDA<float>': N2D2.PoolCell_Frame_CUDA_float,
    }

    def __init__(self,
                 NbOutputs,
                 poolDims,
                 Name=None,
                 **config_parameters):
        Cell.__init__(self, NbOutputs, Name, **config_parameters)
        self._constructor_arguments.update({
            'poolDims': poolDims,

        })
        self._parse_optional_arguments(['strideDims', 'paddingDims', 'pooling'])

        self._N2D2_object = self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                                     self._Name,
                                                                     self._constructor_arguments['poolDims'],
                                                                     self._constructor_arguments['NbOutputs'],
                                                                     **self._optional_constructor_arguments)
        """Set and initialize here all complex cell members"""
        for key, value in self._config_parameters.items():
            if key is 'Pooling':
                # TODO : Need to create a n2d2 version of Pooling
                self._N2D2_object.setPooling(value.N2D2())
            else:
                self._set_N2D2_parameter(key, value)


class LRN(Cell):
    _cell_constructors = {
        'Frame<float>': N2D2.LRNCell_Frame_float,
        'Frame_CUDA<float>': N2D2.LRNCell_Frame_CUDA_float,
    }
    def __init__(self, NbOutputs, Name=None, **config_parameters):
        Cell.__init__(self, NbOutputs=NbOutputs, Name=Name, **config_parameters)

        # No optional parameter
        self._parse_optional_arguments([])
        self._N2D2_object = self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                self._Name,
                                                self._constructor_arguments['NbOutputs'],
                                                **self._optional_constructor_arguments)

class BatchNorm(Cell):
    _cell_constructors = {
        'Frame<float>': N2D2.BatchNormCell_Frame_float,
        'Frame_CUDA<float>': N2D2.BatchNormCell_Frame_CUDA_float,
    }
    def __init__(self, NbOutputs, Name=None, **config_parameters):
        Cell.__init__(self, NbOutputs=NbOutputs, Name=Name, **config_parameters)
        # No optional parameter
        self._parse_optional_arguments([])
        self._N2D2_object = self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                self._Name,
                                                self._constructor_arguments['NbOutputs'],
                                                **self._optional_constructor_arguments)
