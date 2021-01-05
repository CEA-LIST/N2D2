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
from n2d2.parameterizable import Parameterizable

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


class Cell(Block, Parameterizable):

    _Type = None

    def __init__(self, NbOutputs, Name):

        #Block.__init__(self, Name)
        Parameterizable.__init__(self)

        self._Name = Name
        self._blocks = self

        self._inputs = []

        self._constructor_arguments.update({
            'NbOutputs': NbOutputs,
        })

        #self._optional_constructor_arguments = {}

        #self._cell = None
        #self._model_parameters = {}

        # Keeps a trace of modified parameters for print function
        #self._modified_keys = []
                
        self._model_key = ""

    def get_output_cell(self):
        return self

    def get_input_cell(self):
        return self

    def generate_model(self, Model, DataType):
        if self._Name is None:
            raise RuntimeError("Trying to run generate_model on Cell of type " + str(type(self)) + " without name.")
        self._model_key = Model + '<' + DataType + '>'

    def add_input(self, cell):
        self._inputs.append(cell)

    def initialize(self):
        for cell in self._inputs:
            self._N2D2_object.addInput(cell.N2D2())
        self._N2D2_object.initialize()


    def __str__(self):
        output = self.get_type()+"Cell(" + self._model_key + ")"
        output += Parameterizable.__str__(self)
        return output

    def get_type(self):
        if self._Type is not None:
            return self._Type
        else:
            raise n2d2.UndefinedModelError("Abstract Cell has no type")

        # Using the string parser as in the INI files
        # NOTE: Be careful for floats (like in INI)


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
        output += "Type=" + self._Type + "\n"
        output += "NbOutputs=" + str(self._constructor_arguments['NbOutputs']) + "\n"
        return output
   

class Fc(Cell):

    _Type = 'Fc'

    _cell_constructors = {
            'Frame<float>': N2D2.FcCell_Frame_float,
            'Frame_CUDA<float>': N2D2.FcCell_Frame_CUDA_float,
    }

    """
    _weightsExportFormat = {
        'OC': N2D2.FcCell.WeightsExportFormat.OC,
        'CO': N2D2.FcCell.WeightsExportFormat.CO
    }
    """



    def __init__(self, NbOutputs, Name=None, **cell_parameters):
        Cell.__init__(self, NbOutputs=NbOutputs, Name=Name)

        """Equivalent to N2D2 class generator defaults. 
           The default objects are only abstract n2d2 objects with small memory footprint.
           ALL existing cell parameters (in N2D2) are declared here, which also permits to check 
           validity of **cell_parameters entries. For easier compatibility with INI files, we 
           use the same name convention and parameter names.
        """


        """
        These are the FcCell parameters.
        NOTE: Setting the default parameters explicitly is potentially superfluous and risks to produce 
        incoherences, but it increases readability of the library"""

        self._config_parameters.update({
            'ActivationFunction': n2d2.activation.Tanh(),
            'WeightsSolver': n2d2.solver.SGD(),
            'BiasSolver': n2d2.solver.SGD(),
            'WeightsFiller': n2d2.filler.Normal(Mean=0.0, StdDev=0.05),
            'BiasFiller': n2d2.filler.Normal(Mean=0.0, StdDev=0.05),
            'NoBias': False,
            'Normalize': False,
            'BackPropagate': True,
            'WeightsExportFormat': 'OC',
            'OutputsRemap': "",
        })

        self._set_config_parameters(cell_parameters)

        self._frame_model_parameters = {
            'DropConnect': 1.0
        }

        self._frame_CUDA_model_parameters = {}

    #TODO: Add method that initialized based on INI file section
    """
    The n2d2 FcCell type could this way serve as a wrapper for both the FcCell constructor and the
    #FcCellGenerator bindings. 
    """
    """
    def __init__(self, file_INI):
        self._N2D2_object = N2D2.FcCellGenerator(file=file_INI)
    """
         
    """
    # Optional for the moment. Has to assure coherence between n2d2 and N2D2 values
    def set_model_parameter(self, key, value):
        self._model_parameters[key] = value
        #self.cell.setParameter() # N2D2 code
    """
    
    def generate_model(self, deepnet, Model='Frame', DataType='float', **model_parameters):

        Cell.generate_model(self, Model, DataType)

        self._N2D2_object = self._cell_constructors[self._model_key](deepnet,
                                                            self._Name,
                                                            self._constructor_arguments['NbOutputs'])

        """Set and initialize here all complex cell members"""
        for key, value in self._config_parameters.items():
            if key is 'ActivationFunction':
                self._config_parameters['ActivationFunction'].generate_model(Model, DataType)
                self._N2D2_object.setActivation(self._config_parameters['ActivationFunction'].N2D2())
            elif key is 'WeightsSolver':
                self._config_parameters['WeightsSolver'].generate_model(Model, DataType)
                self._N2D2_object.setWeightsSolver(self._config_parameters['WeightsSolver'].N2D2())
            elif key is 'BiasSolver':
                self._config_parameters['BiasSolver'].generate_model(Model, DataType)
                self._N2D2_object.setBiasSolver(self._config_parameters['BiasSolver'].N2D2())
            elif key is 'WeightsFiller':
                self._config_parameters['WeightsFiller'].generate_model(DataType)
                self._N2D2_object.setWeightsFiller(self._config_parameters['WeightsFiller'].N2D2())
            elif key is 'BiasFiller':
                self._config_parameters['BiasFiller'].generate_model(DataType)
                self._N2D2_object.setBiasFiller(self._config_parameters['BiasFiller'].N2D2())
            # Not necessary when parsing strings as parameters
            #elif key is 'WeightsExportFormat':
            #    self._config_parameters['WeightsExportFormat'] = self._weightsExportFormat[value]
            #    self._N2D2_object.setParameter(key, self._config_parameters['WeightsExportFormat'])
            else:
                self._set_N2D2_parameter(key, value)

        #print(model_parameters)


        # TODO: Move to Parameterizable and delete
        if Model is 'Frame':
            for key in model_parameters:
                if key not in self._frame_model_parameters:
                    raise n2d2.UndefinedParameterError(key, self)

            for key, value in self._frame_model_parameters.items():
                if key in model_parameters:
                    self._frame_model_parameters[key] = model_parameters[key]
                    self._modified_keys.append(key)
                # Set even if default did not change
                self._set_N2D2_parameter(key, self._frame_model_parameters[key])

        if Model is 'Frame_CUDA':
            for key in model_parameters:
                if key not in self._frame_CUDA_model_parameters:
                    raise n2d2.UndefinedParameterError(key, self)

            for key, value in self._frame_CUDA_model_parameters.items():
                if key in model_parameters:
                    self._frame_CUDA_model_parameters[key] = model_parameters[key]
                    self._modified_keys.append(key)
                # Set even if default did not change
                self._set_N2D2_parameter(key, self._frame_CUDA_model_parameters[key])

        #self._model_parameters.update(model_parameters)

    # TODO: Move to Parameterizable and delete
    def __str__(self):
        output = Cell.__str__(self)
        for key, value in self._frame_model_parameters.items():
            if key in self._modified_keys:
                output += key + "=" + str(value) + ", "
        for key, value in self._frame_CUDA_model_parameters.items():
            if key in self._modified_keys:
                output += key + "=" + str(value) + ", "
        return output


class Conv(Cell):
    _Type = 'Conv'

    _cell_constructors = {
        'Frame<float>': N2D2.ConvCell_Frame_float,
        'Frame_CUDA<float>': N2D2.ConvCell_Frame_CUDA_float,
    }

    """
    _weightsExportFormat = {
        'OCHW': N2D2.ConvCell.WeightsExportFormat.OCHW,
        'HWCO': N2D2.Convell.WeightsExportFormat.HWCO
    }
    """

    def __init__(self,
                 NbOutputs,
                 KernelDims,
                 SubSampleDims=None,
                 StrideDims=None,
                 PaddingDims=None,
                 DilationDims=None,
                 Name=None,
                 **config_parameters):
        Cell.__init__(self, NbOutputs=NbOutputs, Name=Name)

        """Equivalent to N2D2 class generator defaults. 
           The default objects are only abstract n2d2 objects with small memory footprint.
           ALL existing cell parameters (in N2D2) are declared here, which also permits to check 
           validity of **config_parameters entries. For easier compatibility with INI files, we 
           use the same name convention and parameter names.
        """

        """
        These are the ConvCell parameters.
        NOTE: Setting the default parameters explicitly is potentially superfluous and risks to produce 
        incoherences, but it increases readability of the library. It also allows to check that no invalid
        parameters are passed
        """

        """
        NOTE: For the moment only list definition of KernelDims, SubSampleDims, StrideDims, PaddingDims, DilationDims
        allowed (in contrast to ConvCellGenerator in N2D2)
        """
        self._constructor_arguments.update({
            'KernelDims': KernelDims,
        })

        self._optional_constructor_arguments.update({
            'SubSampleDims': [1, 1],
            'StrideDims': [1, 1],
            'PaddingDims': [0, 0],
            'DilationDims': [1, 1],
        })

        if SubSampleDims is not None:
            self._optional_constructor_arguments['SubSampleDims'] = SubSampleDims
            self._modified_keys.append('SubSampleDims')
        if StrideDims is not None:
            self._optional_constructor_arguments['StrideDims'] = StrideDims
            self._modified_keys.append('StrideDims')
        if PaddingDims is not None:
            self._optional_constructor_arguments['PaddingDims'] = PaddingDims
            self._modified_keys.append('PaddingDims')
        if DilationDims is not None:
            self._optional_constructor_arguments['DilationDims'] = DilationDims
            self._modified_keys.append('DilationDims')

        self._config_parameters.update({
            'ActivationFunction': n2d2.activation.Tanh(),
            'WeightsSolver': n2d2.solver.SGD(),
            'BiasSolver': n2d2.solver.SGD(),
            'WeightsFiller': n2d2.filler.Normal(Mean=0.0, StdDev=0.05),
            'BiasFiller': n2d2.filler.Normal(Mean=0.0, StdDev=0.05),
            'NoBias': False,
            'BackPropagate': True,
            'WeightsExportFormat': 'OCHW',
            'WeightsExportFlip': False,
            'OutputsRemap': "",
        })

        self._set_config_parameters(config_parameters)

        self._frame_model_parameters = {}

        self._frame_CUDA_model_parameters = {}

    # TODO: Add method that initialized based on INI file section

    """
    # Optional for the moment. Has to assure coherence between n2d2 and N2D2 values
    def set_model_parameter(self, key, value):
        self._model_parameters[key] = value
        #self.cell.setParameter() # N2D2 code
    """

    # TODO: There is a lot of duplicate code between fc cell and conv cell. Implement additional parent class?
    def generate_model(self, deepnet, Model='Frame', DataType='float', **model_parameters):

        Cell.generate_model(self, Model, DataType)

        self._N2D2_object = self._cell_constructors[self._model_key](deepnet,
                                                            self._Name,
                                                            self._constructor_arguments['KernelDims'],
                                                            self._constructor_arguments['NbOutputs'],
                                                            self._optional_constructor_arguments['SubSampleDims'],
                                                            self._optional_constructor_arguments['StrideDims'],
                                                            self._optional_constructor_arguments['PaddingDims'],
                                                            self._optional_constructor_arguments['DilationDims'])

        """Set and initialize here all complex cell members"""
        for key, value in self._config_parameters.items():
            if key is 'ActivationFunction':
                self._config_parameters['ActivationFunction'].generate_model(Model, DataType)
                self._N2D2_object.setActivation(self._config_parameters['ActivationFunction'].N2D2())
            elif key is 'WeightsSolver':
                self._config_parameters['WeightsSolver'].generate_model(Model, DataType)
                self._N2D2_object.setWeightsSolver(self._config_parameters['WeightsSolver'].N2D2())
            elif key is 'BiasSolver':
                self._config_parameters['BiasSolver'].generate_model(Model, DataType)
                self._N2D2_object.setBiasSolver(self._config_parameters['BiasSolver'].N2D2())
            elif key is 'WeightsFiller':
                self._config_parameters['WeightsFiller'].generate_model(DataType)
                self._N2D2_object.setWeightsFiller(self._config_parameters['WeightsFiller'].N2D2())
            elif key is 'BiasFiller':
                self._config_parameters['BiasFiller'].generate_model(DataType)
                self._N2D2_object.setBiasFiller(self._config_parameters['BiasFiller'].N2D2())
            # Not necessary when parsing strings as parameters
            # elif key is 'WeightsExportFormat':
            #    self._config_parameters['WeightsExportFormat'] = self._weightsExportFormat[value]
            #    self._N2D2_object.setParameter(key, self._config_parameters['WeightsExportFormat'])
            else:
                self._set_N2D2_parameter(key, value)

        # print(model_parameters)

        if Model is 'Frame':
            for key in model_parameters:
                if key not in self._frame_model_parameters:
                    raise n2d2.UndefinedParameterError(key, self)

            for key, value in self._frame_model_parameters.items():
                if key in model_parameters:
                    self._frame_model_parameters[key] = model_parameters[key]
                    self._modified_keys.append(key)
                # Set even if default did not change
                self._set_N2D2_parameter(key, self._frame_model_parameters[key])

        if Model is 'Frame_CUDA':
            for key in model_parameters:
                if key not in self._frame_CUDA_model_parameters:
                    raise n2d2.UndefinedParameterError(key, self)

            for key, value in self._frame_CUDA_model_parameters.items():
                if key in model_parameters:
                    self._frame_CUDA_model_parameters[key] = model_parameters[key]
                    self._modified_keys.append(key)
                # Set even if default did not change
                self._set_N2D2_parameter(key, self._frame_CUDA_model_parameters[key])

        # self._model_parameters.update(model_parameters)


    def __str__(self):
        output = Cell.__str__(self)
        for key, value in self._frame_model_parameters.items():
            if key in self._modified_keys:
                output += key + "=" + str(value) + ", "
        for key, value in self._frame_CUDA_model_parameters.items():
            if key in self._modified_keys:
                output += key + "=" + str(value) + ", "
        return output


class ElemWise(Cell):
    _Type = 'ElemWise'

    _cell_constructors = {
        'Frame': N2D2.ElemWiseCell_Frame,
        'Frame_CUDA': N2D2.ElemWiseCell_Frame_CUDA,
    }


    _operation= {
        'Sum': N2D2.ElemWiseCell.Operation.Sum,
        'AbsSum': N2D2.ElemWiseCell.Operation.AbsSum,
        'EuclideanSum': N2D2.ElemWiseCell.Operation.EuclideanSum,
        'Prod': N2D2.ElemWiseCell.Operation.Prod,
        'Max': N2D2.ElemWiseCell.Operation.Max,
    }


    def __init__(self, NbOutputs, Operation=None, Weights=None, Shifts=None, Name=None, **config_parameters):
        Cell.__init__(self, NbOutputs=NbOutputs, Name=Name)

        """Equivalent to N2D2 class generator defaults. 
           The default objects are only abstract n2d2 objects with small memory footprint.
           ALL existing cell parameters (in N2D2) are declared here, which also permits to check 
           validity of **config_parameters entries. For easier compatibility with INI files, we 
           use the same name convention and parameter names.
        """

        """
        These are the FcCell parameters.
        NOTE: Setting the default parameters explicitly is potentially superfluous and risks to produce 
        incoherences, but it increases readability of the library"""

        self._optional_constructor_arguments.update({
            'Operation': self._operation['Sum'],
            'Weights': [],
            'Shifts': [],
        })

        if Operation is not None:
            self._optional_constructor_arguments['Operation'] = Operation
            self._modified_keys.append('Operation')
        if Weights is not None:
            self._optional_constructor_arguments['Weights'] = Weights
            self._modified_keys.append('Weights')
        if Shifts is not None:
            self._optional_constructor_arguments['Shifts'] = Shifts
            self._modified_keys.append('Shifts')

        # TODO: What is the default value in N2D2?
        self._config_parameters.update({
            'ActivationFunction': None
        })

        self._set_config_parameters(config_parameters)

        self._frame_model_parameters = {}

        self._frame_CUDA_model_parameters = {}

    # TODO: Add method that initialized based on INI file section
    """
    The n2d2 FcCell type could this way serve as a wrapper for both the FcCell constructor and the
    #FcCellGenerator bindings. 
    """
    """
    def __init__(self, file_INI):
        self._N2D2_object = N2D2.FcCellGenerator(file=file_INI)
    """

    """
    # Optional for the moment. Has to assure coherence between n2d2 and N2D2 values
    def set_model_parameter(self, key, value):
        self._model_parameters[key] = value
        #self.cell.setParameter() # N2D2 code
    """

    def generate_model(self, deepnet, Model='Frame', DataType=None, **model_parameters):

        Cell.generate_model(self, Model, DataType)

        # ElemWise cell has no DataType
        self._N2D2_object = self._cell_constructors[Model](deepnet,
                                                self._Name,
                                                self._constructor_arguments['NbOutputs'],
                                                self._optional_constructor_arguments['Operation'],
                                                self._optional_constructor_arguments['Weights'],
                                                self._optional_constructor_arguments['Shifts'])

        """Set and initialize here all complex cell members"""
        for key, value in self._config_parameters.items():
            if key is 'ActivationFunction' and self._config_parameters['ActivationFunction'] is not None:
                self._config_parameters['ActivationFunction'].generate_model(Model, DataType)
                self._N2D2_object.setActivation(self._config_parameters['ActivationFunction'].N2D2())
            else:
                self._set_N2D2_parameter(key, value)

        # print(model_parameters)

        # NOTE: ElemWiseCell currently has no model parameters
        if Model is 'Frame':
            for key in model_parameters:
                if key not in self._frame_model_parameters:
                    raise n2d2.UndefinedParameterError(key, self)

            for key, value in self._frame_model_parameters.items():
                if key in model_parameters:
                    self._frame_model_parameters[key] = model_parameters[key]
                    self._modified_keys.append(key)
                # Set even if default did not change
                self._set_N2D2_parameter(key, self._frame_model_parameters[key])

        if Model is 'Frame_CUDA':
            for key in model_parameters:
                if key not in self._frame_CUDA_model_parameters:
                    raise n2d2.UndefinedParameterError(key, self)

            for key, value in self._frame_CUDA_model_parameters.items():
                if key in model_parameters:
                    self._frame_CUDA_model_parameters[key] = model_parameters[key]
                    self._modified_keys.append(key)
                # Set even if default did not change
                self._set_N2D2_parameter(key, self._frame_CUDA_model_parameters[key])

        # self._model_parameters.update(model_parameters)

    def __str__(self):
        output = Cell.__str__(self)
        for key, value in self._frame_model_parameters.items():
            if key in self._modified_keys:
                output += key + "=" + str(value) + ", "
        for key, value in self._frame_CUDA_model_parameters.items():
            if key in self._modified_keys:
                output += key + "=" + str(value) + ", "
        return output


class Softmax(Cell):
    """Static members"""

    _Type = 'Softmax'

    _cell_constructors = {
        'Frame<float>': N2D2.SoftmaxCell_Frame_float,
        'Frame_CUDA<float>': N2D2.SoftmaxCell_Frame_CUDA_float
    }

    def __init__(self, NbOutputs, Name=None, **config_parameters):
        Cell.__init__(self, NbOutputs=NbOutputs, Name=Name)

        """
            SoftmaxCell parameters.
            Equivalent to N2D2 class generator defaults. 
            The default objects are only abstract n2d2 objects with small memory footprint.
            ALL existing cell parameters (in N2D2) are declared here, which also permits to check 
            validity of **config_parameters entries. For easier compatibility with INI files, we 
            use the same name convention and parameter names.
        """
        self._config_parameters.update({
            'WithLoss': False,
            'GroupSize': 0,
        })

        self._set_config_parameters(config_parameters)


    # TODO: Add method that initialized based on INI file section

    def generate_model(self, deepnet, Model='Frame', DataType='float', **model_parameters):

        Cell.generate_model(self, Model, DataType)

        self._N2D2_object = self._cell_constructors[self._model_key](deepnet,
                                                            self._Name,
                                                            self._constructor_arguments['NbOutputs'],
                                                            self._config_parameters['WithLoss'],
                                                            self._config_parameters['GroupSize'])


    def __str__(self):
        output = Cell.__str__(self)
        return output
