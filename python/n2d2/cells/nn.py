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
import n2d2.filler
from n2d2.n2d2_interface import N2D2_Interface
from n2d2.cells.cell import Cell
from abc import ABC, abstractmethod

_cell_parameters = {
    "deep_net": "DeepNet", 
    "name": "Name", 
    "inputs_dims": "InputsDims", 
    "outputs_dims": "OutputsDims", 
    "mapping": "Mapping",
    "quantized_nb_bits": "QuantizedNbits", 
    "id_cnt": "IdCnt", 
    "group_map": "GroupMap", 
    "group_map_initialized": "GroupMapInitialized",
    "from_arguments": "", # Pure n2d2
    
}
_cell_frame_parameters = {
    "inputs": "Inputs",
    "outputs": "Outputs",
    "diff_inputs": "DiffInputs",
    "diff_outputs": "DiffOutputs",
    "targets": "Targets",
    "nb_target_outputs": "NbTargetOutputs",
    "loss_mem": "LossMem",
    "activation_desc": "ActivationDesc",
    "keep_in_sync": "KeepInSync",
    "activation": "activation",
    "devices": "Devices",
}
# Cell_frame_parameter contains the parameters from cell_parameter
_cell_frame_parameters.update(_cell_parameters) 


class Datatyped(ABC):

    @abstractmethod
    def __init__(self, datatype=None):
        if datatype:
            self._datatype = datatype
        else:
            self._datatype = n2d2.global_variables.default_datatype


class NeuralNetworkCell(N2D2_Interface, Cell, ABC):

    @abstractmethod
    def __init__(self,  **config_parameters):
        
        # TODO : We may want to check activation parameter type here since every cell_frame can have an activation ?
        # else we need to add this check on every classes
        if "activation" in config_parameters:
            if not isinstance(config_parameters["activation"], n2d2.activation.ActivationFunction):
                raise n2d2.error_handler.WrongInputType("activation", str(type(config_parameters["activation"])), [str(n2d2.activation.ActivationFunction)])
        
        if 'name' in config_parameters:
            name = config_parameters.pop('name')
        else:
            name = None # Set to None so that it can be created in Cell.__init__ 

        Cell.__init__(self, name)

        self._inputs = []

        if 'model' in config_parameters:
            self._model = config_parameters.pop('model')
        else:
            self._model = n2d2.global_variables.default_model

        self._model_key = self._model

        if isinstance(self, Datatyped):
            if 'datatype' in config_parameters:
                Datatyped.__init__(self, config_parameters.pop('datatype'))
            else:
                Datatyped.__init__(self)
            self._model_key += '<' + self._datatype + '>'
        else:
            if 'datatype' in config_parameters:
                raise RuntimeError("'datatype' argument received in un-datatyped cell of type " + str(type(self)))

        N2D2_Interface.__init__(self, **config_parameters)

        self._deepnet = None
        self._inference = False

        

    def learn(self):
        self._inference = False

    def test(self):
        self._inference = True

    def _infer_deepnet(self, inputs):
        if isinstance(inputs, n2d2.tensor.Interface) or isinstance(inputs, n2d2.tensor.Tensor):
            deepnet = inputs.get_deepnet()
        else:
            raise TypeError("Object of type " + str(type(inputs)) + " cannot implicitly provide a deepNet to cells.")
        return deepnet


    def dims(self):
        return self.get_outputs().dims()

    def get_outputs(self):
        return n2d2.Tensor.from_N2D2(self._N2D2_object.getOutputs())._set_cell(self)

    def get_deepnet(self):
        return self._deepnet

    def set_deepnet(self, deepnet):
        self._deepnet = deepnet

    #def clear_data_tensors(self):
    #    self._N2D2_object.clearOutputTensors()

    def clear_input_tensors(self):
        self._inputs = []
        self._N2D2_object.clearInputTensors()


    # TODO: What exactly should be checked? Input identity and/or input dimensions? At the moment we only check dimensions
    # This means a new NeuralNetworkCell with same dimensions is will not be connected!
    def _check_tensor(self, inputs):
        if isinstance(inputs.cell, n2d2.cells.nn.NeuralNetworkCell) or isinstance(inputs.cell, n2d2.provider.Provider):
            input_dims = inputs.cell.dims()
            if not self.dims(): # If not initialized
                return True
        else:
            raise TypeError("Invalid inputs object of type " + str(type(inputs.cell)))

        if inputs.get_deepnet() is not self.get_deepnet():
            raise RuntimeError("The deepnet of the input doesn't match with the deepnet of the cell")
        

        # if self._N2D2_object.getInputsDims()+ [self.dims()[3]]: # If input dimesions changed
        #     raise RuntimeError("NeuralNetworkCell '" + self.get_name() + "' was called with input of dim " + str(inputs.dims())
        #                         + ", but cells input size is " + str(self._N2D2_object.getInputsDims()+ [self.dims()[3]]) +
        #                         ". Inputs dimensions cannot change after first call.")
        return False 


    def add_input(self, inputs):
        # TODO : Some cells like Pool don't have a defined number of channels so I try this to catch them
        # Is it good to keep it this way ?
        have_a_defined_input_size = (self.N2D2().getInputsDims() != [0] and self.N2D2().getInputsDims() != [])
        initialized = self.dims() == True 
        # TODO :this test doesn't pass for Fc cells if it is not initialized.
        # The get_nb_channels() returns dimX * dimY * dimZ if not initialized and then just dimZ.
        # Maybe we want to do an other test if the cell is not initialized (testing if the weights correspond to the inputs)
        #print(initialized)
        if have_a_defined_input_size and inputs.dimZ() != self.get_nb_channels() and initialized:
            raise ValueError("NeuralNetworkCell '" + self.get_name() + "' received a tensor with " + str(inputs.dimZ()) +
            " channels, was expecting : " + str(self.get_nb_channels()))
        
        if isinstance(inputs, n2d2.tensor.Interface):
            inputs = inputs.get_tensors()
        elif isinstance(inputs, n2d2.tensor.Tensor):
            inputs = [inputs]
        else:
            raise TypeError("Cannot add object of type " + str(type(inputs)))

        self.clear_input_tensors()
        initialize = False

        parents = []
        for ipt in inputs:
            if self._check_tensor(ipt):
                initialize = True
            cell = ipt.cell
            self._link_N2D2_input(cell)

            if not isinstance(cell, n2d2.provider.Provider):
                parents.append(cell.N2D2())
            self._inputs.append(cell.get_name())

        self._deepnet.N2D2().addCell(self._N2D2_object, parents)
        if initialize:
            self._N2D2_object.initializeDataDependent()

    """
    Links N2D2 cells 
    """
    def _link_N2D2_input(self, inputs):
        self._N2D2_object.linkInput(inputs.N2D2())

    def _add_to_graph(self, inputs):
        self.add_input(inputs)
        self._deepnet.add_to_current_group(self)

    def set_activation(self, activation):
        print("Note: Replacing potentially existing activation in cells: " + self.get_name())
        self._config_parameters['activation'] = activation
        self._N2D2_object.setActivation(self._config_parameters['activation'].N2D2())

    def get_activation(self):
        return self._config_parameters['activation']

    def get_inputs(self):
        return self._inputs

    def clear_input(self):
        self._inputs = []
        self._N2D2_object.clearInputs()

    def update(self):
        self._N2D2_object.update()

    def import_free_parameters(self, dir_name, ignoreNotExists=False):
        if self._N2D2_object:
            filename = dir_name + "/" + self.get_name() + ".syntxt"
            print("import " + filename)
            self._N2D2_object.importFreeParameters(filename, ignoreNotExists)
            self._N2D2_object.importActivationParameters(dir_name, ignoreNotExists)

    """
    def import_activation_parameters(self, filename, **kwargs):
        print("import " + filename)
        self._N2D2_object.importActivationParameters(filename, **kwargs)
    """

    def get_nb_outputs(self):
        return self._N2D2_object.getNbOutputs()

    def get_nb_channels(self):
        return self._N2D2_object.getNbChannels()

    def _sync_inputs_and_parents(self):
        parents = self._deepnet.N2D2().getParentCells(self.get_name())
        # Necessary because N2D2 returns [None] if no parents
        # TODO: Sometimes parents contains [None], sometimes []. Why?
        for idx, ipt in enumerate(parents):
            if ipt is not None:
                self._inputs.append(parents[idx].getName())
        self._deepnet.add_to_current_group(self)

    def __str__(self):
        output = "\'" + self.get_name() + "\' " + self.get_type() + "(" + self._model_key + ")"
        output += N2D2_Interface.__str__(self)
        if len(self.get_inputs()) > 0:
            output += "(["
            for idx, name in enumerate(self.get_inputs()):
                if idx > 0:
                    output += ", "
                output += "'" + name + "'"
            output += "])"
        else:
            output += ""
        return output

    def __call__(self, inputs):
        """
        Do the common check on the inputs and infer the deepNet from the inputs.
        """
        if not (isinstance(inputs, n2d2.Tensor) or isinstance(inputs, n2d2.Interface)):
            raise TypeError(self.get_name() + " received an inputs of type " + str(type(inputs)) + ", inputs should be of type n2d2.Tensor instead.")
        self._deepnet = self._infer_deepnet(inputs)


class Fc(NeuralNetworkCell, Datatyped):
    """
    Fully connected layer.
    """

    _cell_constructors = {
        'Frame<double>': N2D2.FcCell_Frame_double,
        'Frame<float>': N2D2.FcCell_Frame_float,
        'Frame_CUDA<float>': N2D2.FcCell_Frame_CUDA_float,
    }

    _parameters = {
        "no_bias":"NoBias", 
        "normalize": "Normalize",
        "back_propagate":"BackPropagate",
        "weights_export_format":"WeightsExportFormat",
        "outputs_remap":"OutputsRemap",
        "weights_filler":"WeightsFiller",  
        "bias_filler":"BiasFiller",  
        "weights_solver":"WeightsSolver",
        "bias_solver":"BiasSolver",
        "quantizer":"Quantizer",
        "drop_connect": "DropConnect"
    }  
    _parameters.update(_cell_frame_parameters)

    _convention_converter= n2d2.ConventionConverter(_parameters)


    def __init__(self, nb_inputs, nb_outputs, nb_input_cells=1, from_arguments=True, **config_parameters):
        """
        :param nb_inputs: Number of inputs of the cells.
        :type nb_inputs: int
        :param nb_outputs: Number of outputs of the cells.
        :type nb_outputs: int
        :param from_arguments: If False, allow you to create cells with mandatory arguments set as None, default=True
        :type  from_arguments: bool, optional
        :param name: Name fo the cells.
        :type name: str, optional
        :param activation: Activation function, default= :py:class:`n2d2.activation.Tanh`
        :type activation: :py:class:`n2d2.activation.ActivationFunction`, optional
        :param weights_solver: Solver for weights, default=:py:class:`n2d2.solver.SGD`
        :type weights_solver: :py:class:`n2d2.solver.Solver`, optional
        :param bias_solver: Solver for biases, default= :py:class:`n2d2.filler.Normal`
        :type bias_solver: :py:class:`n2d2.solver.Solver`, optional
        :param weights_filler: Weights initial values filler, default= :py:class:`n2d2.filler.Normal`
        :type weights_filler: :py:class:`n2d2.filler.Filler`, optional
        :param bias_filler: Biases initial values filler, default= :py:class:`n2d2.filler.Normal`
        :type bias_filler: :py:class:`n2d2.filler.Filler`, optional
        :param mapping: Mapping
        :type mapping: :py:class:`n2d2.tensor.Tensor`, optional
        :param no_bias: If True, don’t use bias, default=False
        :type no_bias: bool, optional
        """

        if not from_arguments and (nb_inputs is not None or nb_outputs is not None or len(config_parameters) > 0):
            raise RuntimeError("from_arguments = True but not None constructor arguments")
        if from_arguments:
            self._create_from_arguments(nb_inputs, nb_outputs, nb_input_cells, **config_parameters)


    def _create_from_arguments(self, nb_inputs, nb_outputs, nb_input_cells, **config_parameters):
        NeuralNetworkCell.__init__(self, **config_parameters)

        if not isinstance(nb_inputs, int):
            raise n2d2.error_handler.WrongInputType("nb_inputs", str(type(nb_inputs)), ["int"])
        if not isinstance(nb_outputs, int):
            raise n2d2.error_handler.WrongInputType("nb_outputs", str(type(nb_outputs)), ["int"])

        self._constructor_arguments.update({
            'nb_inputs': nb_inputs,
            'nb_outputs': nb_outputs,
        })

        self._set_N2D2_object(self._cell_constructors[self._model_key](N2D2.DeepNet(n2d2.global_variables.default_net),
                                                                     self.get_name(),
                                                                     self._constructor_arguments['nb_outputs']))
        # Set connection and mapping parameters

        # TODO: Are these parameters actually used outside of the generator?
        #for key in self._config_parameters:
        #    if key is 'inputOffsetX':
        #        self._connection_parameters['x0'] = self._config_parameters.pop('inputOffsetX')
        #    elif key is 'inputOffsetY':
        #        self._connection_parameters['y0'] = self._config_parameters.pop('inputOffsetY')
        #    elif key is 'inputWidth':
        #        self._connection_parameters['width'] = self._config_parameters.pop('inputWidth')
        #    elif key is 'inputHeight':
        #        self._connection_parameters['height'] = self._config_parameters.pop('inputHeight')

        if 'activation' not in self._config_parameters:
            self._config_parameters['activation'] = \
                n2d2.converter.from_N2D2_object(self._N2D2_object.getActivation())
        if 'weights_solver' not in self._config_parameters:
            self._config_parameters['weights_solver'] = \
                n2d2.converter.from_N2D2_object(self._N2D2_object.getWeightsSolver())
        if 'bias_solver' not in self._config_parameters:
            self._config_parameters['bias_solver'] = \
                n2d2.converter.from_N2D2_object(self._N2D2_object.getBiasSolver())
        if 'weights_filler' not in self._config_parameters:
            self._config_parameters['weights_filler'] = \
                n2d2.converter.from_N2D2_object(self._N2D2_object.getWeightsFiller())
        if 'bias_filler' not in self._config_parameters:
            self._config_parameters['bias_filler'] = \
                n2d2.converter.from_N2D2_object(self._N2D2_object.getBiasFiller())

        # Set and initialize here all complex cells members
        for key, value in self._config_parameters.items():
            if key is 'activation':
                    if value: # TODO : @Johannes why is there a check here but not on the others 
                        self._N2D2_object.setActivation(value.N2D2())
            elif key is 'weights_solver':
                if isinstance(value, n2d2.solver.Solver):
                    self._N2D2_object.setWeightsSolver(value.N2D2())
                else:
                    raise n2d2.error_handler.WrongInputType("weights_solver", str(type(value)), [str(n2d2.solver.Solver)])
            elif key is 'bias_solver':
                if isinstance(value, n2d2.solver.Solver):
                    self._N2D2_object.setBiasSolver(value.N2D2())
                else:
                    raise n2d2.error_handler.WrongInputType("bias_solver", str(type(value)), [str(n2d2.solver.Solver)])
            elif key is 'weights_filler':
                if isinstance(value, n2d2.filler.Filler):
                    self._N2D2_object.setWeightsFiller(value.N2D2())
                else:
                    raise n2d2.error_handler.WrongInputType("weights_filler", str(type(value)), [str(n2d2.filler.Filler)])
            elif key is 'bias_filler':
                if isinstance(value, n2d2.filler.Filler):
                    self._N2D2_object.setBiasFiller(value.N2D2())
                else:
                    raise n2d2.error_handler.WrongInputType("bias_filler", str(type(value)), [str(n2d2.filler.Filler)])
            elif key is 'quantizer':
                if isinstance(value, n2d2.quantizer.Quantizer):
                    self._N2D2_object.setQuantizer(value.N2D2())
                else:
                    raise n2d2.error_handler.WrongInputType("quantizer", str(type(value)), [str(n2d2.quantizer.Quantizer)])
                
            else:
                self._set_N2D2_parameter(self.python_to_n2d2_convention(key), value)

        self._N2D2_object.initializeParameters(nb_inputs, nb_input_cells)


    @classmethod
    def create_from_N2D2_object(cls, N2D2_object, n2d2_deepnet=None):

        n2d2_cell = cls(None, None, from_arguments=False)

        NeuralNetworkCell.__init__(n2d2_cell,
                                   name=N2D2_object.getName(),
                                   **cls.load_N2D2_parameters(N2D2_object))

        n2d2_cell._constructor_arguments.update({
            'nb_inputs': N2D2_object.getInputsSize(),
            'nb_outputs': N2D2_object.getNbOutputs(),
        })

        n2d2_cell._set_N2D2_object(N2D2_object)

        n2d2_cell._config_parameters['activation'] = \
            n2d2.converter.from_N2D2_object(n2d2_cell._N2D2_object.getActivation())
        n2d2_cell._config_parameters['weights_solver'] = \
            n2d2.converter.from_N2D2_object(n2d2_cell._N2D2_object.getWeightsSolver())
        n2d2_cell._config_parameters['bias_solver'] = \
            n2d2.converter.from_N2D2_object(n2d2_cell._N2D2_object.getBiasSolver())
        n2d2_cell._config_parameters['weights_filler'] = \
            n2d2.converter.from_N2D2_object(n2d2_cell._N2D2_object.getWeightsFiller())
        n2d2_cell._config_parameters['bias_filler'] = \
            n2d2.converter.from_N2D2_object(n2d2_cell._N2D2_object.getBiasFiller())
        quantizer = n2d2_cell._N2D2_object.getQuantizer()
        if quantizer:
            n2d2_cell._config_parameters['quantizer'] = \
                n2d2.converter.from_N2D2_object(quantizer)

        if n2d2_deepnet is not None:
            n2d2_cell._deepnet = n2d2_deepnet
            n2d2_cell._sync_inputs_and_parents()
        else:
            n2d2_cell._deepnet = None
            n2d2_cell._N2D2_object.clearInputs()

        return n2d2_cell


    def __call__(self, inputs):
        super().__call__(inputs)

        self._add_to_graph(inputs)

        self._N2D2_object.propagate(self._inference)

        return self.get_outputs()


    def set_weight(self, output_index, channel_index, value):
        """
        :param output_index: 
        :type output_index:
        :param channel_index:
        :type channel_index:
        :param value:
        :type value: :py:class:`n2d2.tensor.Tensor`
        """
        if channel_index >= self.N2D2().getNbChannels():
            raise ValueError("Channel index : " + str(channel_index) + " must be < " + str(self.N2D2().getNbChannels()) +")")
        if output_index >= self.N2D2().getNbOutputs():
            raise ValueError("Output index : " + str(output_index) + " must be < " + str(self.N2D2().getNbOutputs()) +")")
        self.N2D2().setWeight(output_index, channel_index, value.N2D2())

    def get_weight(self, output_index, channel_index):
        """
        :param output_index: 
        :type output_index: int
        :param channel_index:
        :type channel_index: int
        """
        if channel_index >= self.N2D2().getNbChannels():
            raise ValueError("Channel index : " + str(channel_index) + " must be < " + str(self.N2D2().getNbChannels()) +")")
        if output_index >= self.N2D2().getNbOutputs():
            raise ValueError("Output index : " + str(output_index) + " must be < " + str(self.N2D2().getNbOutputs()) +")")
        tensor = N2D2.Tensor_float([])
        self.N2D2().getWeight(output_index, channel_index, tensor)
        return n2d2.Tensor.from_N2D2(tensor)

    def get_weights(self):
        """
        :return: list of weights
        :rtype: list
        """
        weights = []
        tensor = N2D2.Tensor_float([])
        for o in range(self.N2D2().getNbOutputs()):
            chan = []
            for c in range(self.N2D2().getNbChannels()):
                self.N2D2().getWeight(o, c, tensor)
                chan.append(n2d2.Tensor.from_N2D2(tensor))
            weights.append(chan)
        return weights

    def set_bias(self, output_index, value):
        """
        :param output_index: 
        :type output_index: int
        :param value: 
        :type value: :py:class:`n2d2.Tensor`
        """
        if output_index >= self.N2D2().getNbOutputs():
            raise ValueError("Output index : " + str(output_index) + " must be < " + str(self.N2D2().getNbOutputs()) +")")
        self.N2D2().setBias(output_index, value.N2D2())

    def get_bias(self, output_index):
        """
        :param output_index: 
        :type output_index: int
        """
        if output_index >= self.N2D2().getNbOutputs():
            raise ValueError("Output index : " + str(output_index) + " must be < " + str(self.N2D2().getNbOutputs()) +")")
        tensor = N2D2.Tensor_float([])
        self.N2D2().getBias(output_index, tensor)
        return n2d2.Tensor.from_N2D2(tensor)
        
    def get_biases(self):
        """
        :return: list of biases
        :rtype: list
        """
        biases = []
        for output_index in range(self.N2D2().getNbOutputs()):
            tensor = N2D2.Tensor_float([])
            self.N2D2().getBias(output_index, tensor)
            biases.append(n2d2.Tensor.from_N2D2(tensor))
        return biases

    def set_quantizer(self, quantizer):
        if 'quantizer' in self._config_parameters:
            raise RuntimeError("Quantizer already exists in cell '" + self.get_name() + "'")
        else:
            self._config_parameters['quantizer'] = quantizer
            self._N2D2_object.setQuantizer(self._config_parameters['quantizer'].N2D2())
            self._N2D2_object.initializeWeightQuantizer()

    def get_quantizer(self):
        if 'quantizer' in self._config_parameters:
            return self._config_parameters['quantizer']
        else:
            raise RuntimeError("No Quantizer in cell '" + self.get_name() + "'")


    def get_bias_solver(self):
        return self._config_parameters['bias_solver']

    def get_weights_solver(self):
        return self._config_parameters['weights_solver']

    def set_weights_solver(self, solver):
        self._config_parameters['weights_solver'] = solver
        self._N2D2_object.setWeightsSolver(self._config_parameters['weights_solver'].N2D2())
    
    def set_bias_solver(self, solver):
        self._config_parameters['bias_solver'] = solver
        self._N2D2_object.setBiasSolver(self._config_parameters['bias_solver'].N2D2())

    def set_solver(self, solver):
        self._config_parameters['bias_solver'] = solver
        self._N2D2_object.setBiasSolver(self._config_parameters['bias_solver'].N2D2())
        self._config_parameters['weights_solver'] = solver.copy()
        self._N2D2_object.setWeightsSolver(self._config_parameters['weights_solver'].N2D2())




class Conv(NeuralNetworkCell, Datatyped):
    """
    Convolutional layer.
    """


    _cell_constructors = {
        'Frame<float>': N2D2.ConvCell_Frame_float,
        'Frame_CUDA<float>': N2D2.ConvCell_Frame_CUDA_float,
        'Frame<double>': N2D2.ConvCell_Frame_double,
        'Frame_CUDA<double>': N2D2.ConvCell_Frame_CUDA_double,
    }
    
    _parameters = {
        "no_bias":"NoBias", 
        "back_propagate":"BackPropagate",
        "weights_export_format":"WeightsExportFormat",
        "weights_export_flip":"WeightsExportFlip",
        "outputs_remap":"OutputsRemap",
        "kernel_dims":"kernelDims",
        "sub_sample_dims":"subSampleDims",
        "stride_dims":"strideDims",
        "padding_dims":"paddingDims",
        "dilation_dims":"dilationDims",  
        "ext_padding_dims":"ExtPaddingDims",   
        "dilation_dims":"dilationDims",
        "ext_padding_dims":"ExtPaddingDims", 
        "weights_filler":"WeightsFiller",  
        "bias_filler":"BiasFiller",  
        "weights_solver":"WeightsSolver",
        "bias_solver":"BiasSolver",
        "quantizer":"Quantizer",
    }  
    _parameters.update(_cell_frame_parameters)

    _convention_converter= n2d2.ConventionConverter(_parameters)


    def __init__(self,
                 nb_inputs,
                 nb_outputs,
                 kernel_dims,
                 nb_input_cells=1,
                 from_arguments=True,
                 **config_parameters):
        """
        :param nb_inputs: Number of inputs of the cells.
        :type nb_inputs: int
        :param nb_outputs: Number of output channels
        :type nb_outputs: int
        :param kernel_dims: Kernel dimension.
        :type kernel_dims: list
        :param from_arguments: If False, allow you to create cells with mandatory arguments set as None, default=False
        :type  from_arguments: bool, optional
        :param name: Name for the cells.
        :type name: str
        :param sub_sample_dims: Dimension of the subsampling factor of the output feature maps
        :type sub_sample_dims: list, optional
        :param stride_dims: Dimension of the stride of the kernel.
        :type stride_dims: list, optional
        :param padding_dims: Dimensions of the padding.
        :type padding_dims: list, optional
        :param dilation_dims: Dimensions of the dilation of the kernels 
        :type dilation_dims: list, optional
        :param activation: Activation function, default= :py:class:`n2d2.activation.Tanh`
        :type activation: :py:class:`n2d2.activation.ActivationFunction`, optional
        :param mapping: Mapping
        :type mapping: :py:class:`n2d2.tensor.Tensor`
        :param weights_filler: Weights initial values filler, default= :py:class:`n2d2.filler.Normal`
        :type weights_filler: :py:class:`n2d2.filler.Filler`, optional
        :param bias_filler: Biases initial values filler, default= :py:class:`n2d2.filler.Normal`
        :type bias_filler: :py:class:`n2d2.filler.Filler`, optional
        :param weights_solver: Solver for weights
        :type weights_solver: :py:class:`n2d2.solver.Solver`, optional
        :param bias_solver: Solver for biases
        :type bias_solver: :py:class:`n2d2.solver.Solver`, optional
        :param no_bias: If True, don’t use bias, default=False
        :type no_bias: bool, optional
        :param weights_export_flip: If true, import/export flipped kernels, default=False
        :type weights_export_flip: bool, optional
        :param back_propagate: If true, enable backpropagation, default=True
        :type back_propagate: bool, optional
        
        """

        if not from_arguments and (nb_inputs is not None or nb_outputs is not None or kernel_dims is not None or len(config_parameters) > 0):
            raise RuntimeError("from_arguments = True but not None constructor arguments")
        if from_arguments:
            self._create_from_arguments(nb_inputs, nb_outputs, kernel_dims, nb_input_cells, **config_parameters)


    def _create_from_arguments(self, nb_inputs, nb_outputs, kernel_dims, nb_input_cells, **config_parameters):
        if not isinstance(nb_inputs, int):
            raise n2d2.error_handler.WrongInputType("nb_inputs", str(type(nb_inputs)), ["int"])
        if not isinstance(nb_outputs, int):
            raise n2d2.error_handler.WrongInputType("nb_outputs", str(type(nb_outputs)), ["int"])
        if not isinstance(kernel_dims, list): # TODO : check the type of the elements ?
            raise n2d2.error_handler.WrongInputType("kernel_dims", str(type(kernel_dims)), ["list"])

        NeuralNetworkCell.__init__(self, **config_parameters)

        

        self._constructor_arguments.update({
            'nb_inputs': nb_inputs,
            'nb_outputs': nb_outputs,
            'kernel_dims': kernel_dims,
        })

        self._parse_optional_arguments(['sub_sample_dims', 'stride_dims', 'padding_dims', 'dilation_dims'])

        self._set_N2D2_object(self._cell_constructors[self._model_key](N2D2.DeepNet(n2d2.global_variables.default_net),
                                                                     self.get_name(),
                                                                     self._constructor_arguments['kernel_dims'],
                                                                     self._constructor_arguments['nb_outputs'],
                                                                     **self.n2d2_function_argument_parser(self._optional_constructor_arguments)))

        # TODO: Add Kernel section of generator

        if 'activation' not in self._config_parameters:
            self._config_parameters['activation'] = \
                n2d2.converter.from_N2D2_object(self._N2D2_object.getActivation())
        if 'weights_solver' not in self._config_parameters:
            self._config_parameters['weights_solver'] = \
                n2d2.converter.from_N2D2_object(self._N2D2_object.getWeightsSolver())
        if 'bias_solver' not in self._config_parameters:
            self._config_parameters['bias_solver'] = \
                n2d2.converter.from_N2D2_object(self._N2D2_object.getBiasSolver())
        if 'weights_filler' not in self._config_parameters:
            self._config_parameters['weights_filler'] = \
                n2d2.converter.from_N2D2_object(self._N2D2_object.getWeightsFiller())
        if 'bias_filler' not in self._config_parameters:
            self._config_parameters['bias_filler'] = \
                n2d2.converter.from_N2D2_object(self._N2D2_object.getBiasFiller())

        """Set and initialize here all complex cells members"""
        for key, value in self._config_parameters.items():
            if key is 'activation':
                if value: # TODO : @Johannes why is there a check here but not on the others 
                    self._N2D2_object.setActivation(value.N2D2())
            elif key is 'weights_solver':
                if isinstance(value, n2d2.solver.Solver):
                    self._N2D2_object.setWeightsSolver(value.N2D2())
                else:
                    raise n2d2.error_handler.WrongInputType("weights_solver", str(type(value)), [str(n2d2.solver.Solver)])
            elif key is 'bias_solver':
                if isinstance(value, n2d2.solver.Solver):
                    self._N2D2_object.setBiasSolver(value.N2D2())
                else:
                    raise n2d2.error_handler.WrongInputType("bias_solver", str(type(value)), [str(n2d2.solver.Solver)])
            elif key is 'weights_filler':
                if isinstance(value, n2d2.filler.Filler):
                    self._N2D2_object.setWeightsFiller(value.N2D2())
                else:
                    raise n2d2.error_handler.WrongInputType("weights_filler", str(type(value)), [str(n2d2.filler.Filler)])
            elif key is 'bias_filler':
                if isinstance(value, n2d2.filler.Filler):
                    self._N2D2_object.setBiasFiller(value.N2D2())
                else:
                    raise n2d2.error_handler.WrongInputType("bias_filler", str(type(value)), [str(n2d2.filler.Filler)])
            elif key is 'quantizer':
                if isinstance(value, n2d2.quantizer.Quantizer):
                    self._N2D2_object.setQuantizer(value.N2D2())
                else:
                    raise n2d2.error_handler.WrongInputType("quantizer", str(type(value)), [str(n2d2.quantizer.Quantizer)])
                self._N2D2_object.setQuantizer(value.N2D2())
            elif key is 'mapping':
                if isinstance(value, n2d2.Tensor):
                    self._N2D2_object.setMapping(value.N2D2())
                else:
                    raise n2d2.error_handler.WrongInputType('mapping', type(value), [str(type(n2d2.Tensor))])
            else:
                self._set_N2D2_parameter(self.python_to_n2d2_convention(key), value)

        self._N2D2_object.initializeParameters(nb_inputs, nb_input_cells)



    @classmethod
    def create_from_N2D2_object(cls, N2D2_object,  n2d2_deepnet=None):

        n2d2_cell = cls(None, None, None, from_arguments=False)

        NeuralNetworkCell.__init__(n2d2_cell,
                                   name=N2D2_object.getName(),
                                   **cls.load_N2D2_parameters(N2D2_object))

        n2d2_cell._set_N2D2_object(N2D2_object)

        n2d2_cell._constructor_arguments.update({
            'nb_inputs':  n2d2_cell._N2D2_object.getNbChannels(),
            'nb_outputs':  n2d2_cell._N2D2_object.getNbOutputs(),
            'kernel_dims': [n2d2_cell._N2D2_object.getKernelWidth(), n2d2_cell._N2D2_object.getKernelHeight()]
        })

        n2d2_cell._optional_constructor_arguments['sub_sample_dims'] = [n2d2_cell._N2D2_object.getSubSampleX(), n2d2_cell._N2D2_object.getSubSampleY()]
        n2d2_cell._optional_constructor_arguments['stride_dims'] = [n2d2_cell._N2D2_object.getStrideX(), n2d2_cell._N2D2_object.getStrideY()]
        n2d2_cell._optional_constructor_arguments['padding_dims'] = [n2d2_cell._N2D2_object.getPaddingX(), n2d2_cell._N2D2_object.getPaddingY()]
        n2d2_cell._optional_constructor_arguments['dilation_dims'] = [n2d2_cell._N2D2_object.getDilationX(), n2d2_cell._N2D2_object.getDilationY()]

        n2d2_cell._config_parameters['activation'] = \
            n2d2.converter.from_N2D2_object(n2d2_cell._N2D2_object.getActivation())
        n2d2_cell._config_parameters['weights_solver'] = \
            n2d2.converter.from_N2D2_object(n2d2_cell._N2D2_object.getWeightsSolver())
        n2d2_cell._config_parameters['bias_solver'] = \
            n2d2.converter.from_N2D2_object(n2d2_cell._N2D2_object.getBiasSolver())
        n2d2_cell._config_parameters['weights_filler'] = \
            n2d2.converter.from_N2D2_object(n2d2_cell._N2D2_object.getWeightsFiller())
        n2d2_cell._config_parameters['bias_filler'] = \
            n2d2.converter.from_N2D2_object(n2d2_cell._N2D2_object.getBiasFiller())
        quantizer = n2d2_cell._N2D2_object.getQuantizer()
        if quantizer:
            n2d2_cell._config_parameters['quantizer'] = \
                n2d2.converter.from_N2D2_object(quantizer)

        if n2d2_deepnet is not None:
            n2d2_cell._deepnet = n2d2_deepnet
            n2d2_cell._sync_inputs_and_parents()
        else:
            n2d2_cell._deepnet = None
            n2d2_cell._N2D2_object.clearInputs()

        return n2d2_cell

    def __call__(self, inputs):
        super().__call__(inputs)

        self._add_to_graph(inputs)

        self._N2D2_object.propagate(self._inference)

        return self.get_outputs()


    def set_weights_solver(self, solver):
        self._config_parameters['weights_solver'] = solver
        self._N2D2_object.setWeightsSolver(self._config_parameters['weights_solver'].N2D2())

    def set_bias_solver(self, solver):
        self._config_parameters['bias_solver'] = solver
        self._N2D2_object.setBiasSolver(self._config_parameters['bias_solver'].N2D2())

    def set_solver(self, solver):
        self._config_parameters['bias_solver'] = solver
        self._N2D2_object.setBiasSolver(self._config_parameters['bias_solver'].N2D2())
        self._config_parameters['weights_solver'] = solver.copy()
        self._N2D2_object.setWeightsSolver(self._config_parameters['weights_solver'].N2D2())


    def set_quantizer(self, quantizer):
        if 'quantizer' in self._config_parameters:
            raise RuntimeError("Quantizer already exists in cell '" + self.get_name() + "'")
        else:
            self._config_parameters['quantizer'] = quantizer
            self._N2D2_object.setQuantizer(self._config_parameters['quantizer'].N2D2())
            self._N2D2_object.initializeWeightQuantizer()

    def get_quantizer(self):
        if 'quantizer' in self._config_parameters:
            return self._config_parameters['quantizer']

        else:
            raise RuntimeError("No Quantizer in cell '" + self.get_name() + "'")

    def set_weight(self, output_index, channel_index, value):
        """
        :param output_index: 
        :type output_index:
        :param channel_index:
        :type channel_index:
        :param value:
        :type value: :py:class:`n2d2.tensor.Tensor`
        """
        if channel_index >= self.N2D2().getNbChannels():
            raise ValueError("Channel index : " + str(channel_index) + " must be < " + str(self.N2D2().getNbChannels()) +")")
        if output_index >= self.N2D2().getNbOutputs():
            raise ValueError("Output index : " + str(output_index) + " must be < " + str(self.N2D2().getNbOutputs()) +")")
        self.N2D2().setWeight(output_index, channel_index, value.N2D2())

    def get_weight(self, output_index, channel_index):
        """
        :param output_index: 
        :type output_index: int
        :param channel_index:
        :type channel_index: int
        """
        if channel_index >= self.N2D2().getNbChannels():
            raise ValueError("Channel index : " + str(channel_index) + " must be < " + str(self.N2D2().getNbChannels()) +")")
        if output_index >= self.N2D2().getNbOutputs():
            raise ValueError("Output index : " + str(output_index) + " must be < " + str(self.N2D2().getNbOutputs()) +")")
        tensor = N2D2.Tensor_float([])
        self.N2D2().getWeight(output_index, channel_index, tensor)
        return n2d2.Tensor.from_N2D2(tensor)

    def get_weights(self):
        """
        :return: list of weights
        :rtype: list
        """
        weights = []
        tensor = N2D2.Tensor_float([])
        for o in range(self.N2D2().getNbOutputs()):
            chan = []
            for c in range(self.N2D2().getNbChannels()):
                self.N2D2().getWeight(o, c, tensor)
                chan.append(n2d2.Tensor.from_N2D2(tensor))
            weights.append(chan)
        return weights

    def set_bias(self, output_index, value):
        """
        :param output_index: 
        :type output_index: int
        :param value: 
        :type value: :py:class:`n2d2.Tensor`
        """
        if output_index >= self.N2D2().getNbOutputs():
            raise ValueError("Output index : " + str(output_index) + " must be < " + str(self.N2D2().getNbOutputs()) +")")
        self.N2D2().setBias(output_index, value.N2D2())

    def get_bias(self, output_index):
        """
        :param output_index: 
        :type output_index: int
        :return: list of biases
        :rtype: list
        """
        if output_index >= self.N2D2().getNbOutputs():
            raise ValueError("Output index : " + str(output_index) + " must be < " + str(self.N2D2().getNbOutputs()) +")")
        tensor = N2D2.Tensor_float([])
        self.N2D2().getBias(output_index, tensor)
        return n2d2.Tensor.from_N2D2(tensor)
        
    def get_biases(self): # TODO : Is it better to return a list of float instead of a list of Tensor ?
        """
        :return: list of biases
        :rtype: list
        """
        biases = []
        for output_index in range(self.N2D2().getNbOutputs()):
            tensor = N2D2.Tensor_float([])
            self.N2D2().getBias(output_index, tensor)
            biases.append(n2d2.Tensor.from_N2D2(tensor))
        return biases

class ConvDepthWise(Conv):

    def __init__(self,
                 nbChannels,
                 kernel_dims,
                 **config_parameters):
        if 'mapping' in config_parameters:
            raise RuntimeError('ConvDepthWise does not support custom mappings')
        else:
            config_parameters['mapping'] = n2d2.mapping.Mapping(nb_channels_per_group=1).create_mapping(
                nbChannels, nbChannels)
        Conv.__init__(self, nbChannels, nbChannels, kernel_dims, **config_parameters)


class ConvPointWise(Conv):

    def __init__(self,
                 nb_inputs,
                 nb_outputs,
                 **config_parameters):
        Conv.__init__(self, nb_inputs, nb_outputs, [1, 1], stride_dims=[1, 1], **config_parameters)




class Softmax(NeuralNetworkCell, Datatyped):
    """
    Softmax layer.
    """

    _cell_constructors = {
        'Frame<float>': N2D2.SoftmaxCell_Frame_float,
        'Frame_CUDA<float>': N2D2.SoftmaxCell_Frame_CUDA_float,
        'Frame<double>': N2D2.SoftmaxCell_Frame_double,
        'Frame_CUDA<double>': N2D2.SoftmaxCell_Frame_CUDA_double,
    }

    _parameters = {
        "with_loss": "withLoss",
        "group_size": "groupSize",
    }  
    _parameters.update(_cell_frame_parameters)

    _convention_converter= n2d2.ConventionConverter(_parameters)

    def __init__(self, from_arguments=True, **config_parameters):
        r"""
        :param from_arguments: If False, allow you to create cells with mandatory arguments set as None, default=False
        :type  from_arguments: bool, optional
        :param nb_outputs: Number of output channels
        :type nb_outputs: int
        :param name: Name for the cells.
        :type name: str
        :param with_loss: Softmax followed with a multinomial logistic layer, default=False
        :type with_loss: bool, optional
        :param group_size: Softmax is applied on groups of outputs. The group size must be a divisor of ``nb_outputs`` parameter, default=0
        :type group_size: int, optional    
        """
        if not from_arguments and len(config_parameters) > 0:
            raise RuntimeError(
                "N2D2_object argument give to cells but 'inputs' or 'nb_outputs' or 'config parameters' not None")
        if from_arguments:
            self._create_from_arguments(**config_parameters)

    def _create_from_arguments(self, **config_parameters):
        NeuralNetworkCell.__init__(self, **config_parameters)
        self._parse_optional_arguments(['with_loss', 'group_size'])

    @classmethod
    def create_from_N2D2_object(cls, N2D2_object, n2d2_deepnet=None):

        n2d2_cell = cls(from_arguments=False)

        NeuralNetworkCell.__init__(n2d2_cell,
                                   name=N2D2_object.getName(),
                                   **cls.load_N2D2_parameters(N2D2_object))

        n2d2_cell._set_N2D2_object(N2D2_object)

        n2d2_cell._optional_constructor_arguments['with_loss'] = n2d2_cell._N2D2_object.getWithLoss()
        n2d2_cell._optional_constructor_arguments['group_size'] = n2d2_cell._N2D2_object.getGroupSize()

        n2d2_cell._config_parameters['activation'] = \
            n2d2.converter.from_N2D2_object(n2d2_cell._N2D2_object.getActivation())

        if n2d2_deepnet is not None:
            n2d2_cell._deepnet = n2d2_deepnet
            n2d2_cell._sync_inputs_and_parents()
        else:
            n2d2_cell._deepnet = None
            n2d2_cell._N2D2_object.clearInputs()

        return n2d2_cell

    def __call__(self, inputs):
        super().__call__(inputs)

        if self._N2D2_object is None:
            nb_outputs = inputs.dims()[2]

            self._set_N2D2_object(self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                                         self.get_name(),
                                                                         nb_outputs,
                                                                         **self.n2d2_function_argument_parser(self._optional_constructor_arguments)))
            if 'activation' not in self._config_parameters:
                self._config_parameters['activation'] = \
                    n2d2.converter.from_N2D2_object(self._N2D2_object.getActivation())

            """Set and initialize here all complex cells members"""
            for key, value in self._config_parameters.items():
                if key is 'activation':
                    if value:
                        self._N2D2_object.setActivation(value.N2D2())
                else:
                    self._set_N2D2_parameter(self.python_to_n2d2_convention(key), value)

        self._add_to_graph(inputs)

        self._N2D2_object.propagate(self._inference)

        return self.get_outputs()



class Pool(NeuralNetworkCell, Datatyped):
    '''
    Pooling layer.
    '''

    _cell_constructors = {
        'Frame<float>': N2D2.PoolCell_Frame_float,
        'Frame_CUDA<float>': N2D2.PoolCell_Frame_CUDA_float,
    }

    _parameters = {
        "pool_dims": "poolDims",
        "stride_dims": "strideDims",
        "padding_dims": "paddingDims",
        "pooling": "pooling",
        "ext_padding_dims": "ExtPaddingDims",
    }  
    _parameters.update(_cell_frame_parameters)

    _convention_converter= n2d2.ConventionConverter(_parameters)

    def __init__(self,
                 pool_dims,
                 from_arguments=True,
                 **config_parameters):
        """
        :param pool_dims: Pooling area dimensions
        :type pool_dims: list
        :param from_arguments: If False, allow you to create cells with mandatory arguments set as None, default=False
        :type  from_arguments: bool, optional
        :param name: Name for the cells.
        :type name: str
        :param pooling: Type of pooling (``Max`` or ``Average``), default="Max" 
        :type pooling: str, optional
        :param stride_dims: Dimension of the stride of the kernel.
        :type stride_dims: list, optional
        :param padding_dims: Dimensions of the padding.
        :type padding_dims: list, optional
        :param activation: Activation function, default= :py:class:`n2d2.activation.Linear`
        :type activation: :py:class:`n2d2.activation.ActivationFunction`, optional
        :param mapping: Mapping
        :type mapping: :py:class:`n2d2.tensor.Tensor`, optional
        """

        if not from_arguments and (pool_dims is not None or len(config_parameters) > 0):
            raise RuntimeError("from_arguments = True but not None constructor arguments")
        if from_arguments:
            self._create_from_arguments(pool_dims, **config_parameters)


    def _create_from_arguments(self, pool_dims, **config_parameters):
        if not isinstance(pool_dims, list): # TODO : check the type of the elements ?
            raise n2d2.error_handler.WrongInputType("pool_dims", str(type(pool_dims)), ["list"])
        NeuralNetworkCell.__init__(self, **config_parameters)

        self._constructor_arguments.update({
            'pool_dims': pool_dims,
        })

        # Note: Removed Pooling
        self._parse_optional_arguments(['stride_dims', 'padding_dims', 'pooling'])
        if "pooling" in self._optional_constructor_arguments: 
            pooling = self._optional_constructor_arguments["pooling"]
            if not isinstance(pooling, str):
                raise n2d2.error_handler.WrongInputType("pooling", str(type(pooling)), ["str"])
            if pooling not in N2D2.PoolCell.Pooling.__members__.keys():
                raise n2d2.error_handler.WrongValue("pooling", pooling,
                                                    ", ".join(N2D2.PoolCell.Pooling.__members__.keys()))
            self._optional_constructor_arguments['pooling'] = \
                N2D2.PoolCell.Pooling.__members__[self._optional_constructor_arguments['pooling']]

    @classmethod
    def create_from_N2D2_object(cls, N2D2_object, n2d2_deepnet):

        n2d2_cell = cls(None, from_arguments=False)

        NeuralNetworkCell.__init__(n2d2_cell,
                                   name=N2D2_object.getName(),
                                   **cls.load_N2D2_parameters(N2D2_object))

        n2d2_cell._set_N2D2_object(N2D2_object)

        n2d2_cell._constructor_arguments['pool_dims'] = [n2d2_cell._N2D2_object.getPoolWidth(),
                                                        n2d2_cell._N2D2_object.getPoolHeight()]
        n2d2_cell._optional_constructor_arguments['stride_dims'] = [n2d2_cell._N2D2_object.getStrideX(),
                                                                   n2d2_cell._N2D2_object.getStrideY()]
        n2d2_cell._optional_constructor_arguments['padding_dims'] = [n2d2_cell._N2D2_object.getPaddingX(),
                                                                    n2d2_cell._N2D2_object.getPaddingY()]
        n2d2_cell._optional_constructor_arguments['pooling'] = n2d2_cell._N2D2_object.getPooling()

        n2d2_cell._config_parameters['activation'] = \
            n2d2.converter.from_N2D2_object(n2d2_cell._N2D2_object.getActivation())

        if n2d2_deepnet is not None:
            n2d2_cell._deepnet = n2d2_deepnet
            n2d2_cell._sync_inputs_and_parents()
        else:
            n2d2_cell._deepnet = None
            n2d2_cell._N2D2_object.clearInputs()

        return n2d2_cell

    def __call__(self, inputs):
        super().__call__(inputs)
        
        if self._N2D2_object is None:
            mapping_row = 0
            if isinstance(inputs, n2d2.tensor.Interface): # Here we try to support multi input
                for tensor in inputs.get_tensors():
                    if tensor.nb_dims() != 4:
                        raise ValueError("Input Tensor should have 4 dimensions, " + str(inputs.nb_dims()), " were given.")
                    mapping_row += tensor.dimZ()
            elif isinstance(inputs, n2d2.Tensor):
                if inputs.nb_dims() != 4:
                    raise ValueError("Input Tensor should have 4 dimensions, " + str(inputs.nb_dims()), " were given.")
                mapping_row += inputs.dimZ()
            else:
                raise n2d2.wrong_input_type("inputs", inputs, [str(type(list)), str(type(n2d2.Tensor))])

            self._set_N2D2_object(self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                                         self.get_name(),
                                                                         self._constructor_arguments['pool_dims'],
                                                                         mapping_row,
                                                                         **self.n2d2_function_argument_parser(self._optional_constructor_arguments)))

            if 'activation' not in self._config_parameters:
                self._config_parameters['activation'] = \
                    n2d2.converter.from_N2D2_object(self._N2D2_object.getActivation())

            """Set and initialize here all complex cells members"""
            for key, value in self._config_parameters.items():
                if key is 'activation':
                    if value: # TODO : @Johannes why is there a check here but not on the others 
                        self._N2D2_object.setActivation(value.N2D2())
                elif key is 'mapping':
                    if isinstance(value, n2d2.Tensor):
                        if value.dimX() != value.dimY():
                            raise ValueError("Pool Cell supports only unit maps")
                        self._N2D2_object.setMapping(value.N2D2())
                    else:
                        raise n2d2.error_handler.WrongInputType('mapping', type(value), [str(type(n2d2.Tensor))])
                else:
                    self._set_N2D2_parameter(self.python_to_n2d2_convention(key), value)

        self._add_to_graph(inputs)
        
        self._N2D2_object.propagate(self._inference)

        return self.get_outputs()

class Pool2d(NeuralNetworkCell, Datatyped): # Should inherit Pool ?

    _cell_constructors = {
        'Frame<float>': N2D2.PoolCell_Frame_float,
        'Frame_CUDA<float>': N2D2.PoolCell_Frame_CUDA_float,
    }
    _parameters = {
        "pool_dims": "poolDims",
        "stride_dims": "strideDims",
        "padding_dims": "paddingDims",
        "pooling": "pooling",
        "ext_padding_dims": "ExtPaddingDims",
    }  
    _parameters.update(_cell_frame_parameters)

    _convention_converter= n2d2.ConventionConverter(_parameters)
    def __init__(self,
                 pool_dims,
                 from_arguments=True,
                 **config_parameters):


        if not from_arguments and (pool_dims is not None or len(config_parameters) > 0):
            raise RuntimeError("from_arguments = True but not None constructor arguments")
        if from_arguments:
            self._create_from_arguments(pool_dims, **config_parameters)

    def _create_from_arguments(self, pool_dims, **config_parameters):
        if not isinstance(pool_dims, list): # TODO : check the type of the elements ?
            raise n2d2.error_handler.WrongInputType("pool_dims", str(type(pool_dims)), ["list"])
        NeuralNetworkCell.__init__(self, **config_parameters)

        self._constructor_arguments.update({
            'pool_dims': pool_dims,
        })

        self._parse_optional_arguments(['stride_dims', 'padding_dims', 'pooling'])

        if "pooling" in self._optional_constructor_arguments: 
            pooling = self._optional_constructor_arguments["pooling"]
            if not isinstance(pooling, str):
                raise n2d2.error_handler.WrongInputType("pooling", str(type(pooling)), ["str"])
            if pooling not in N2D2.PoolCell.Pooling.__members__.keys():
                raise n2d2.error_handler.WrongValue("pooling", pooling,
                                                    ", ".join(N2D2.PoolCell.Pooling.__members__.keys()))
            self._optional_constructor_arguments['pooling'] = \
                N2D2.PoolCell.Pooling.__members__[self._optional_constructor_arguments['pooling']]

        if 'mapping' in self._config_parameters:
            raise RuntimeError('Pool2d does not support custom mappings')


    def __call__(self, inputs):
        super().__call__(inputs)

        if self._N2D2_object is None:

            self._set_N2D2_object(self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                                         self.get_name(),
                                                                         self._constructor_arguments['pool_dims'],
                                                                         inputs.dims()[2],
                                                                         **self.n2d2_function_argument_parser(self._optional_constructor_arguments)))

            if 'activation' not in self._config_parameters:
                self._config_parameters['activation'] = \
                    n2d2.converter.from_N2D2_object(self._N2D2_object.getActivation())

            """Set and initialize here all complex cells members"""
            for key, value in self._config_parameters.items():
                if key is 'activation':
                    if value:
                        self._N2D2_object.setActivation(value.N2D2())
                else:
                    self._set_N2D2_parameter(self.python_to_n2d2_convention(key), value)

            self._N2D2_object.setMapping(
                n2d2.mapping.Mapping(nb_channels_per_group=1).create_mapping(inputs.dims()[2],
                                                                             inputs.dims()[2]).N2D2())

        self._add_to_graph(inputs)

        self._N2D2_object.propagate(self._inference)

        return self.get_outputs()


class GlobalPool2d(NeuralNetworkCell, Datatyped): # Should inherit Pool ?

    _cell_constructors = {
        'Frame<float>': N2D2.PoolCell_Frame_float,
        'Frame_CUDA<float>': N2D2.PoolCell_Frame_CUDA_float,
    }
    _parameters = {
        "pool_dims": "poolDims",
        "stride_dims": "strideDims",
        "padding_dims": "paddingDims",
        "pooling": "pooling",
        "ext_padding_dims": "ExtPaddingDims",
    }  
    _parameters.update(_cell_frame_parameters)

    _convention_converter= n2d2.ConventionConverter(_parameters)
    def __init__(self,
                 from_arguments=True,
                 **config_parameters):

        if not from_arguments and (len(config_parameters) > 0):
            raise RuntimeError(
                "N2D2_object argument give to cells")
        if from_arguments:
            self._create_from_arguments(**config_parameters)


    def _create_from_arguments(self, **config_parameters):
        NeuralNetworkCell.__init__(self, **config_parameters)

        self._parse_optional_arguments(['pooling'])

        if "pooling" in self._optional_constructor_arguments: 
            pooling = self._optional_constructor_arguments["pooling"]
            if not isinstance(pooling, str):
                raise n2d2.error_handler.WrongInputType("pooling", str(type(pooling)), ["str"])
            if pooling not in self._cell_constructors[self._model_key].Pooling.__members__.keys():
                raise n2d2.error_handler.WrongValue("pooling", pooling,
                                                    ", ".join(self._cell_constructors[self._model_key].Pooling.__members__.keys()))
            self._optional_constructor_arguments['pooling'] = \
                N2D2.PoolCell.Pooling.__members__[self._optional_constructor_arguments['pooling']]

        if 'mapping' in self._config_parameters:
            raise RuntimeError('Pool2d does not support custom mappings')


    #@classmethod
    #def create_from_N2D2_object(cls, inputs, N2D2_object, n2d2_deepnet):
    #    return n2d2_cell

    def __call__(self, inputs):
        super().__call__(inputs)

        if self._N2D2_object is None:

            self._set_N2D2_object(self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                                         self.get_name(),
                                                                         [inputs.dims()[0], inputs.dims()[1]],
                                                                         inputs.dims()[2],
                                                                         strideDims=[1, 1],
                                                                         **self.n2d2_function_argument_parser(self._optional_constructor_arguments)))

            if 'activation' not in self._config_parameters:
                self._config_parameters['activation'] = \
                    n2d2.converter.from_N2D2_object(self._N2D2_object.getActivation())

            """Set and initialize here all complex cells members"""
            for key, value in self._config_parameters.items():
                if key is 'activation':
                    if value:
                        self._N2D2_object.setActivation(value.N2D2())
                else:
                    self._set_N2D2_parameter(self.python_to_n2d2_convention(key), value)

            self._N2D2_object.setMapping(n2d2.mapping.Mapping(nb_channels_per_group=1).create_mapping(inputs.dims()[2], inputs.dims()[2]).N2D2())

        self._add_to_graph(inputs)

        self._N2D2_object.propagate(self._inference)

        return self.get_outputs()


class Deconv(NeuralNetworkCell, Datatyped):
    """
    Deconvolution layer.
    """
    _cell_constructors = {
        'Frame<float>': N2D2.DeconvCell_Frame_float,
        'Frame_CUDA<float>': N2D2.DeconvCell_Frame_CUDA_float,
        'Frame<double>': N2D2.DeconvCell_Frame_double,
        'Frame_CUDA<double>': N2D2.DeconvCell_Frame_CUDA_double,
    }
    _parameters = {
        "no_bias":"NoBias", 
        "back_propagate":"BackPropagate",
        "weights_export_format":"WeightsExportFormat",
        "weights_export_flip":"WeightsExportFlip",
        "outputs_remap":"OutputsRemap",
        "kernel_dims":"kernelDims",
        "sub_sample_dims":"subSampleDims",
        "stride_dims":"strideDims",
        "padding_dims":"paddingDims",
        "dilation_dims":"dilationDims",  
        "ext_padding_dims":"ExtPaddingDims",   
        "dilation_dims":"dilationDims",
        "weights_filler":"WeightsFiller",  
        "bias_filler":"BiasFiller",  
        "weights_solver":"WeightsSolver",
        "bias_solver":"BiasSolver",
        # TODO : @Johannes No quantizer ?
    }  
    _parameters.update(_cell_frame_parameters)

    _convention_converter= n2d2.ConventionConverter(_parameters)

    def __init__(self,
                 nb_inputs,
                 nb_outputs,
                 kernel_dims,
                 nb_input_cells=1,
                 from_arguments=True,
                 **config_parameters):
        """
        :param nb_inputs: Number of inputs of the cells.
        :type nb_inputs: int
        :param nb_outputs: Number of output channels
        :type nb_outputs: int
        :param kernel_dims: Kernel dimension.
        :type kernel_dims: list
        :param from_arguments: If False, allow you to create cells with mandatory arguments set as None, default=False
        :type  from_arguments: bool, optional
        :param name: Name for the cells.
        :type name: str
        :param stride_dims: Dimension of the stride of the kernel.
        :type stride_dims: list, optional
        :param padding_dims: Dimensions of the padding.
        :type padding_dims: list, optional
        :param dilation_dims: Dimensions of the dilation of the kernels 
        :type dilation_dims: list, optional
        :param activation: Activation function, default= :py:class:`n2d2.activation.Tanh`
        :type activation: :py:class:`n2d2.activation.ActivationFunction`, optional
        :param weights_filler: Weights initial values filler, default=NormalFiller
        :type weights_filler: :py:class:`n2d2.filler.Filler`, optional
        :param bias_filler: Biases initial values filler, default=NormalFiller
        :type bias_filler: :py:class:`n2d2.filler.Filler`, optional
        :param weights_solver: Solver for weights
        :type weights_solver: :py:class:`n2d2.solver.Solver`, optional
        :param bias_solver: Solver for biases
        :type bias_solver: :py:class:`n2d2.solver.Solver`, optional
        :param no_bias: If True, don’t use bias, default=False
        :type no_bias: bool, optional
        :param back_propagate: If True, enable backpropagation, default=True
        :type back_propagate: bool, optional
        :param weights_export_flip: If true, import/export flipped kernels, default=False
        :type weights_export_flip: bool, optional
        :param mapping: Mapping
        :type mapping: :py:class:`n2d2.tensor.Tensor`, optional
        """

        if not from_arguments and (nb_inputs is not None or nb_outputs is not None or kernel_dims is not None or len(config_parameters) > 0):
            raise RuntimeError("from_arguments = True but not None constructor arguments")
        if from_arguments:
            self._create_from_arguments(nb_inputs, nb_outputs, kernel_dims, nb_input_cells, **config_parameters)

        

    def _create_from_arguments(self, nb_inputs, nb_outputs, kernel_dims, nb_input_cells, **config_parameters):
        if not isinstance(nb_inputs, int):
            raise n2d2.error_handler.WrongInputType("nb_inputs", str(type(nb_inputs)), ["int"])
        if not isinstance(nb_outputs, int):
            raise n2d2.error_handler.WrongInputType("nb_outputs", str(type(nb_outputs)), ["int"])
        if not isinstance(kernel_dims, list): # TODO : check the type of the elements ?
            raise n2d2.error_handler.WrongInputType("kernel_dims", str(type(kernel_dims)), ["list"])

        NeuralNetworkCell.__init__(self, **config_parameters)

        self._constructor_arguments.update({
            'nb_inputs': nb_inputs,
            'nb_outputs': nb_outputs,
            'kernel_dims': kernel_dims,
        })

        self._parse_optional_arguments(['stride_dims', 'padding_dims', 'dilation_dims'])

        self._set_N2D2_object(self._cell_constructors[self._model_key](N2D2.DeepNet(n2d2.global_variables.default_net),
                                                                     self.get_name(),
                                                                     self._constructor_arguments['kernel_dims'],
                                                                     self._constructor_arguments['nb_outputs'],
                                                                     **self.n2d2_function_argument_parser(self._optional_constructor_arguments)))


        # TODO: Add Kernel section of generator

        if 'activation' not in self._config_parameters:
            self._config_parameters['activation'] = \
                n2d2.converter.from_N2D2_object(self._N2D2_object.getActivation())
        if 'weights_solver' not in self._config_parameters:
            self._config_parameters['weights_solver'] = \
                n2d2.converter.from_N2D2_object(self._N2D2_object.getWeightsSolver())
        if 'bias_solver' not in self._config_parameters:
            self._config_parameters['bias_solver'] = \
                n2d2.converter.from_N2D2_object(self._N2D2_object.getBiasSolver())
        if 'weights_filler' not in self._config_parameters:
            self._config_parameters['weights_filler'] = \
                n2d2.converter.from_N2D2_object(self._N2D2_object.getWeightsFiller())
        if 'bias_filler' not in self._config_parameters:
            self._config_parameters['bias_filler'] = \
                n2d2.converter.from_N2D2_object(self._N2D2_object.getBiasFiller())

        """Set and initialize here all complex cells members"""
        for key, value in self._config_parameters.items():
            if key is 'activation':
                if value: # TODO : @Johannes why is there a check here but not on the others 
                        self._N2D2_object.setActivation(value.N2D2())
            elif key is 'weights_solver':
                if isinstance(value, n2d2.solver.Solver):
                    self._N2D2_object.setWeightsSolver(value.N2D2())
                else:
                    raise n2d2.error_handler.WrongInputType("weights_solver", str(type(value)), [str(n2d2.solver.Solver)])
            elif key is 'bias_solver':
                if isinstance(value, n2d2.solver.Solver):
                    self._N2D2_object.setBiasSolver(value.N2D2())
                else:
                    raise n2d2.error_handler.WrongInputType("bias_solver", str(type(value)), [str(n2d2.solver.Solver)])
            elif key is 'weights_filler':
                if isinstance(value, n2d2.filler.Filler):
                    self._N2D2_object.setWeightsFiller(value.N2D2())
                else:
                    raise n2d2.error_handler.WrongInputType("weights_filler", str(type(value)), [str(n2d2.filler.Filler)])
            elif key is 'bias_filler':
                if isinstance(value, n2d2.filler.Filler):
                    self._N2D2_object.setBiasFiller(value.N2D2())
                else:
                    raise n2d2.error_handler.WrongInputType("bias_filler", str(type(value)), [str(n2d2.filler.Filler)])
            elif key is 'quantizer':
                if isinstance(value, n2d2.quantizer.Quantizer):
                    self._N2D2_object.setQuantizer(value.N2D2())
                else:
                    raise n2d2.error_handler.WrongInputType("quantizer", str(type(value)), [str(n2d2.quantizer.Quantizer)])
            elif key is 'mapping':
                if isinstance(value, n2d2.Tensor):
                    if not (value.data_type() == "bool" or value.data_type() == "b"):
                        raise ValueError("Mapping Tensor datatype should be boolean !")
                    self._N2D2_object.setMapping(value.N2D2())
                else:
                    raise n2d2.error_handler.WrongInputType('mapping', type(value), [str(type(n2d2.Tensor))])
            else:
                self._set_N2D2_parameter(self.python_to_n2d2_convention(key), value)

        self._N2D2_object.initializeParameters(nb_inputs, nb_input_cells)

    @classmethod
    def create_from_N2D2_object(cls, N2D2_object, n2d2_deepnet=None):

        n2d2_cell = cls(None, None, None, from_arguments=False)

        NeuralNetworkCell.__init__(n2d2_cell,
                                   name=N2D2_object.getName(),
                                   **cls.load_N2D2_parameters(N2D2_object))

        n2d2_cell._set_N2D2_object(N2D2_object)

        n2d2_cell._constructor_arguments.update({
            'nb_inputs': n2d2_cell._N2D2_object.getNbChannels(),
            'nb_outputs': n2d2_cell._N2D2_object.getNbOutputs(),
            'kernel_dims': [n2d2_cell._N2D2_object.getKernelWidth(), n2d2_cell._N2D2_object.getKernelHeight()]
        })

        n2d2_cell._optional_constructor_arguments['stride_dims'] = [n2d2_cell._N2D2_object.getStrideX(),
                                                                    n2d2_cell._N2D2_object.getStrideY()]
        n2d2_cell._optional_constructor_arguments['padding_dims'] = [n2d2_cell._N2D2_object.getPaddingX(),
                                                                     n2d2_cell._N2D2_object.getPaddingY()]
        n2d2_cell._optional_constructor_arguments['dilation_dims'] = [n2d2_cell._N2D2_object.getDilationX(),
                                                                      n2d2_cell._N2D2_object.getDilationY()]
        n2d2_cell._config_parameters['activation'] = \
            n2d2.converter.from_N2D2_object(n2d2_cell._N2D2_object.getActivation())
        n2d2_cell._config_parameters['weights_solver'] = \
            n2d2.converter.from_N2D2_object(n2d2_cell._N2D2_object.getWeightsSolver())
        n2d2_cell._config_parameters['bias_solver'] = \
            n2d2.converter.from_N2D2_object(n2d2_cell._N2D2_object.getBiasSolver())
        n2d2_cell._config_parameters['weights_filler'] = \
            n2d2.converter.from_N2D2_object(n2d2_cell._N2D2_object.getWeightsFiller())
        n2d2_cell._config_parameters['bias_filler'] = \
            n2d2.converter.from_N2D2_object(n2d2_cell._N2D2_object.getBiasFiller())
        quantizer = n2d2_cell._N2D2_object.getQuantizer()
        if quantizer:
            n2d2_cell._config_parameters['quantizer'] = \
                n2d2.converter.from_N2D2_object(quantizer)

        if n2d2_deepnet is not None:
            n2d2_cell._deepnet = n2d2_deepnet
            n2d2_cell._sync_inputs_and_parents()
        else:
            n2d2_cell._deepnet = None
            n2d2_cell._N2D2_object.clearInputs()

        return n2d2_cell

    def __call__(self, inputs):
        super().__call__(inputs)

        self._add_to_graph(inputs)

        self._N2D2_object.propagate(self._inference)

        return self.get_outputs()

    def set_weights_solver(self, solver):
        self._config_parameters['weights_solver'] = solver
        self._N2D2_object.setWeightsSolver(self._config_parameters['weights_solver'].N2D2())


    def set_bias_solver(self, solver):
        self._config_parameters['bias_solver'] = solver
        self._N2D2_object.setBiasSolver(self._config_parameters['bias_solver'].N2D2())

    def set_solver(self, solver):
        self._config_parameters['bias_solver'] = solver
        self._N2D2_object.setBiasSolver(self._config_parameters['bias_solver'].N2D2())
        self._config_parameters['weights_solver'] = solver.copy()
        self._N2D2_object.setWeightsSolver(self._config_parameters['weights_solver'].N2D2())

    def set_weight(self, output_index, channel_index, value):
        """
        :param output_index: 
        :type output_index:
        :param channel_index:
        :type channel_index:
        :param value:
        :type value: :py:class:`n2d2.tensor.Tensor`
        """
        if channel_index >= self.N2D2().getNbChannels():
            raise ValueError("Channel index : " + str(channel_index) + " must be < " + str(self.N2D2().getNbChannels()) +")")
        if output_index >= self.N2D2().getNbOutputs():
            raise ValueError("Output index : " + str(output_index) + " must be < " + str(self.N2D2().getNbOutputs()) +")")
        self.N2D2().setWeight(output_index, channel_index, value.N2D2())

    def get_weight(self, output_index, channel_index):
        """
        :param output_index: 
        :type output_index: int
        :param channel_index:
        :type channel_index: int
        """
        if channel_index >= self.N2D2().getNbChannels():
            raise ValueError("Channel index : " + str(channel_index) + " must be < " + str(self.N2D2().getNbChannels()) +")")
        if output_index >= self.N2D2().getNbOutputs():
            raise ValueError("Output index : " + str(output_index) + " must be < " + str(self.N2D2().getNbOutputs()) +")")
        tensor = N2D2.Tensor_float([])
        self.N2D2().getWeight(output_index, channel_index, tensor)
        return n2d2.Tensor.from_N2D2(tensor)

    def get_weights(self):
        """
        :return: list of weights
        :rtype: list
        """
        weights = []
        tensor = N2D2.Tensor_float([])
        for o in range(self.N2D2().getNbOutputs()):
            chan = []
            for c in range(self.N2D2().getNbChannels()):
                self.N2D2().getWeight(o, c, tensor)
                chan.append(n2d2.Tensor.from_N2D2(tensor))
            weights.append(chan)
        return weights

    def set_bias(self, output_index, value):
        """
        :param output_index: 
        :type output_index: int
        :param value: 
        :type value: :py:class:`n2d2.Tensor`
        """
        if output_index >= self.N2D2().getNbOutputs():
            raise ValueError("Output index : " + str(output_index) + " must be < " + str(self.N2D2().getNbOutputs()) +")")
        self.N2D2().setBias(output_index, value.N2D2())

    def get_bias(self, output_index):
        """
        :param output_index: 
        :type output_index: int
        """
        if output_index >= self.N2D2().getNbOutputs():
            raise ValueError("Output index : " + str(output_index) + " must be < " + str(self.N2D2().getNbOutputs()) +")")
        tensor = N2D2.Tensor_float([])
        self.N2D2().getBias(output_index, tensor)
        return n2d2.Tensor.from_N2D2(tensor)
        
    def get_biases(self):
        """
        :return: list of biases
        :rtype: list
        """
        biases = []
        for output_index in range(self.N2D2().getNbOutputs()):
            tensor = N2D2.Tensor_float([])
            self.N2D2().getBias(output_index, tensor)
            biases.append(n2d2.Tensor.from_N2D2(tensor))
        return biases

class ElemWise(NeuralNetworkCell):
    """
    Element-wise operation layer.
    """

    _cell_constructors = {
        'Frame': N2D2.ElemWiseCell_Frame,
        'Frame_CUDA': N2D2.ElemWiseCell_Frame_CUDA,
    }
    # TODO: Incoherence coeff_mode/mode
    _parameters = {
        "operation": "operation",
        "coeff_mode": "mode",
        "weights": "weights",
        "shifts": "shifts"
    }  
    _parameters.update(_cell_frame_parameters)

    _convention_converter = n2d2.ConventionConverter(_parameters)

    def __init__(self, from_arguments=True, **config_parameters):
        """
        :param from_arguments: If False, allow you to create cells with mandatory arguments set as None, default=False
        :type  from_arguments: bool, optional
        :param operation: Type of operation (``Sum``, ``AbsSum``, ``EuclideanSum``, ``Prod``, or ``Max``), default="Sum"
        :type operation: str, optional
        :param mode: (``PerLayer``, ``PerInput``, ``PerChannel``), default="PerLayer"
        :type mode: str, optional
        :param weights: Weights for the ``Sum``, ``AbsSum``, and ``EuclideanSum`` operation, in the same order as the inputs, default=[1.0]
        :type weights: list, optional
        :param shifts: Shifts for the ``Sum`` and ``EuclideanSum`` operation, in the same order as the inputs, default=[0.0]
        :type shifts: list, optional
        :param activation: Activation function, default= :py:class:`n2d2.activation.Linear`
        :type activation: :py:class:`n2d2.activation.ActivationFunction`, optional
        """
        if not from_arguments and (len(config_parameters) > 0):
            raise RuntimeError("from_arguments = True but not None constructor arguments")
        if from_arguments:
            self._create_from_arguments(**config_parameters)
        


    def _create_from_arguments(self, **config_parameters):
        NeuralNetworkCell.__init__(self, **config_parameters)

        self._parse_optional_arguments(['operation', 'coeff_mode', 'weights', 'shifts'])
        
        if "operation" in self._optional_constructor_arguments:
            operation = self._optional_constructor_arguments["operation"]
            if not isinstance(operation, str):
                raise n2d2.error_handler.WrongInputType("operation", str(type(operation)), ["str"])
            if operation not in N2D2.ElemWiseCell.Operation.__members__.keys():
                raise n2d2.error_handler.WrongValue("operation", operation,
                                                    ", ".join(N2D2.ElemWiseCell.Operation.__members__.keys()))
            self._optional_constructor_arguments['operation'] = \
                N2D2.ElemWiseCell.Operation.__members__[self._optional_constructor_arguments['operation']]
        if "coeff_mode" in self._optional_constructor_arguments:
            mode = self._optional_constructor_arguments["coeff_mode"]
            if not isinstance(mode, str):
                raise n2d2.error_handler.WrongInputType("coeff_mode", str(type(coeff_mode)), ["str"])
            if mode not in N2D2.ElemWiseCell.CoeffMode.__members__.keys():
                raise n2d2.error_handler.WrongValue("operation", operation,
                                                    ", ".join(N2D2.ElemWiseCell.CoeffMode.__members__.keys()))
            self._optional_constructor_arguments['coeff_mode'] = \
                N2D2.ElemWiseCell.CoeffMode.__members__[self._optional_constructor_arguments['coeff_mode']]
        if "weights" in self._optional_constructor_arguments:
            if not isinstance(self._optional_constructor_arguments["weights"], list):
                raise n2d2.error_handler.WrongInputType("weights", str(type(self._optional_constructor_arguments["weights"])), ["float"])
        if "shifts" in self._optional_constructor_arguments:
            if not isinstance(self._optional_constructor_arguments["shifts"], list):
                raise n2d2.error_handler.WrongInputType("shifts", str(type(self._optional_constructor_arguments["shifts"])), ["float"])



    @classmethod
    def create_from_N2D2_object(cls, N2D2_object, n2d2_deepnet=None):

        n2d2_cell = cls(from_arguments=False)

        NeuralNetworkCell.__init__(n2d2_cell,
                                   name=N2D2_object.getName(),
                                   **cls.load_N2D2_parameters(N2D2_object))

        n2d2_cell._set_N2D2_object(N2D2_object)

        n2d2_cell._optional_constructor_arguments['operation'] = n2d2_cell._N2D2_object.getOperation()
        n2d2_cell._optional_constructor_arguments['coeff_mode'] = n2d2_cell._N2D2_object.getCoeffMode()
        n2d2_cell._optional_constructor_arguments['weights'] = n2d2_cell._N2D2_object.getWeights()
        n2d2_cell._optional_constructor_arguments['shifts'] = n2d2_cell._N2D2_object.getShifts()

        n2d2_cell._config_parameters['activation'] = \
            n2d2.converter.from_N2D2_object(n2d2_cell._N2D2_object.getActivation())

        if n2d2_deepnet is not None:
            n2d2_cell._deepnet = n2d2_deepnet
            n2d2_cell._sync_inputs_and_parents()
        else:
            n2d2_cell._deepnet = None
            n2d2_cell._N2D2_object.clearInputs()

        return n2d2_cell

    def __call__(self, inputs):

        super().__call__(inputs)

        if self._N2D2_object is None:

            mapping_row = 0

            if isinstance(inputs,
                          n2d2.tensor.Interface):
                for tensor in inputs.get_tensors():
                    if tensor.nb_dims() != 4:
                        raise ValueError("Input Tensor should have 4 dimensions, " + str(inputs.nb_dims()),
                                         " were given.")
                    mapping_row = tensor.dimZ()
            elif isinstance(inputs, n2d2.Tensor):
                if inputs.nb_dims() != 4:
                    raise ValueError("Input Tensor should have 4 dimensions, " + str(inputs.nb_dims()), " were given.")
                mapping_row = inputs.dimZ()

            else:
                raise n2d2.wrong_input_type("inputs", inputs, [str(type(list)), str(type(n2d2.Tensor))])


            self._set_N2D2_object(self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                                     self.get_name(),
                                                                     mapping_row,
                                                                     **self.n2d2_function_argument_parser(
                                                                         self._optional_constructor_arguments)))

            if 'activation' not in self._config_parameters:
                self._config_parameters['activation'] = \
                    n2d2.converter.from_N2D2_object(self._N2D2_object.getActivation())

            """Set and initialize here all complex cells members"""
            for key, value in self._config_parameters.items():
                if key is 'activation':
                    if value:
                        self._N2D2_object.setActivation(value.N2D2())
                else:
                    self._set_N2D2_parameter(self.python_to_n2d2_convention(key), value)

        self._add_to_graph(inputs)

        self._N2D2_object.propagate(self._inference)

        return self.get_outputs()


class Dropout(NeuralNetworkCell, Datatyped):
    """
    Dropout layer :cite:`Srivastava2014`.
    """
    _type = "Dropout"

    _cell_constructors = {
        'Frame<float>': N2D2.DropoutCell_Frame_float,
        'Frame_CUDA<float>': N2D2.DropoutCell_Frame_CUDA_float,
        'Frame<double>': N2D2.DropoutCell_Frame_double,
        'Frame_CUDA<double>': N2D2.DropoutCell_Frame_CUDA_double,
    }

    _parameters = {
        "dropout": "Dropout",
    }  
    _parameters.update(_cell_frame_parameters)

    _convention_converter= n2d2.ConventionConverter(_parameters)

    def __init__(self, from_arguments=True, **config_parameters):
        """
        :param from_arguments: If False, allow you to create cells with mandatory arguments set as None, default=False
        :type  from_arguments: bool, optional
        :param name: Name for the cells.
        :type name: str
        :param dropout: The probability with which the value from input would be dropped, default=0.5
        :type dropout: float, optional
        """

        if not from_arguments and  len(config_parameters) > 0:
            raise RuntimeError("from_arguments = True but not None constructor arguments")
        if from_arguments:
            self._create_from_arguments(**config_parameters)

    def _create_from_arguments(self,  **config_parameters):
        NeuralNetworkCell.__init__(self, **config_parameters)

        self._parse_optional_arguments([])
        if "dropout" in config_parameters:
            if not isinstance(config_parameters["dropout"], float):
                raise n2d2.error_handler.WrongInputType("dropout", str(type(config_parameters["dropout"])), ["float"])

    @classmethod
    def create_from_N2D2_object(cls, N2D2_object, n2d2_deepnet=None):

        n2d2_cell = cls(from_arguments=False)

        NeuralNetworkCell.__init__(n2d2_cell,
                                   name=N2D2_object.getName(),
                                   **cls.load_N2D2_parameters(N2D2_object))

        n2d2_cell._set_N2D2_object(N2D2_object)

        n2d2_cell._config_parameters['activation'] = \
            n2d2.converter.from_N2D2_object(n2d2_cell._N2D2_object.getActivation())

        if n2d2_deepnet is not None:
            n2d2_cell._deepnet = n2d2_deepnet
            n2d2_cell._sync_inputs_and_parents()
        else:
            n2d2_cell._deepnet = None
            n2d2_cell._N2D2_object.clearInputs()

        return n2d2_cell
        
    def __call__(self, inputs):
        super().__call__(inputs)

        if self._N2D2_object is None:
            nb_outputs = inputs.dims()[2]

            self._set_N2D2_object(self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                                         self.get_name(),
                                                                         nb_outputs,
                                                                         **self.n2d2_function_argument_parser(self._optional_constructor_arguments)))

            if 'activation' not in self._config_parameters:
                self._config_parameters['activation'] = \
                    n2d2.converter.from_N2D2_object(self._N2D2_object.getActivation())

            """Set and initialize here all complex cells members"""
            for key, value in self._config_parameters.items():
                if key is 'activation':
                    if value:
                        self._N2D2_object.setActivation(value.N2D2())
                else:
                    self._set_N2D2_parameter(self.python_to_n2d2_convention(key), value)

        self._add_to_graph(inputs)

        self._N2D2_object.propagate(self._inference)

        return self.get_outputs()




class Padding(NeuralNetworkCell):

    _cell_constructors = {
        'Frame': N2D2.PaddingCell_Frame,
        'Frame_CUDA': N2D2.PaddingCell_Frame_CUDA,
    }

    _parameters = {
        "top_pad":"top_pad",
        "bot_pad":"bot_pad",
        "left_pad":"left_pad",
        "right_pad":"right_pad",
    }
    _parameters.update(_cell_frame_parameters)

    _convention_converter= n2d2.ConventionConverter(_parameters)

    def __init__(self,
                 top_pad,
                 bot_pad,
                 left_pad,
                 right_pad,
                 from_arguments=True,
                 **config_parameters):
        """
        :param top_pad: Size of the top padding (positive or negative)
        :type top_pad: int
        :param bot_pad: Size of the bottom padding (positive or negative)
        :type bot_pad: int
        :param left_pad: Size of the left padding (positive or negative)
        :type left_pad: int
        :param right_pad: Size of the right padding (positive or negative)
        :type right_pad: int
        """
        if not from_arguments and (top_pad is not None
                                    or bot_pad is not None
                                    or left_pad is not None
                                    or right_pad is not None
                                    or len(config_parameters) > 0):
            raise RuntimeError("from_arguments = True but not None constructor arguments")
        if from_arguments:
            self._create_from_arguments(top_pad, bot_pad, left_pad, right_pad, **config_parameters)
        

    def _create_from_arguments(self, top_pad, bot_pad, left_pad, right_pad, **config_parameters):
        if not isinstance(top_pad, int):
            raise n2d2.error_handler.WrongInputType("top_pad", str(type(top_pad)), ["int"])
        if not isinstance(bot_pad, int):
            raise n2d2.error_handler.WrongInputType("bot_pad", str(type(bot_pad)), ["int"])
        if not isinstance(left_pad, int):
            raise n2d2.error_handler.WrongInputType("left_pad", str(type(left_pad)), ["int"])
        if not isinstance(right_pad, int):
            raise n2d2.error_handler.WrongInputType("right_pad", str(type(right_pad)), ["int"])
        NeuralNetworkCell.__init__(self, **config_parameters)

        self._constructor_arguments.update({
                 'top_pad': top_pad,
                 'bot_pad': bot_pad,
                 'left_pad': left_pad,
                 'right_pad': right_pad
        })
        # No optional args
        self._parse_optional_arguments([])

    @classmethod
    def create_from_N2D2_object(cls, N2D2_object, n2d2_deepnet):

        n2d2_cell = cls(None, None, None, None, from_arguments=False)

        NeuralNetworkCell.__init__(n2d2_cell,
                                   name=N2D2_object.getName(),
                                   **cls.load_N2D2_parameters(N2D2_object))

        n2d2_cell._set_N2D2_object(N2D2_object)

        n2d2_cell._constructor_arguments['top_pad'] = n2d2_cell._N2D2_object.getTopPad()
        n2d2_cell._constructor_arguments['bot_pad'] = n2d2_cell._N2D2_object.getBotPad()
        n2d2_cell._constructor_arguments['left_pad'] = n2d2_cell._N2D2_object.getLeftPad()
        n2d2_cell._constructor_arguments['right_pad'] = n2d2_cell._N2D2_object.getRightPad()

        n2d2_cell._config_parameters['activation'] = \
            n2d2.converter.from_N2D2_object(n2d2_cell._N2D2_object.getActivation())

        if n2d2_deepnet is not None:
            n2d2_cell._deepnet = n2d2_deepnet
            n2d2_cell._sync_inputs_and_parents()
        else:
            n2d2_cell._deepnet = None
            n2d2_cell._N2D2_object.clearInputs()

        return n2d2_cell


    def __call__(self, inputs):
        super().__call__(inputs)

        if self._N2D2_object is None:
            nb_outputs = inputs.dims()[2]

            self._set_N2D2_object(self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                                     self.get_name(),
                                                                     nb_outputs,
                                                                     self._constructor_arguments['top_pad'],
                                                                     self._constructor_arguments['bot_pad'],
                                                                     self._constructor_arguments['left_pad'],
                                                                     self._constructor_arguments['right_pad'],
                                                                     **self.n2d2_function_argument_parser(
                                                                         self._optional_constructor_arguments)))
            if 'activation' not in self._config_parameters:
                self._config_parameters['activation'] = \
                    n2d2.converter.from_N2D2_object(self._N2D2_object.getActivation())

            """Set and initialize here all complex cells members"""
            for key, value in self._config_parameters.items():
                if key is 'activation':
                    if value:
                        self._N2D2_object.setActivation(value.N2D2())
                else:
                    self._set_N2D2_parameter(self.python_to_n2d2_convention(key), value)

        self._add_to_graph(inputs)

        self._N2D2_object.propagate(self._inference)

        return self.get_outputs()


class BatchNorm2d(NeuralNetworkCell, Datatyped):

    _cell_constructors = {
        'Frame<float>': N2D2.BatchNormCell_Frame_float,
        'Frame_CUDA<float>': N2D2.BatchNormCell_Frame_CUDA_float,
    }

    _parameters = {
        "nb_inputs": "NbInputs",
        "scale_solver": "ScaleSolver",
        "bias_solver": "BiasSolver",
        "moving_average_momentum": "MovingAverageMomentum",
        "epsilon": "Epsilon",
        "backpropagate": "BackPropagate"
    }
    _parameters.update(_cell_frame_parameters)

    _convention_converter= n2d2.ConventionConverter(_parameters)

    def __init__(self, nb_inputs, nb_input_cells=1, from_arguments=True, **config_parameters):
        # TODO : Update doc string

        """
        :param moving_average_momentum: Moving average rate: used for the moving average of batch-wise means and standard deviations during training.The closer to 1.0, the more it will depend on the last batch
        :type moving_average_momentum: float, optional
        """
        if not from_arguments and len(config_parameters) > 0:
            raise RuntimeError("from_arguments = True but not None constructor arguments")
        if from_arguments:
            self._create_from_arguments(nb_inputs, nb_input_cells, **config_parameters)

    def _create_from_arguments(self, nb_inputs, nb_input_cells, **config_parameters):
        if not isinstance(nb_inputs, int):
            raise n2d2.error_handler.WrongInputType("nb_inputs", str(type(nb_inputs)), ["int"])

        NeuralNetworkCell.__init__(self, **config_parameters)
        # No optional parameter
        self._parse_optional_arguments([])
        self._set_N2D2_object(self._cell_constructors[self._model_key](N2D2.DeepNet(n2d2.global_variables.default_net),
                                                self.get_name(),
                                                nb_inputs,
                                                **self.n2d2_function_argument_parser(self._optional_constructor_arguments)))

        if 'activation' not in self._config_parameters:
            self._config_parameters['activation'] = \
                n2d2.converter.from_N2D2_object(self._N2D2_object.getActivation())
        if 'scale_solver' not in self._config_parameters:
            self._config_parameters['scale_solver'] = \
                n2d2.converter.from_N2D2_object(self._N2D2_object.getScaleSolver())
        if 'bias_solver' not in self._config_parameters:
            self._config_parameters['bias_solver'] = \
                n2d2.converter.from_N2D2_object(self._N2D2_object.getBiasSolver())

        """Set and initialize here all complex cells members"""
        for key, value in self._config_parameters.items():
            if key is 'activation':
                    if value:
                        self._N2D2_object.setActivation(value.N2D2())
            elif key is 'scale_solver':
                if not isinstance(value, n2d2.solver.Solver):
                    raise n2d2.error_handler.WrongInputType("scale_solver", str(type(value)), [str(n2d2.solver.Solver)])
                self._N2D2_object.setScaleSolver(value.N2D2())
            elif key is 'bias_solver':
                if not isinstance(value, n2d2.solver.Solver):
                    raise n2d2.error_handler.WrongInputType("bias_solver", str(type(value)), [str(n2d2.solver.Solver)])
                self._N2D2_object.setBiasSolver(value.N2D2())
            else:
                self._set_N2D2_parameter(self.python_to_n2d2_convention(key), value)

        self._N2D2_object.initializeParameters(nb_inputs, nb_input_cells)



    @classmethod
    def create_from_N2D2_object(cls, N2D2_object, n2d2_deepnet=None):

        n2d2_cell = cls(None, from_arguments=False)

        NeuralNetworkCell.__init__(n2d2_cell,
                                   name=N2D2_object.getName(),
                                   **cls.load_N2D2_parameters(N2D2_object))

        n2d2_cell._set_N2D2_object(N2D2_object)

        n2d2_cell._constructor_arguments.update({
            'nb_inputs': n2d2_cell._N2D2_object.getNbChannels(),
        })

        n2d2_cell._config_parameters['activation'] = \
            n2d2.converter.from_N2D2_object(n2d2_cell._N2D2_object.getActivation())
        n2d2_cell._config_parameters['scale_solver'] = \
            n2d2.converter.from_N2D2_object(n2d2_cell._N2D2_object.getScaleSolver())
        n2d2_cell._config_parameters['bias_solver'] = \
            n2d2.converter.from_N2D2_object(n2d2_cell._N2D2_object.getBiasSolver())

        if n2d2_deepnet is not None:
            n2d2_cell._deepnet = n2d2_deepnet
            n2d2_cell._sync_inputs_and_parents()
        else:
            n2d2_cell._deepnet = None
            n2d2_cell._N2D2_object.clearInputs()

        return n2d2_cell

    def __call__(self, inputs):
        super().__call__(inputs)

        self._add_to_graph(inputs)

        self._N2D2_object.propagate(self._inference)

        return self.get_outputs()

    def set_scale_solver(self, solver):
        self._config_parameters['scale_solver'] = solver
        self._N2D2_object.setScaleSolver(self._config_parameters['scale_solver'].N2D2())

    def set_bias_solver(self, solver):
        self._config_parameters['bias_solver'] = solver
        self._N2D2_object.setBiasSolver(self._config_parameters['bias_solver'].N2D2())



class Activation(NeuralNetworkCell, Datatyped):

    _cell_constructors = {
        'Frame<float>': N2D2.ActivationCell_Frame_float,
        'Frame_CUDA<float>': N2D2.ActivationCell_Frame_CUDA_float,
    }

    _parameters = {
    }
    _parameters.update(_cell_frame_parameters)
    
    _convention_converter= n2d2.ConventionConverter(_parameters)
    def __init__(self, from_arguments=True, **config_parameters):

        if not from_arguments and len(config_parameters) > 0:
            raise RuntimeError("from_arguments = True but not None constructor arguments")
        if from_arguments:
            self._create_from_arguments(**config_parameters)
        

    def _create_from_arguments(self, **config_parameters):
        NeuralNetworkCell.__init__(self, **config_parameters)
        # No optional parameter
        self._parse_optional_arguments([])


    @classmethod
    def create_from_N2D2_object(cls, N2D2_object, n2d2_deepnet=None):

        n2d2_cell = cls(from_arguments=False)

        NeuralNetworkCell.__init__(n2d2_cell,
                                   name=N2D2_object.getName(),
                                   **cls.load_N2D2_parameters(N2D2_object))

        n2d2_cell._set_N2D2_object(N2D2_object)

        n2d2_cell._config_parameters['activation'] = \
            n2d2.converter.from_N2D2_object(n2d2_cell._N2D2_object.getActivation())

        if n2d2_deepnet is not None:
            n2d2_cell._deepnet = n2d2_deepnet
            n2d2_cell._sync_inputs_and_parents()
        else:
            n2d2_cell._deepnet = None
            n2d2_cell._N2D2_object.clearInputs()

        return n2d2_cell

    def __call__(self, inputs):
        super().__call__(inputs)

        if self._N2D2_object is None:
            nb_outputs = inputs.dims()[2]

            self._set_N2D2_object(self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                                         self.get_name(),
                                                                         nb_outputs,
                                                                         **self._optional_constructor_arguments))

            if 'activation' not in self._config_parameters:
                self._config_parameters['activation'] = \
                    n2d2.converter.from_N2D2_object(self._N2D2_object.getActivation())

            """Set and initialize here all complex cells members"""
            for key, value in self._config_parameters.items():
                if key is 'activation':
                    self._N2D2_object.setActivation(value.N2D2())
                else:
                    self._set_N2D2_parameter(self.python_to_n2d2_convention(key), value)


        self._add_to_graph(inputs)

        self._N2D2_object.propagate(self._inference)

        return self.get_outputs()



class Reshape(NeuralNetworkCell, Datatyped):

    _cell_constructors = {
        'Frame<float>': N2D2.ReshapeCell_Frame_float,
        'Frame_CUDA<float>': N2D2.ReshapeCell_Frame_CUDA_float,
    }
    _parameters = {
        "dims": "Dims",
    }
    _parameters.update(_cell_frame_parameters)

    _convention_converter= n2d2.ConventionConverter(_parameters)
    def __init__(self, dims, from_arguments=True, **config_parameters):
        """
        :param dims: dims of the new shape of the layer
        :type dims: list
        """

        if not from_arguments and (dims is not None or len(config_parameters) > 0):
            raise RuntimeError("from_arguments = True but not None constructor arguments")
        if from_arguments:
            self._create_from_arguments(dims, **config_parameters)
        

    def _create_from_arguments(self, dims, **config_parameters):
        if not isinstance(dims, list): # TODO : check the type of the elements ?
            raise n2d2.error_handler.WrongInputType("dims", str(type(dims)), ["list"])
        NeuralNetworkCell.__init__(self, **config_parameters)

        self._constructor_arguments.update({
            'dims': dims,
        })

        # No optional parameter
        self._parse_optional_arguments([])



    @classmethod
    def create_from_N2D2_object(cls, N2D2_object, n2d2_deepnet=None):

        n2d2_cell = cls(None, from_arguments=False)

        NeuralNetworkCell.__init__(n2d2_cell,
                                   name=N2D2_object.getName(),
                                   **cls.load_N2D2_parameters(N2D2_object))

        n2d2_cell._set_N2D2_object(N2D2_object)

        n2d2_cell._constructor_arguments['dims'] = n2d2_cell._N2D2_object.getDims()

        n2d2_cell._config_parameters['activation'] = \
            n2d2.converter.from_N2D2_object(n2d2_cell._N2D2_object.getActivation())

        if n2d2_deepnet is not None:
            n2d2_cell._deepnet = n2d2_deepnet
            n2d2_cell._sync_inputs_and_parents()
        else:
            n2d2_cell._deepnet = None
            n2d2_cell._N2D2_object.clearInputs()

        return n2d2_cell


    def __call__(self, inputs):
        super().__call__(inputs)

        if self._N2D2_object is None:
            nb_outputs = inputs.dims()[2]

            self._set_N2D2_object(self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                                         self.get_name(),
                                                                         nb_outputs,
                                                                         self._constructor_arguments['dims'],
                                                                         **self.n2d2_function_argument_parser(
                                                                             self._optional_constructor_arguments)))

            if 'activation' not in self._config_parameters:
                self._config_parameters['activation'] = \
                    n2d2.converter.from_N2D2_object(self._N2D2_object.getActivation())

            """Set and initialize here all complex cells members"""
            for key, value in self._config_parameters.items():
                if key is 'activation':
                    if value:
                        self._N2D2_object.setActivation(value.N2D2())
                else:
                    self._set_N2D2_parameter(self.python_to_n2d2_convention(key), value)

        self._add_to_graph(inputs)

        self._N2D2_object.propagate(self._inference)

        return self.get_outputs()


class Resize(NeuralNetworkCell):
    _cell_constructors = {
        'Frame': N2D2.ResizeCell_Frame,
        'Frame_CUDA': N2D2.ResizeCell_Frame_CUDA,
    }
    _parameters = {
        "align_corners": "AlignCorners",
    }
    _parameters.update(_cell_frame_parameters)

    _convention_converter = n2d2.ConventionConverter(_parameters)

    def __init__(self, outputs_width, outputs_height, resize_mode, from_arguments=True, **config_parameters):
        """
        :param dims: dims of the new shape of the layer
        :type dims: list
        """

        if not from_arguments and (outputs_width is not None or outputs_height is not None or
                                resize_mode is not None or len(config_parameters) > 0):
            raise RuntimeError("from_arguments = True but not None constructor arguments")
        if from_arguments:
            self._create_from_arguments(outputs_width, outputs_height, resize_mode, **config_parameters)

    def _create_from_arguments(self, outputs_width, outputs_height, resize_mode, **config_parameters):
        if not isinstance(outputs_width, int):
            raise n2d2.error_handler.WrongInputType("outputs_width", type(outputs_width), ["int"])
        if not isinstance(outputs_height, int):
            raise n2d2.error_handler.WrongInputType("outputs_height", type(outputs_height), ["int"])

        NeuralNetworkCell.__init__(self, **config_parameters)

        self._constructor_arguments.update({
            'outputs_width': outputs_width,
            'outputs_height': outputs_height,
            'resize_mode': N2D2.ResizeCell.ResizeMode.__members__[resize_mode],
        })

        # No optional parameter
        self._parse_optional_arguments([])

    @classmethod
    def create_from_N2D2_object(cls, N2D2_object, n2d2_deepnet=None):

        n2d2_cell = cls(None, None, None, from_arguments=False)

        NeuralNetworkCell.__init__(n2d2_cell,
                                   name=N2D2_object.getName(),
                                   **cls.load_N2D2_parameters(N2D2_object))

        n2d2_cell._set_N2D2_object(N2D2_object)

        n2d2_cell._constructor_arguments['outputs_width'] = n2d2_cell._N2D2_object.getResizeOutputWidth()
        n2d2_cell._constructor_arguments['outputs_height'] = n2d2_cell._N2D2_object.getResizeOutputHeight()
        n2d2_cell._constructor_arguments['resize_mode'] = n2d2_cell._N2D2_object.getMode()

        if n2d2_deepnet is not None:
            n2d2_cell._deepnet = n2d2_deepnet
            n2d2_cell._sync_inputs_and_parents()
        else:
            n2d2_cell._deepnet = None
            n2d2_cell._N2D2_object.clearInputs()

        return n2d2_cell

    def __call__(self, inputs):
        super().__call__(inputs)

        if self._N2D2_object is None:
            nb_outputs = inputs.dims()[2]

            self._set_N2D2_object(self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                                           self.get_name(),
                                                                           self._constructor_arguments['outputs_width'],
                                                                           self._constructor_arguments['outputs_height'],
                                                                           nb_outputs,
                                                                           self._constructor_arguments['resize_mode'],
                                                                           **self.n2d2_function_argument_parser(
                                                                               self._optional_constructor_arguments)))

            if 'activation' not in self._config_parameters:
                self._config_parameters['activation'] = \
                    n2d2.converter.from_N2D2_object(self._N2D2_object.getActivation())

            """Set and initialize here all complex cells members"""
            for key, value in self._config_parameters.items():
                if key is 'activation':
                    if value:
                        self._N2D2_object.setActivation(value.N2D2())
                else:
                    self._set_N2D2_parameter(self.python_to_n2d2_convention(key), value)

        self._add_to_graph(inputs)

        self._N2D2_object.propagate(self._inference)

        return self.get_outputs()





class Transpose(NeuralNetworkCell):
    _cell_constructors = {
        'Frame<float>': N2D2.TransposeCell_Frame_float,
        'Frame_CUDA<float>': N2D2.TransposeCell_Frame_CUDA_float,
    }
    _parameters = {}
    _parameters.update(_cell_frame_parameters)

    _convention_converter = n2d2.ConventionConverter(_parameters)

    def __init__(self, perm, from_arguments=True, **config_parameters):
        """
        :param dims: dims of the new shape of the layer
        :type dims: list
        """

        if not from_arguments and (perm is not None or len(config_parameters) > 0):
            raise RuntimeError("from_arguments = True but not None constructor arguments")
        if from_arguments:
            self._create_from_arguments(perm, **config_parameters)

    def _create_from_arguments(self, perm, **config_parameters):
        if not isinstance(perm, list):
            raise n2d2.error_handler.WrongInputType("outputs_width", type(perm), ["list"])

        NeuralNetworkCell.__init__(self, **config_parameters)

        self._constructor_arguments.update({
            'perm': perm,
        })

        # No optional parameter
        self._parse_optional_arguments([])

    @classmethod
    def create_from_N2D2_object(cls, N2D2_object, n2d2_deepnet=None):

        n2d2_cell = cls(None, from_arguments=False)

        NeuralNetworkCell.__init__(n2d2_cell,
                                   name=N2D2_object.getName(),
                                   **cls.load_N2D2_parameters(N2D2_object))

        n2d2_cell._set_N2D2_object(N2D2_object)

        n2d2_cell._constructor_arguments['perm'] = n2d2_cell._N2D2_object.getPermutation()

        if n2d2_deepnet is not None:
            n2d2_cell._deepnet = n2d2_deepnet
            n2d2_cell._sync_inputs_and_parents()
        else:
            n2d2_cell._deepnet = None
            n2d2_cell._N2D2_object.clearInputs()

        return n2d2_cell

    def __call__(self, inputs):
        super().__call__(inputs)

        if self._N2D2_object is None:
            nb_outputs = inputs.dims()[2]

            self._set_N2D2_object(self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                                           self.get_name(),
                                                                           nb_outputs,
                                                                           self._constructor_arguments['perm'],
                                                                           **self.n2d2_function_argument_parser(
                                                                               self._optional_constructor_arguments)))

            if 'activation' not in self._config_parameters:
                self._config_parameters['activation'] = \
                    n2d2.converter.from_N2D2_object(self._N2D2_object.getActivation())

            """Set and initialize here all complex cells members"""
            for key, value in self._config_parameters.items():
                if key is 'activation':
                    if value:
                        self._N2D2_object.setActivation(value.N2D2())
                else:
                    self._set_N2D2_parameter(self.python_to_n2d2_convention(key), value)

        self._add_to_graph(inputs)

        self._N2D2_object.propagate(self._inference)

        return self.get_outputs()