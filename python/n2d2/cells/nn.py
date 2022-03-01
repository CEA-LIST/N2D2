"""
    (C) Copyright 2021 CEA LIST. All Rights Reserved.
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
from n2d2.quantizer import ActivationQuantizer
import n2d2.solver
import n2d2.filler
from n2d2.n2d2_interface import N2D2_Interface
from n2d2.cells.cell import Cell, Trainable
from abc import ABC, abstractmethod
from n2d2.error_handler import deprecated
import n2d2.global_variables as gb
cuda_compiled = gb.cuda_compiled

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


class Mapable(ABC):

    @abstractmethod
    def __init__(self):
        pass



class NeuralNetworkCell(Cell, N2D2_Interface, ABC):

    @abstractmethod
    def __init__(self,  **config_parameters):
        if "activation" in config_parameters:
            if not isinstance(config_parameters["activation"], n2d2.activation.ActivationFunction):
                raise n2d2.error_handler.WrongInputType("activation", str(type(config_parameters["activation"])), [str(n2d2.activation.ActivationFunction)])
        else:
            config_parameters["activation"] = None
        if 'name' in config_parameters:
            name = config_parameters.pop('name')
        else:
            name = None # Set to None so that it can be created in Cell.__init__ 

        Cell.__init__(self, name)

        self._input_cells = []

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

        self.nb_input_cells = 0

    def __getattr__(self, key: str) -> None:
        if key is "name":
            return self.get_name()
        else:
            return N2D2_Interface.__getattr__(self, key)

    def __setattr__(self, key: str, value) -> None:
        
        if key is 'activation':
            if not (isinstance(value, n2d2.activation.ActivationFunction) or value is None):
                raise n2d2.error_handler.WrongInputType("activation", str(type(value)), [str(n2d2.activation.ActivationFunction), "None"])
            else:
                self._config_parameters["activation"] = value
                if self._N2D2_object:
                    if value:
                        self._N2D2_object.setActivation(value.N2D2())
                    else:
                        self._N2D2_object.setActivation(None)
        else:
            return super().__setattr__(key, value)

    @classmethod
    def _get_N2D2_complex_parameters(cls, N2D2_object):
        parameters = {}
        parameters['activation'] = \
            n2d2.converter.from_N2D2_object(N2D2_object.getActivation())
        return parameters

    @classmethod
    def create_from_N2D2_object(cls, N2D2_object, n2d2_deepnet=None):
        n2d2_cell = super().create_from_N2D2_object(N2D2_object)

        n2d2_cell._model = N2D2_object.getPyModel()
        if isinstance(n2d2_cell, Datatyped):
            Datatyped.__init__(n2d2_cell, datatype=N2D2_object.getPyDataType())
            n2d2_cell._model_key = n2d2_cell._model + '<' + n2d2_cell._datatype + '>'
        else:
            n2d2_cell._model_key = n2d2_cell._model

        n2d2_cell._input_cells = []

        n2d2_cell._name = N2D2_object.getName()
        n2d2_cell._inference = True
        if n2d2_deepnet is not None:
            n2d2_cell._deepnet = n2d2_deepnet
            n2d2_cell._sync_inputs_and_parents()
        else:
            n2d2_cell._deepnet = None
            n2d2_cell._N2D2_object.clearInputs()
        return n2d2_cell

    def learn(self):
        self._inference = False
        return self

    def test(self):
        self._inference = True
        return self

    def dims(self):
        if self.get_outputs().dims() == []:
            RuntimeError("Cell has no dims since it is not initialized yet.")
        return self.get_outputs().dims()

    def get_outputs(self):
        return n2d2.Tensor.from_N2D2(self._N2D2_object.getOutputs())._set_cell(self)

    def get_deepnet(self):
        return self._deepnet

    def set_deepnet(self, deepnet):
        self._deepnet = deepnet

    def clear_input_tensors(self):
        self._input_cells = []
        self._N2D2_object.clearInputTensors()


    def _check_tensor(self, inputs):
        if isinstance(inputs.cell, n2d2.cells.nn.NeuralNetworkCell) or isinstance(inputs.cell, n2d2.provider.Provider):
            # Check x-y dimension consistency
            if not isinstance(self, Fc):

                if not inputs.dims()[0:2] == self.N2D2().getInputsDims()[0:2]:
                    raise RuntimeError("Unmatching dims " + str(inputs.dims()[0:2])
                                       + " " + str(self.N2D2().getInputsDims()[0:2]))
        else:
            raise TypeError("Invalid inputs object of type " + str(type(inputs.cell)))

        # NOTE: This cannot really happen in current implementation
        if inputs.get_deepnet() is not self.get_deepnet():
            raise RuntimeError("The deepnet of the input doesn't match with the deepnet of the cell")

        return False 


    def add_input(self, inputs):

        initialized = not (self.dims() == [])
        
        if isinstance(inputs, n2d2.tensor.Interface):
            tensor_inputs = inputs.get_tensors()
        elif isinstance(inputs, n2d2.Tensor):
            tensor_inputs = [inputs]
        else:
            raise TypeError("Cannot add object of type " + str(type(inputs)))

        # Check input dimension consistency before connecting new inputs
        if initialized:
            # Check for input tensor element consistency
            if isinstance(inputs, n2d2.tensor.Interface):
                if len(inputs.get_tensors()) != self.nb_input_cells:
                    raise RuntimeError(
                        "Total number of input tensors != number inputs in cell '" + self.get_name() + "': " +
                        str(len(inputs.get_tensors())) + " vs. " + str(self.nb_input_cells))
            dim_z = 0
            for ipt in tensor_inputs:
                self._check_tensor(ipt)
                dim_z += ipt.dimZ()
            # Check for z dimension consistency
            if not dim_z == self.get_nb_channels():
                raise RuntimeError("Total number of input dimZ != cell '" + self.get_name() + "' number channels")

        parents = []

        # Clear old input tensors of cell to connect new inputs
        self.clear_input_tensors()
        for ipt in tensor_inputs:
            cell = ipt.cell
            self._N2D2_object.linkInput(cell.N2D2())

            if not initialized:
                self.nb_input_cells += 1

            if not isinstance(cell, n2d2.provider.Provider):
                parents.append(cell.N2D2())
            else:
                parents.append(None)
            self._input_cells.append(cell.get_name())

        self._deepnet.N2D2().addCell(self._N2D2_object, parents)
        if not initialized: 
            self._N2D2_object.initializeDataDependent()
            if isinstance(self, Mapable):
                if self._N2D2_object.getMapping().empty():
                    self._N2D2_object.setMapping(n2d2.Tensor([self.get_nb_outputs(), inputs.dimZ()],
                                                             datatype="bool", dim_format="N2D2").N2D2())

    def _add_to_graph(self, inputs):
        self.add_input(inputs)

    @deprecated(reason="You should use activation as a python attribute.")
    def set_activation(self, activation):
        """Set an activation function to the N2D2 object and update config parameter of the n2d2 object.
        
        :param activation: The activation function to set.
        :type activation: :py:class:`n2d2.activation.ActivationFunction`
        """
        if not isinstance(activation, n2d2.activation.ActivationFunction):
            raise n2d2.error_handler.WrongInputType("activation", activation, ["n2d2.activation.ActivationFunction"])
        print("Note: Replacing potentially existing activation in cells: " + self.get_name())
        self._config_parameters['activation'] = activation
        self._N2D2_object.setActivation(self._config_parameters['activation'].N2D2())
    
    @deprecated(reason="You should use activation as a python attribute.")
    def get_activation(self):
        if 'activation' in self._config_parameters:
            return self._config_parameters['activation']
        else:
            return None

    def get_inputs(self):
        return self._input_cells

    def clear_input(self):
        self._input_cells = []
        self._N2D2_object.clearInputs()

    def update(self):
        self._N2D2_object.update()

    def import_free_parameters(self, dir_name, ignore_not_exists=False):
        if self._N2D2_object:
            filename = dir_name + "/" + self.get_name() + ".syntxt"
            print("Import " + filename)
            self._N2D2_object.importFreeParameters(filename, ignore_not_exists)
            self._N2D2_object.importActivationParameters(dir_name, ignore_not_exists)

    def export_free_parameters(self, dir_name):
        if self._N2D2_object:
            filename = dir_name + "/" + self.get_name() + ".syntxt"
            print("Export to " + filename)
            self._N2D2_object.exportFreeParameters(filename)
            filename = dir_name + "/" + self.get_name() + "_quant.syntxt"
            print("Export to " + filename)
            self._N2D2_object.exportQuantFreeParameters(filename)
            self._N2D2_object.exportActivationParameters(dir_name)

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
        # NOTE: Sometimes parents contains [None], sometimes [].
        for idx, ipt in enumerate(parents):
            if ipt is not None:
                self._input_cells.append(parents[idx].getName())

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

    def __call__(self, x):
        super(NeuralNetworkCell, self).__call__(x)


class Fc(NeuralNetworkCell, Datatyped, Trainable):
    """
    Fully connected layer.
    """

    _cell_constructors = {
        'Frame<float>': N2D2.FcCell_Frame_float,
    }

    if cuda_compiled:
        _cell_constructors.update({
            'Frame_CUDA<float>': N2D2.FcCell_Frame_CUDA_float,
        })

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


    def __init__(self, nb_inputs, nb_outputs, nb_input_cells=1, **config_parameters):
        """
        :param nb_inputs: Number of inputs of the cells.
        :type nb_inputs: int
        :param nb_outputs: Number of outputs of the cells.
        :type nb_outputs: int
        :param name: Name fo the cells.
        :type name: str, optional
        :param activation: Activation function, default= None
        :type activation: :py:class:`n2d2.activation.ActivationFunction`, optional
        :param solver: Set the weights and bias solver, this parameter override parameters ``weights_solver`` and ``bias_solver``, default= :py:class:`n2d2.solver.SGD`
        :type solver: :py:class:`n2d2.solver.Solver`, optional
        :param weights_solver: Solver for weights, default= :py:class:`n2d2.solver.SGD`
        :type weights_solver: :py:class:`n2d2.solver.Solver`, optional
        :param bias_solver: Solver for biases, default= :py:class:`n2d2.filler.Normal`
        :type bias_solver: :py:class:`n2d2.solver.Solver`, optional
        :param filler: Set the weights and bias filler, this parameter override parameters ``weights_filler`` and ``bias_filler``, default= :py:class:`n2d2.filler.NormalFiller`
        :type filler: :py:class:`n2d2.filler.Filler`, optional
        :param weights_filler: Weights initial values filler, default= :py:class:`n2d2.filler.Normal`
        :type weights_filler: :py:class:`n2d2.filler.Filler`, optional
        :param bias_filler: Biases initial values filler, default= :py:class:`n2d2.filler.Normal`
        :type bias_filler: :py:class:`n2d2.filler.Filler`, optional
        :param no_bias: If ``True``, don’t use bias, default=False
        :type no_bias: bool, optional
        """
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

        Trainable.__init__(self)

        # Set and initialize here all complex cells members
        for key, value in self._config_parameters.items():
            if key is not "quantizer":
                self.__setattr__(key, value)
            
        self._N2D2_object.initializeParameters(nb_inputs, nb_input_cells)
        if 'quantizer' in self._config_parameters:
            self.quantizer = self._config_parameters["quantizer"]
        self.load_N2D2_parameters(self.N2D2())

    def __setattr__(self, key: str, value) -> None:
        if key is 'weights_solver':
            if isinstance(value, n2d2.solver.Solver):
                self._N2D2_object.resetWeightsSolver(value.N2D2())
                self._config_parameters["weights_solver"] = value
            else:
                raise n2d2.error_handler.WrongInputType("weights_solver", str(type(value)), [str(n2d2.solver.Solver)])
        elif key is 'bias_solver':
            if isinstance(value, n2d2.solver.Solver):
                self._N2D2_object.setBiasSolver(value.N2D2())
                self._config_parameters["bias_solver"] = value
            else:
                raise n2d2.error_handler.WrongInputType("bias_solver", str(type(value)), [str(n2d2.solver.Solver)])
        elif key is 'weights_filler':
            if isinstance(value, n2d2.filler.Filler):
                self._N2D2_object.setWeightsFiller(value.N2D2())
                self._config_parameters["weights_filler"] = value
            else:
                raise n2d2.error_handler.WrongInputType("weights_filler", str(type(value)), [str(n2d2.filler.Filler)])
        elif key is 'bias_filler':
            if isinstance(value, n2d2.filler.Filler):
                self._N2D2_object.setBiasFiller(value.N2D2())
                self._config_parameters["bias_filler"] = value
            else:
                raise n2d2.error_handler.WrongInputType("bias_filler", str(type(value)), [str(n2d2.filler.Filler)])
        elif key is 'quantizer':
            if isinstance(value, n2d2.quantizer.Quantizer):
                self._N2D2_object.setQuantizer(value.N2D2())
                self._N2D2_object.initializeWeightQuantizer()
                self._config_parameters["quantizer"] = value
                
            else:
                raise n2d2.error_handler.WrongInputType("quantizer", str(type(value)), [str(n2d2.quantizer.Quantizer)])
        elif key is 'filler':
            self.set_filler(value)
        elif key is 'solver':
            self.set_solver(value)
        else:
            return super().__setattr__(key, value)

    @classmethod
    def _get_N2D2_complex_parameters(cls, N2D2_object):
        parameter = super()._get_N2D2_complex_parameters(N2D2_object)
        parameter['weights_solver'] = \
            n2d2.converter.from_N2D2_object(N2D2_object.getWeightsSolver())
        parameter['bias_solver'] = \
            n2d2.converter.from_N2D2_object(N2D2_object.getBiasSolver())
        parameter['weights_filler'] = \
            n2d2.converter.from_N2D2_object(N2D2_object.getWeightsFiller())
        parameter['bias_filler'] = \
                n2d2.converter.from_N2D2_object(N2D2_object.getBiasFiller())
        parameter['quantizer'] = \
            n2d2.converter.from_N2D2_object(N2D2_object.getQuantizer())
        return parameter

    def _load_N2D2_constructor_parameters(self, N2D2_object):
        self._constructor_arguments.update({
            'nb_inputs': N2D2_object.getInputsSize(),
            'nb_outputs': N2D2_object.getNbOutputs(),
        })
        
    def _load_N2D2_optional_parameters(self, N2D2_object):
        # No optional paramaters !
        pass

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
        :type value: :py:class:`n2d2.Tensor`
        """
        if channel_index >= self.N2D2().getInputsSize():
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
        if channel_index >= self.N2D2().getInputsSize():
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
        for o in range(self.N2D2().getNbOutputs()):
            chan = []
            for c in range(self.N2D2().getInputsSize()):
                tensor = N2D2.Tensor_float([])
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
        if "no_bias" in self._config_parameters and self._config_parameters["no_bias"]:
            raise RuntimeError("You try to set a bias on " + self.get_name() +" but no_bias=True")
        if output_index >= self.N2D2().getNbOutputs():
            raise ValueError("Output index : " + str(output_index) + " must be < " + str(self.N2D2().getNbOutputs()) +")")
        self.N2D2().setBias(output_index, value.N2D2())

    def has_bias(self):
        return not self.no_bias

    def get_bias(self, output_index):
        """
        :param output_index: 
        :type output_index: int
        """
        if "no_bias" in self._config_parameters and self._config_parameters["no_bias"]:
            raise RuntimeError("You try to access a bias on " + self.get_name() +" but no_bias=True")
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
        if "no_bias" in self._config_parameters and self._config_parameters["no_bias"]:
            raise RuntimeError("You try to access a bias on " + self.get_name() +" but no_bias=True")
        biases = []
        for output_index in range(self.N2D2().getNbOutputs()):
            tensor = N2D2.Tensor_float([])
            self.N2D2().getBias(output_index, tensor)
            biases.append(n2d2.Tensor.from_N2D2(tensor))
        return biases

    def has_quantizer(self):
        return 'quantizer' in self._config_parameters


    def set_filler(self, filler, refill=False):
        """Set a filler for the weights and bias.

        :param filler: Filler object
        :type filler: :py:class:`n2d2.filler.Filler`
        """
        self.set_weights_filler(filler, refill=refill)
        self.set_bias_filler(filler, refill=refill)


    def set_bias_filler(self, filler, refill=False):
        """Set a filler for the bias.
       :param filler: Filler object
       :type filler: :py:class:`n2d2.filler.Filler`
       """
        if self.has_bias():
            if not isinstance(filler, n2d2.filler.Filler):
                raise n2d2.error_handler.WrongInputType("filler", str(type(filler)), ["n2d2.filler.Filler"])
            self._config_parameters['bias_filler'] = filler
            self._N2D2_object.setBiasFiller(self._config_parameters['bias_filler'].N2D2())
            if refill:
                self.refill_bias()
        else:
            raise RuntimeError("You try to set a bias filler on " + self.get_name() +" but no_bias=True")

    def set_weights_filler(self, filler, refill=False):
        """Set a filler for the weights.

        :param filler: Filler object
        :type filler: :py:class:`n2d2.filler.Filler`
        """
        if not isinstance(filler, n2d2.filler.Filler):
            raise n2d2.error_handler.WrongInputType("filler", str(type(filler)), ["n2d2.filler.Filler"])
        self._config_parameters['weights_filler'] = filler # No need to copy filler ?
        self._N2D2_object.setWeightsFiller(self._config_parameters['weights_filler'].N2D2())
        if refill:
            self.refill_weights()

    def refill_bias(self):
        """Re-fill the bias using the associated bias filler
        """
        self._N2D2_object.resetBias()
        
    def refill_weights(self):
        """Re-fill the weights using the associated weights filler
        """
        self._N2D2_object.resetWeights()

    def set_solver_parameter(self, key, value):
        """Set the parameter ``key`` with the value ``value`` for the attribute weight and bias solver.

        :param key: Parameter name
        :type key: str
        :param value: The value of the parameter
        :type value: Any
        """
        self._config_parameters['weights_solver'].set_parameter(key, value)
        self._config_parameters['bias_solver'].set_parameter(key, value)

    @deprecated(reason="You should use bias_solver as an attribute.")
    def get_bias_solver(self):
        return self._config_parameters['bias_solver']
    @deprecated(reason="You should use bias_solver as an attribute.")
    def set_bias_solver(self, solver):
        self._config_parameters['bias_solver'] = solver
        self._N2D2_object.setBiasSolver(self._config_parameters['bias_solver'].N2D2())
    @deprecated(reason="You should use weights_solver as an attribute.")
    def get_weights_solver(self):
        return self._config_parameters['weights_solver']
    @deprecated(reason="You should use weights_solver as an attribute.")
    def set_weights_solver(self, solver):
        self._config_parameters['weights_solver'] = solver
        self._N2D2_object.resetWeightsSolver(self._config_parameters['weights_solver'].N2D2())
    
    def set_solver(self, solver):
        """"Set the weights and bias solver with the same solver.

        :param solver: Solver object
        :type solver: :py:class:`n2d2.solver.Solver`
        """
        if not isinstance(solver, n2d2.solver.Solver):
            raise n2d2.error_handler.WrongInputType("solver", str(type(solver)), ["n2d2.solver.Solver"])
        self.bias_solver = solver.copy()
        self.weights_solver = solver.copy()


class Conv(NeuralNetworkCell, Datatyped, Trainable, Mapable):
    """
    Convolutional layer.
    """


    _cell_constructors = {
        'Frame<float>': N2D2.ConvCell_Frame_float,
    }

    if cuda_compiled:
        _cell_constructors.update({
            'Frame_CUDA<float>': N2D2.ConvCell_Frame_CUDA_float,
        })
    
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
                 **config_parameters):
        """
        :param nb_inputs: Number of inputs of the cells.
        :type nb_inputs: int
        :param nb_outputs: Number of output channels
        :type nb_outputs: int
        :param kernel_dims: Kernel dimension.
        :type kernel_dims: list
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
        :param activation: Activation function, default= None
        :type activation: :py:class:`n2d2.activation.ActivationFunction`, optional
        :param mapping: Mapping
        :type mapping: :py:class:`n2d2.Tensor`
        :param filler: Set the weights and bias filler, this parameter override parameters ``weights_filler`` and ``bias_filler``, default= :py:class:`n2d2.filler.NormalFiller`
        :type filler: :py:class:`n2d2.filler.Filler`, optional
        :param weights_filler: Weights initial values filler, default= :py:class:`n2d2.filler.Normal`
        :type weights_filler: :py:class:`n2d2.filler.Filler`, optional
        :param bias_filler: Biases initial values filler, default= :py:class:`n2d2.filler.Normal`
        :type bias_filler: :py:class:`n2d2.filler.Filler`, optional
        :param solver: Set the weights and bias solver, this parameter override parameters ``weights_solver`` and ``bias_solver``, default= :py:class:`n2d2.solver.SGD`
        :type solver: :py:class:`n2d2.solver.Solver`, optional
        :param weights_solver: Solver for weights
        :type weights_solver: :py:class:`n2d2.solver.Solver`, optional
        :param bias_solver: Solver for biases
        :type bias_solver: :py:class:`n2d2.solver.Solver`, optional
        :param no_bias: If ``True``, don’t use bias, default=False
        :type no_bias: bool, optional
        :param weights_export_flip: If ``True``, import/export flipped kernels, default=False
        :type weights_export_flip: bool, optional
        :param back_propagate: If ``True``, enable backpropagation, default=True
        :type back_propagate: bool, optional
        """
        if not isinstance(nb_inputs, int):
            raise n2d2.error_handler.WrongInputType("nb_inputs", str(type(nb_inputs)), ["int"])
        if not isinstance(nb_outputs, int):
            raise n2d2.error_handler.WrongInputType("nb_outputs", str(type(nb_outputs)), ["int"])
        if not isinstance(kernel_dims, list):
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
        
        Trainable.__init__(self)

        """Set and initialize here all complex cells members"""
        for key, value in self._config_parameters.items():
            if key is not "quantizer":
                self.__setattr__(key, value)
            
        self._N2D2_object.initializeParameters(nb_inputs, nb_input_cells)
        if 'quantizer' in self._config_parameters:
            self.quantizer = self._config_parameters["quantizer"]
        self.load_N2D2_parameters(self.N2D2())


    def __setattr__(self, key: str, value) -> None:
        if key is 'weights_solver':
            if isinstance(value, n2d2.solver.Solver):
                self._N2D2_object.resetWeightsSolver(value.N2D2())
                self._config_parameters["weights_solver"] = value
            else:
                raise n2d2.error_handler.WrongInputType("weights_solver", str(type(value)), [str(n2d2.solver.Solver)])
        elif key is 'bias_solver':
            if isinstance(value, n2d2.solver.Solver):
                self._N2D2_object.setBiasSolver(value.N2D2())
                self._config_parameters["bias_solver"] = value
            else:
                raise n2d2.error_handler.WrongInputType("bias_solver", str(type(value)), [str(n2d2.solver.Solver)])
        elif key is 'weights_filler':
            if isinstance(value, n2d2.filler.Filler):
                self._N2D2_object.setWeightsFiller(value.N2D2())
                self._config_parameters["weights_filler"] = value
            else:
                raise n2d2.error_handler.WrongInputType("weights_filler", str(type(value)), [str(n2d2.filler.Filler)])
        elif key is 'bias_filler':
            if isinstance(value, n2d2.filler.Filler):
                self._N2D2_object.setBiasFiller(value.N2D2())
                self._config_parameters["bias_filler"] = value
            else:
                raise n2d2.error_handler.WrongInputType("bias_filler", str(type(value)), [str(n2d2.filler.Filler)])
        elif key is 'quantizer':
            if isinstance(value, n2d2.quantizer.Quantizer):
                self._N2D2_object.setQuantizer(value.N2D2())
                self._N2D2_object.initializeWeightQuantizer()
                self._config_parameters["quantizer"] = value
            else:
                raise n2d2.error_handler.WrongInputType("quantizer", str(type(value)), [str(n2d2.quantizer.Quantizer)])
        elif key is 'mapping':
            if isinstance(value, n2d2.Tensor):
                self._N2D2_object.setMapping(value.N2D2())
            else:
                raise n2d2.error_handler.WrongInputType('mapping', type(value), [str(type(n2d2.Tensor))])
        elif key is 'filler':
            self.set_filler(value)
        elif key is 'solver':
            self.set_solver(value)
        else:
            return super().__setattr__(key, value)

    def _load_N2D2_constructor_parameters(self, N2D2_object):
        self._constructor_arguments.update({
            'nb_inputs':  N2D2_object.getNbChannels(),
            'nb_outputs':  N2D2_object.getNbOutputs(),
            'kernel_dims': [N2D2_object.getKernelWidth(), N2D2_object.getKernelHeight()]
        })
    
    def _load_N2D2_optional_parameters(self, N2D2_object):
        self._optional_constructor_arguments.update({
            'sub_sample_dims':  [N2D2_object.getSubSampleX(), N2D2_object.getSubSampleY()],
            'stride_dims':  [N2D2_object.getStrideX(), N2D2_object.getStrideY()],
            'padding_dims': [N2D2_object.getPaddingX(), N2D2_object.getPaddingY()],
            'dilation_dims': [N2D2_object.getDilationX(), N2D2_object.getDilationY()],
        })
    @classmethod
    def _get_N2D2_complex_parameters(cls, N2D2_object):
        parameter =  super()._get_N2D2_complex_parameters(N2D2_object)
        parameter['weights_solver'] = \
            n2d2.converter.from_N2D2_object(N2D2_object.getWeightsSolver())
        parameter['bias_solver'] = \
            n2d2.converter.from_N2D2_object(N2D2_object.getBiasSolver())
        parameter['weights_filler'] = \
            n2d2.converter.from_N2D2_object(N2D2_object.getWeightsFiller())
        parameter['bias_filler'] = \
            n2d2.converter.from_N2D2_object(N2D2_object.getBiasFiller())
        parameter['quantizer'] = \
            n2d2.converter.from_N2D2_object(N2D2_object.getQuantizer())
        return parameter

    def __call__(self, inputs):
        super().__call__(inputs)

        self._add_to_graph(inputs)

        self._N2D2_object.propagate(self._inference)

        return self.get_outputs()

    def set_filler(self, filler, refill=False):
        """Set a filler for the weights and bias.

        :param filler: Filler object
        :type filler: :py:class:`n2d2.filler.Filler`
        """
        self.set_weights_filler(filler, refill=refill)
        self.set_bias_filler(filler, refill=refill)
    
    def set_bias_filler(self, filler, refill=False):
        """Set a filler for the bias.

        :param filler: Filler object
        :type filler: :py:class:`n2d2.filler.Filler`
        """

        if not isinstance(filler, n2d2.filler.Filler):
            raise n2d2.error_handler.WrongInputType("filler", str(type(filler)), ["n2d2.filler.Filler"])
        self._config_parameters['bias_filler'] = filler
        self._N2D2_object.setBiasFiller(self._config_parameters['bias_filler'].N2D2())
        if refill:
            self.refill_bias()            

    def set_weights_filler(self, filler, refill=False):
        """Set a filler for the weights.

        :param filler: Filler object
        :type filler: :py:class:`n2d2.filler.Filler`
        """
        if not isinstance(filler, n2d2.filler.Filler):
            raise n2d2.error_handler.WrongInputType("filler", str(type(filler)), ["n2d2.filler.Filler"])
        self._config_parameters['weights_filler'] = filler # No need to copy filler ?
        self._N2D2_object.setWeightsFiller(self._config_parameters['weights_filler'].N2D2())
        if refill:
            self.refill_weights()

    def refill_bias(self):
        """Re-fill the bias using the associated bias filler
        """
        self._N2D2_object.resetBias()

    def refill_weights(self):
        """Re-fill the weights using the associated weights filler
        """
        self._N2D2_object.resetWeights()

    def set_solver_parameter(self, key, value):
        """Set the parameter ``key`` with the value ``value`` for the attribute weight and bias solver.

        :param key: Parameter name
        :type key: str
        :param value: The value of the parameter
        :type value: Any
        """
        self.weights_solver.set_parameter(key, value)
        self.bias_solver.set_parameter(key, value)

    @deprecated(reason="You should use weights_solver as an attribute.")
    def get_weights_solver(self):
        return self._config_parameters['weights_solver']
    @deprecated(reason="You should use bias_solver as an attribute.")
    def get_bias_solver(self):
        return self._config_parameters['bias_solver']
    @deprecated(reason="You should use weights_solver as an attribute.")
    def set_weights_solver(self, solver):
        self._config_parameters['weights_solver'] = solver
        self._N2D2_object.resetWeightsSolver(self._config_parameters['weights_solver'].N2D2())
    @deprecated(reason="You should use bias_solver as an attribute.")
    def set_bias_solver(self, solver):
        self._config_parameters['bias_solver'] = solver
        self._N2D2_object.setBiasSolver(self._config_parameters['bias_solver'].N2D2())

    def set_solver(self, solver):
        """"Set the weights and bias solver with the same solver.

        :param solver: Solver object
        :type solver: :py:class:`n2d2.solver.Solver`
        """
        if not isinstance(solver, n2d2.solver.Solver):
            raise n2d2.error_handler.WrongInputType("solver", str(type(solver)), ["n2d2.solver.Solver"])
        self.bias_solver = solver.copy()
        self.weights_solver = solver.copy()

    def has_quantizer(self):
        return 'quantizer' in self._config_parameters

    def set_weight(self, output_index, channel_index, value):
        """
        :param output_index: 
        :type output_index:
        :param channel_index:
        :type channel_index:
        :param value:
        :type value: :py:class:`n2d2.Tensor`
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
        for o in range(self.N2D2().getNbOutputs()):
            chan = []
            for c in range(self.N2D2().getNbChannels()):
                tensor = N2D2.Tensor_float([])
                self.N2D2().getWeight(o, c, tensor)
                chan.append(n2d2.Tensor.from_N2D2(tensor))
            weights.append(chan)
        return weights

    def has_bias(self):
        return not self._get_N2D2_parameters(self.N2D2())['no_bias']

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

class ConvDepthWise(Conv):

    def __init__(self,
                 nb_channel,
                 kernel_dims,
                 **config_parameters):
        if 'mapping' in config_parameters:
            raise RuntimeError('ConvDepthWise does not support custom mappings')
        else:
            config_parameters['mapping'] = n2d2.mapping.Mapping(nb_channels_per_group=1).create_mapping(
                nb_channel, nb_channel)
        Conv.__init__(self, nb_channel, nb_channel, kernel_dims, **config_parameters)


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
    }


    if cuda_compiled:
        _cell_constructors.update({
            'Frame_CUDA<float>': N2D2.SoftmaxCell_Frame_CUDA_float,
        })
    
    _parameters = {
        "with_loss": "withLoss",
        "group_size": "groupSize",
    }  
    _parameters.update(_cell_frame_parameters)

    _convention_converter= n2d2.ConventionConverter(_parameters)

    def __init__(self, **config_parameters):
        r"""
        :param nb_outputs: Number of output channels
        :type nb_outputs: int
        :param name: Name for the cells.
        :type name: str
        :param with_loss: :py:class:`Softmax` followed with a multinomial logistic layer, default=False
        :type with_loss: bool, optional
        :param group_size: :py:class:`Softmax` is applied on groups of outputs. The group size must be a divisor of ``nb_outputs`` parameter, default=0
        :type group_size: int, optional    
        """

        NeuralNetworkCell.__init__(self, **config_parameters)
        self._parse_optional_arguments(['with_loss', 'group_size'])

    def __setattr__(self, key: str, value) -> None:
        if key is 'with_loss':
            if isinstance(value, bool):
                self._N2D2_object.setWithLoss(value)
                self._optional_constructor_arguments["with_loss"] = value
            else:
                raise n2d2.error_handler.WrongInputType("with_loss", str(type(value)), ["bool"])
        else:
            return super().__setattr__(key, value)

    def _load_N2D2_constructor_parameters(self, N2D2_object):
        # No constructor parameters
        pass 
    
    def _load_N2D2_optional_parameters(self, N2D2_object):
        self._optional_constructor_arguments.update({
            'with_loss': N2D2_object.getWithLoss(),
            'group_size': N2D2_object.getGroupSize(),
        })

    @classmethod
    def _get_N2D2_complex_parameters(cls, N2D2_object):
        parameters = super()._get_N2D2_complex_parameters(N2D2_object)
        return parameters

    def __call__(self, inputs):
        super().__call__(inputs)

        if self._N2D2_object is None:
            nb_outputs = inputs.dims()[2]

            self._set_N2D2_object(self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                                         self.get_name(),
                                                                         nb_outputs,
                                                                         **self.n2d2_function_argument_parser(self._optional_constructor_arguments)))

            """Set and initialize here all complex cells members"""
            for key, value in self._config_parameters.items():
                self.__setattr__(key, value)
            self.load_N2D2_parameters(self.N2D2())

        self._add_to_graph(inputs)

        self._N2D2_object.propagate(self._inference)

        return self.get_outputs()


class Pool(NeuralNetworkCell, Datatyped, Mapable):
    '''
    Pooling layer.
    '''

    _cell_constructors = {
        'Frame<float>': N2D2.PoolCell_Frame_float,
    }
    if cuda_compiled:
        _cell_constructors.update({
            'Frame_CUDA<float>': N2D2.PoolCell_Frame_CUDA_float,
        })

    _parameters = {
        "pool_dims": "poolDims",
        "stride_dims": "strideDims",
        "padding_dims": "paddingDims",
        "pooling": "pooling",
        "ext_padding_dims": "ExtPaddingDims",
    }  
    _parameters.update(_cell_frame_parameters)

    _convention_converter = n2d2.ConventionConverter(_parameters)

    def __init__(self,
                 pool_dims,
                 **config_parameters):
        """
        :param pool_dims: Pooling area dimensions
        :type pool_dims: list
        :param name: Name for the cells.
        :type name: str
        :param pooling: Type of pooling (``Max`` or ``Average``), default="Max" 
        :type pooling: str, optional
        :param stride_dims: Dimension of the stride of the kernel.
        :type stride_dims: list, optional
        :param padding_dims: Dimensions of the padding.
        :type padding_dims: list, optional
        :param activation: Activation function, default= None
        :type activation: :py:class:`n2d2.activation.ActivationFunction`, optional
        :param mapping: Mapping
        :type mapping: :py:class:`n2d2.Tensor`, optional
        """
        if not isinstance(pool_dims, list):
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
    
    def _load_N2D2_constructor_parameters(self, N2D2_object):
        self._constructor_arguments['pool_dims'] = [N2D2_object.getPoolWidth(),
                                                        N2D2_object.getPoolHeight()]
    
    def _load_N2D2_optional_parameters(self, N2D2_object):
        self._optional_constructor_arguments.update({
            'stride_dims': [N2D2_object.getStrideX(), N2D2_object.getStrideY()],
            'padding_dims': [N2D2_object.getPaddingX(), N2D2_object.getPaddingY()],
            'pooling': N2D2_object.getPooling(),
        })

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
            
            """Set and initialize here all complex cells members"""
            for key, value in self._config_parameters.items():
                self.__setattr__(key, value)

            self.load_N2D2_parameters(self.N2D2())

        self._add_to_graph(inputs)
        
        self._N2D2_object.propagate(self._inference)

        return self.get_outputs()

    def __setattr__(self, key: str, value) -> None:
        if key is 'mapping':
            if not isinstance(value, n2d2.Tensor):
                raise n2d2.error_handler.WrongInputType('mapping', type(value), [str(type(n2d2.Tensor))])
            if value.dimX() != value.dimY():
                raise ValueError("Pool Cell supports only unit maps")
            self._N2D2_object.setMapping(value.N2D2())
        else:
            return super().__setattr__(key, value)


class Pool2d(Pool):
    """
    'Standard' pooling where all feature maps are pooled independently.
    """

    _cell_constructors = {
        'Frame<float>': N2D2.PoolCell_Frame_float,
    }
    if cuda_compiled:
        _cell_constructors.update({
            'Frame_CUDA<float>': N2D2.PoolCell_Frame_CUDA_float,
        })
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
                 **config_parameters):
        """
        :param pool_dims: Pooling area dimensions
        :type pool_dims: list
        :param name: Name for the cells.
        :type name: str
        :param pooling: Type of pooling (``Max`` or ``Average``), default="Max"
        :type pooling: str, optional
        :param stride_dims: Dimension of the stride of the kernel.
        :type stride_dims: list, optional
        :param padding_dims: Dimensions of the padding.
        :type padding_dims: list, optional
        :param activation: Activation function, default= None
        :type activation: :py:class:`n2d2.activation.ActivationFunction`, optional
        """
        if 'mapping' in config_parameters:
            raise RuntimeError('Pool2d does not support custom mappings')
        Pool.__init__(self, pool_dims, **config_parameters)

    def __call__(self, inputs):
        NeuralNetworkCell.__call__(self, inputs)

        if self._N2D2_object is None:

            self._set_N2D2_object(self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                                         self.get_name(),
                                                                         self._constructor_arguments['pool_dims'],
                                                                         inputs.dims()[2],
                                                                         **self.n2d2_function_argument_parser(self._optional_constructor_arguments)))

            """Set and initialize here all complex cells members"""
            for key, value in self._config_parameters.items():
                self.__setattr__(key, value)

            self._N2D2_object.setMapping(
                n2d2.mapping.Mapping(nb_channels_per_group=1).create_mapping(inputs.dims()[2],
                                                                             inputs.dims()[2]).N2D2())
            self.load_N2D2_parameters(self.N2D2())

        self._add_to_graph(inputs)

        self._N2D2_object.propagate(self._inference)

        return self.get_outputs()


class GlobalPool2d(Pool2d):
    """
    Global 2d pooling on full spatial dimension of input. Before the first call, the pooling
    dimension will be an empty list, which will be filled with the inferred dimensions after
    the first call.
    """
    _cell_constructors = {
        'Frame<float>': N2D2.PoolCell_Frame_float,
    }
    if cuda_compiled:
        _cell_constructors.update({
            'Frame_CUDA<float>': N2D2.PoolCell_Frame_CUDA_float,
        })

    _parameters = {
        "pooling": "pooling",
    }
    _parameters.update(_cell_frame_parameters)

    _convention_converter= n2d2.ConventionConverter(_parameters)

    def __init__(self, **config_parameters):
        """
        :param name: Name for the cells.
        :type name: str
        :param activation: Activation function, default= None
        :type activation: :py:class:`n2d2.activation.ActivationFunction`, optional
        """
        if 'pool_dims' in config_parameters:
            raise RuntimeError('GlobalPool2d does not support custom pool dims')
        Pool2d.__init__(self, [], **config_parameters)


    def __call__(self, inputs):
        NeuralNetworkCell.__call__(self, inputs)

        if self._N2D2_object is None:

            self._set_N2D2_object(self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                                         self.get_name(),
                                                                         [inputs.dims()[0], inputs.dims()[1]],
                                                                         inputs.dims()[2],
                                                                         strideDims=[1, 1],
                                                                         **self.n2d2_function_argument_parser(self._optional_constructor_arguments)))

            """Set and initialize here all complex cells members"""
            for key, value in self._config_parameters.items():
                self.__setattr__(key, value)

            self._N2D2_object.setMapping(n2d2.mapping.Mapping(nb_channels_per_group=1).create_mapping(inputs.dims()[2], inputs.dims()[2]).N2D2())
            self.load_N2D2_parameters(self.N2D2())

        self._add_to_graph(inputs)

        self._N2D2_object.propagate(self._inference)

        return self.get_outputs()


class Deconv(NeuralNetworkCell, Datatyped, Trainable, Mapable):
    """
    Deconvolution layer.
    """
    _cell_constructors = {
        'Frame<float>': N2D2.DeconvCell_Frame_float,
    }
    if cuda_compiled:
        _cell_constructors.update({
            'Frame_CUDA<float>': N2D2.DeconvCell_Frame_CUDA_float,
        })
    _parameters = {
        "no_bias": "NoBias",
        "back_propagate": "BackPropagate",
        "weights_export_format": "WeightsExportFormat",
        "weights_export_flip": "WeightsExportFlip",
        "outputs_remap": "OutputsRemap",
        "kernel_dims": "kernelDims",
        "sub_sample_dims": "subSampleDims",
        "stride_dims": "strideDims",
        "padding_dims": "paddingDims",
        "dilation_dims": "dilationDims",
        "ext_padding_dims":" ExtPaddingDims",
        "weights_filler": "WeightsFiller",
        "bias_filler": "BiasFiller",
        "weights_solver": "WeightsSolver",
        "bias_solver": "BiasSolver",
    }
    _parameters.update(_cell_frame_parameters)

    _convention_converter= n2d2.ConventionConverter(_parameters)

    def __init__(self,
                 nb_inputs,
                 nb_outputs,
                 kernel_dims,
                 nb_input_cells=1,
                 **config_parameters):
        """
        :param nb_inputs: Number of inputs of the cells.
        :type nb_inputs: int
        :param nb_outputs: Number of output channels
        :type nb_outputs: int
        :param kernel_dims: Kernel dimension.
        :type kernel_dims: list
        :param name: Name for the cells.
        :type name: str
        :param stride_dims: Dimension of the stride of the kernel.
        :type stride_dims: list, optional
        :param padding_dims: Dimensions of the padding.
        :type padding_dims: list, optional
        :param dilation_dims: Dimensions of the dilation of the kernels 
        :type dilation_dims: list, optional
        :param activation: Activation function, default= None
        :type activation: :py:class:`n2d2.activation.ActivationFunction`, optional
        :param filler: Set the weights and bias filler, this parameter override parameters ``weights_filler`` and ``bias_filler``, default= :py:class:`n2d2.filler.NormalFiller`
        :type filler: :py:class:`n2d2.filler.Filler`, optional
        :param weights_filler: Weights initial values filler, default= :py:class:`n2d2.filler.NormalFiller`
        :type weights_filler: :py:class:`n2d2.filler.Filler`, optional
        :param bias_filler: Biases initial values filler, default= :py:class:`n2d2.filler.NormalFiller`
        :type bias_filler: :py:class:`n2d2.filler.Filler`, optional
        :param solver: Set the weights and bias solver, this parameter override parameters ``weights_solver`` and ``bias_solver``, default= :py:class:`n2d2.solver.SGD`
        :type solver: :py:class:`n2d2.solver.Solver`, optional
        :param weights_solver: Solver for weights, default= :py:class:`n2d2.solver.SGD`
        :type weights_solver: :py:class:`n2d2.solver.Solver`, optional
        :param bias_solver: Solver for biases, default= :py:class:`n2d2.solver.SGD`
        :type bias_solver: :py:class:`n2d2.solver.Solver`, optional
        :param no_bias: If ``True``, don’t use bias, default=False
        :type no_bias: bool, optional
        :param back_propagate: If ``True``, enable backpropagation, default=True
        :type back_propagate: bool, optional
        :param weights_export_flip: If ``True``, import/export flipped kernels, default=False
        :type weights_export_flip: bool, optional
        :param mapping: Mapping
        :type mapping: :py:class:`n2d2.Tensor`, optional
        """
        if not isinstance(nb_inputs, int):
            raise n2d2.error_handler.WrongInputType("nb_inputs", str(type(nb_inputs)), ["int"])
        if not isinstance(nb_outputs, int):
            raise n2d2.error_handler.WrongInputType("nb_outputs", str(type(nb_outputs)), ["int"])
        if not isinstance(kernel_dims, list):
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
        
        Trainable.__init__(self)

        """Set and initialize here all complex cells members"""
        for key, value in self._config_parameters.items():
            self.__setattr__(key, value)

        self._N2D2_object.initializeParameters(nb_inputs, nb_input_cells)
        self.load_N2D2_parameters(self.N2D2())

    def __setattr__(self, key: str, value) -> None:
        if key is 'weights_solver':
            if isinstance(value, n2d2.solver.Solver):
                self._N2D2_object.resetWeightsSolver(value.N2D2())
                self._config_parameters["weights_solver"] = value
            else:
                raise n2d2.error_handler.WrongInputType("weights_solver", str(type(value)), [str(n2d2.solver.Solver)])
        elif key is 'bias_solver':
            if isinstance(value, n2d2.solver.Solver):
                self._N2D2_object.setBiasSolver(value.N2D2())
                self._config_parameters["bias_solver"] = value
            else:
                raise n2d2.error_handler.WrongInputType("bias_solver", str(type(value)), [str(n2d2.solver.Solver)])
        elif key is 'weights_filler':
            if isinstance(value, n2d2.filler.Filler):
                self._N2D2_object.setWeightsFiller(value.N2D2())
                self._config_parameters["weights_filler"] = value
            else:
                raise n2d2.error_handler.WrongInputType("weights_filler", str(type(value)), [str(n2d2.filler.Filler)])
        elif key is 'bias_filler':
            if isinstance(value, n2d2.filler.Filler):
                self._N2D2_object.setBiasFiller(value.N2D2())
                self._config_parameters["bias_filler"] = value
            else:
                raise n2d2.error_handler.WrongInputType("bias_filler", str(type(value)), [str(n2d2.filler.Filler)])
        elif key is 'mapping':
            if isinstance(value, n2d2.Tensor):
                self._N2D2_object.setMapping(value.N2D2())
            else:
                raise n2d2.error_handler.WrongInputType('mapping', type(value), [str(type(n2d2.Tensor))])
        elif key is 'filler':
            self.set_filler(value)
        elif key is 'solver':
            self.set_solver(value)
        else:
            return super().__setattr__(key, value)

    def _load_N2D2_constructor_parameters(self, N2D2_object):
        self._constructor_arguments.update({
            'nb_inputs':  N2D2_object.getNbChannels(),
            'nb_outputs':  N2D2_object.getNbOutputs(),
            'kernel_dims': [N2D2_object.getKernelWidth(), N2D2_object.getKernelHeight()]
        })
    
    def _load_N2D2_optional_parameters(self, N2D2_object):
        self._optional_constructor_arguments.update({
            'stride_dims':  [N2D2_object.getStrideX(), N2D2_object.getStrideY()],
            'padding_dims': [N2D2_object.getPaddingX(), N2D2_object.getPaddingY()],
            'dilation_dims': [N2D2_object.getDilationX(), N2D2_object.getDilationY()],
        })
    @classmethod
    def _get_N2D2_complex_parameters(cls, N2D2_object):
        parameter =  super()._get_N2D2_complex_parameters(N2D2_object)
        parameter['weights_solver'] = \
            n2d2.converter.from_N2D2_object(N2D2_object.getWeightsSolver())
        parameter['bias_solver'] = \
            n2d2.converter.from_N2D2_object(N2D2_object.getBiasSolver())
        parameter['weights_filler'] = \
            n2d2.converter.from_N2D2_object(N2D2_object.getWeightsFiller())
        parameter['bias_filler'] = \
            n2d2.converter.from_N2D2_object(N2D2_object.getBiasFiller())

        return parameter

    def __call__(self, inputs):
        super().__call__(inputs)

        self._add_to_graph(inputs)

        self._N2D2_object.propagate(self._inference)

        return self.get_outputs()

    def set_solver(self, solver):
        """"Set the weights and bias solver with the same solver.

        :param solver: Solver object
        :type solver: :py:class:`n2d2.solver.Solver`
        """
        if not isinstance(solver, n2d2.solver.Solver):
            raise n2d2.error_handler.WrongInputType("solver", str(type(solver)), ["n2d2.solver.Solver"])
        self.bias_solver = solver.copy()
        self.weights_solver= solver.copy()
    
    def set_filler(self, filler, refill=False):
        """Set a filler for the weights and bias.

        :param filler: Filler object
        :type filler: :py:class:`n2d2.filler.Filler`
        """
        if not isinstance(filler, n2d2.filler.Filler):
            raise n2d2.error_handler.WrongInputType("filler", str(type(filler)), ["n2d2.filler.Filler"])
        self.weights_filler = filler
        self.bias_filler = filler
        if refill:
            self.refill_bias()
            self.refill_weights()
    def set_weights_filler(self, solver, refill=False):
        self._config_parameters['weights_filler'] = solver
        self._N2D2_object.setWeightsFiller(self._config_parameters['weights_filler'].N2D2())
        if refill:
            self.refill_weights()
    def set_bias_filler(self, solver, refill=False):
        self._config_parameters['bias_filler'] = solver
        self._N2D2_object.setBiasFiller(self._config_parameters['bias_filler'].N2D2())
        if refill:
            self.refill_weights()

    @deprecated(reason="You should use weights_filler as an argument")
    def get_weights_filler(self):
        return self._config_parameters['weights_filler']
    @deprecated(reason="You should use bias_filler as an argument")
    def get_bias_filler(self):
        return self._config_parameters['bias_filler']
    @deprecated(reason="You should use weights_solver as an argument")
    def get_weights_solver(self):
        return self._config_parameters['weights_solver']
    @deprecated(reason="You should use bias_solver as an argument")
    def get_bias_solver(self):
        return self._config_parameters['bias_solver']
    @deprecated(reason="You should use weights_solver as an argument")
    def set_weights_solver(self, solver):
        self._config_parameters['weights_solver'] = solver
        self._N2D2_object.resetWeightsSolver(self._config_parameters['weights_solver'].N2D2())
    @deprecated(reason="You should use bias_solver as an argument")
    def set_bias_solver(self, solver):
        self._config_parameters['bias_solver'] = solver
        self._N2D2_object.setBiasSolver(self._config_parameters['bias_solver'].N2D2())

    def refill_bias(self):
        """Re-fill the bias using the associated bias filler
        """
        self._N2D2_object.resetBias()
    def refill_weights(self):
        """Re-fill the weights using the associated weights filler
        """
        self._N2D2_object.resetWeights()   

    def set_weight(self, output_index, channel_index, value):
        """
        :param output_index: 
        :type output_index:
        :param channel_index:
        :type channel_index:
        :param value:
        :type value: :py:class:`n2d2.Tensor`
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
        for o in range(self.N2D2().getNbOutputs()):
            chan = []
            for c in range(self.N2D2().getNbChannels()):
                tensor = N2D2.Tensor_float([])
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

    def has_bias(self):
        return not self.no_bias

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

    def has_quantizer(self):
        return False

class ElemWise(NeuralNetworkCell):
    """
    Element-wise operation layer.
    """

    _cell_constructors = {
        'Frame': N2D2.ElemWiseCell_Frame,
    }

    if cuda_compiled:
        _cell_constructors.update({
            'Frame_CUDA': N2D2.ElemWiseCell_Frame_CUDA,
        })
    _parameters = {
        "operation": "operation",
        "mode": "mode",
        "weights": "weights",
        "shifts": "shifts"
    }  
    _parameters.update(_cell_frame_parameters)

    _convention_converter = n2d2.ConventionConverter(_parameters)

    _parameter_loaded = True # boolean to indicate if parameters have been loaded.

    def __init__(self, **config_parameters):
        """
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

        NeuralNetworkCell.__init__(self, **config_parameters)

        self._parse_optional_arguments(['operation', 'mode', 'weights', 'shifts'])
        
        if "operation" in self._optional_constructor_arguments:
            operation = self._optional_constructor_arguments["operation"]
            if not isinstance(operation, str):
                raise n2d2.error_handler.WrongInputType("operation", str(type(operation)), ["str"])
            if operation not in N2D2.ElemWiseCell.Operation.__members__.keys():
                raise n2d2.error_handler.WrongValue("operation", operation,
                                                    ", ".join(N2D2.ElemWiseCell.Operation.__members__.keys()))
            self._optional_constructor_arguments['operation'] = \
                N2D2.ElemWiseCell.Operation.__members__[self._optional_constructor_arguments['operation']]
        if "mode" in self._optional_constructor_arguments:
            mode = self._optional_constructor_arguments["mode"]
            if not isinstance(mode, str):
                raise n2d2.error_handler.WrongInputType("mode", str(type(mode)), ["str"])
            if mode not in N2D2.ElemWiseCell.CoeffMode.__members__.keys():
                raise n2d2.error_handler.WrongValue("mode", mode,
                                                    ", ".join(N2D2.ElemWiseCell.CoeffMode.__members__.keys()))
            self._optional_constructor_arguments['mode'] = \
                N2D2.ElemWiseCell.CoeffMode.__members__[self._optional_constructor_arguments['mode']]
        if "weights" in self._optional_constructor_arguments:
            if not isinstance(self._optional_constructor_arguments["weights"], list):
                raise n2d2.error_handler.WrongInputType("weights", str(type(self._optional_constructor_arguments["weights"])), ["float"])
        if "shifts" in self._optional_constructor_arguments:
            if not isinstance(self._optional_constructor_arguments["shifts"], list):
                raise n2d2.error_handler.WrongInputType("shifts", str(type(self._optional_constructor_arguments["shifts"])), ["float"])
    
    def _load_N2D2_optional_parameters(self, N2D2_object):
        self._optional_constructor_arguments['operation'] = N2D2_object.getOperation()
        self._optional_constructor_arguments['mode'] = N2D2_object.getCoeffMode()
        self._optional_constructor_arguments['weights'] = N2D2_object.getWeights()
        self._optional_constructor_arguments['shifts'] = N2D2_object.getShifts()

    def __call__(self, inputs):
        super().__call__(inputs)

        if self._N2D2_object is None:

            mapping_row = 0

            if isinstance(inputs, n2d2.tensor.Interface):
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
            """Set and initialize here all complex cells members"""
            for key, value in self._config_parameters.items():
                self.__setattr__(key, value)
            self._parameter_loaded = False
        
        self._add_to_graph(inputs)

        self._N2D2_object.propagate(self._inference)
        
        if not self._parameter_loaded:
            # ElemWise initialize weights and shift after propagation
            self.load_N2D2_parameters(self.N2D2())

        return self.get_outputs()

class Dropout(NeuralNetworkCell, Datatyped):
    """
    Dropout layer :cite:`Srivastava2014`.
    """
    _type = "Dropout"

    _cell_constructors = {
        'Frame<float>': N2D2.DropoutCell_Frame_float,
    }
    if cuda_compiled:
        _cell_constructors.update({
            'Frame_CUDA<float>': N2D2.DropoutCell_Frame_CUDA_float,
        })
    _parameters = {
        "dropout": "Dropout",
    }  
    _parameters.update(_cell_frame_parameters)

    _convention_converter= n2d2.ConventionConverter(_parameters)

    def __init__(self, **config_parameters):
        """
        :param name: Name for the cells.
        :type name: str
        :param dropout: The probability with which the value from input would be dropped, default=0.5
        :type dropout: float, optional
        """
        NeuralNetworkCell.__init__(self, **config_parameters)

        self._parse_optional_arguments([])
        if "dropout" in config_parameters:
            if not isinstance(config_parameters["dropout"], float):
                raise n2d2.error_handler.WrongInputType("dropout", str(type(config_parameters["dropout"])), ["float"])

        
    def __call__(self, inputs):
        super().__call__(inputs)

        if self._N2D2_object is None:
            nb_outputs = inputs.dims()[2]

            self._set_N2D2_object(self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                                         self.get_name(),
                                                                         nb_outputs,
                                                                         **self.n2d2_function_argument_parser(self._optional_constructor_arguments)))
            """Set and initialize here all complex cells members"""
            for key, value in self._config_parameters.items():
                self.__setattr__(key, value)

            self.load_N2D2_parameters(self.N2D2())

        self._add_to_graph(inputs)

        self._N2D2_object.propagate(self._inference)

        return self.get_outputs()




class Padding(NeuralNetworkCell):

    _cell_constructors = {
        'Frame': N2D2.PaddingCell_Frame,
    }    
    if cuda_compiled:
        _cell_constructors.update({
            'Frame_CUDA': N2D2.PaddingCell_Frame_CUDA,
        })
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

    def _load_N2D2_constructor_parameters(self, N2D2_object):
        self._constructor_arguments['top_pad'] = N2D2_object.getTopPad()
        self._constructor_arguments['bot_pad'] = N2D2_object.getBotPad()
        self._constructor_arguments['left_pad'] = N2D2_object.getLeftPad()
        self._constructor_arguments['right_pad'] = N2D2_object.getRightPad()

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
            """Set and initialize here all complex cells members"""
            for key, value in self._config_parameters.items():
                self.__setattr__(key, value)

            self.load_N2D2_parameters(self.N2D2())

        self._add_to_graph(inputs)

        self._N2D2_object.propagate(self._inference)

        return self.get_outputs()


class BatchNorm2d(NeuralNetworkCell, Datatyped, Trainable):

    _cell_constructors = {
        'Frame<float>': N2D2.BatchNormCell_Frame_float,
    }
    if cuda_compiled:
        _cell_constructors.update({
            'Frame_CUDA<float>': N2D2.BatchNormCell_Frame_CUDA_float,
        })
    _parameters = {
        "nb_inputs": "NbInputs",
        "scale_solver": "ScaleSolver",
        "bias_solver": "BiasSolver",
        "moving_average_momentum": "MovingAverageMomentum",
        "epsilon": "Epsilon",
        "back_propagate": "BackPropagate"
    }
    _parameters.update(_cell_frame_parameters)

    _convention_converter = n2d2.ConventionConverter(_parameters)

    def __init__(self, nb_inputs, nb_input_cells=1, **config_parameters):
        """
        :param nb_inputs: Number of intput neurons
        :type nb_inputs: int
        :param solver: Set the scale and bias solver, this parameter override parameters ``scale_solver`` and bias_solver``, default= :py:class:`n2d2.solver.SGD`
        :type solver: :py:class:`n2d2.solver.Solver`, optional
        :param scale_solver: Scale solver parameters, default= :py:class:`n2d2.solver.SGD`
        :type scale_solver: :py:class:`n2d2.solver.Solver`, optional
        :param bias_solver: Bias  solver parameters, default= :py:class:`n2d2.solver.SGD`
        :type bias_solver: :py:class:`n2d2.solver.Solver`, optional
        :param epsilon: Epsilon value used in the batch normalization formula. If ``0.0``, automatically choose the minimum possible value, default=0.0
        :type epsilon: float, optional
        :param moving_average_momentum: Moving average rate: used for the moving average of batch-wise means and standard deviations during training.The closer to ``1.0``, the more it will depend on the last batch. 
        :type moving_average_momentum: float, optional
        """
        if not isinstance(nb_inputs, int):
            raise n2d2.error_handler.WrongInputType("nb_inputs", str(type(nb_inputs)), ["int"])

        NeuralNetworkCell.__init__(self, **config_parameters)
        # No optional parameter
        self._parse_optional_arguments([])
        self._set_N2D2_object(self._cell_constructors[self._model_key](N2D2.DeepNet(n2d2.global_variables.default_net),
                                                self.get_name(),
                                                nb_inputs,
                                                **self.n2d2_function_argument_parser(self._optional_constructor_arguments)))

        Trainable.__init__(self)


        """Set and initialize here all complex cells members"""
        for key, value in self._config_parameters.items():
            self.__setattr__(key, value)

        self._N2D2_object.initializeParameters(nb_inputs, nb_input_cells)
        self.load_N2D2_parameters(self.N2D2())

    def __setattr__(self, key: str, value) -> None:
        if key is 'scale_solver':
            if not isinstance(value, n2d2.solver.Solver):
                raise n2d2.error_handler.WrongInputType("scale_solver", str(type(value)), [str(n2d2.solver.Solver)])
            self._N2D2_object.setScaleSolver(value.N2D2())
            self._config_parameters["scale_solver"] = value
        elif key is 'bias_solver':
            if not isinstance(value, n2d2.solver.Solver):
                raise n2d2.error_handler.WrongInputType("bias_solver", str(type(value)), [str(n2d2.solver.Solver)])
            self._N2D2_object.setBiasSolver(value.N2D2())
            self._config_parameters["bias_solver"] = value
        elif key is 'solver':
            self.set_solver(value)
        else:
            return super().__setattr__(key, value)

    def _load_N2D2_constructor_parameters(self, N2D2_object):
        self._constructor_arguments.update({
            'nb_inputs':  N2D2_object.getNbChannels(),
        })
    
    @classmethod
    def _get_N2D2_complex_parameters(cls, N2D2_object):
        parameter =  super()._get_N2D2_complex_parameters(N2D2_object)
        parameter['scale_solver'] = \
            n2d2.converter.from_N2D2_object(N2D2_object.getScaleSolver())
        parameter['bias_solver'] = \
            n2d2.converter.from_N2D2_object(N2D2_object.getBiasSolver())
        return parameter

    def __call__(self, inputs):
        if self._constructor_arguments["nb_inputs"] != inputs.dimZ():
            raise ValueError(self.get_name() + " : expected an input with " + str(self._constructor_arguments["nb_inputs"]) + " channels got a tensor with " + str(inputs.dimZ()) + " instead.")
        super().__call__(inputs)

        self._add_to_graph(inputs)

        self._N2D2_object.propagate(self._inference)

        return self.get_outputs()

    def has_bias(self):
        return True
    
    @deprecated(reason="You should use scale_solver as an attribute.")
    def set_scale_solver(self, solver):
        self._config_parameters['scale_solver'] = solver
        self._N2D2_object.setScaleSolver(self._config_parameters['scale_solver'].N2D2())
    
    @deprecated(reason="You should use bias_solver as an attribute.")
    def set_bias_solver(self, solver):
        self._config_parameters['bias_solver'] = solver
        self._N2D2_object.setBiasSolver(self._config_parameters['bias_solver'].N2D2())

    def get_biases(self) -> n2d2.Tensor:
        return n2d2.Tensor.from_N2D2(self.N2D2().getBiases())

    def get_scales(self) -> n2d2.Tensor:
        return n2d2.Tensor.from_N2D2(self.N2D2().getScales())

    def get_means(self) -> n2d2.Tensor:
        return n2d2.Tensor.from_N2D2(self.N2D2().getMeans())

    def get_variances(self) -> n2d2.Tensor:
        return n2d2.Tensor.from_N2D2(self.N2D2().getVariances())

    def set_solver_parameter(self, key, value):
        """Set the parameter ``key`` with the value ``value`` for the attribute ``scale`` and ``bias`` solver.

        :param key: Parameter name
        :type key: str
        :param value: The value of the parameter
        :type value: Any
        """
        self._config_parameters['scale_solver'].set_parameter(key, value)
        self._config_parameters['bias_solver'].set_parameter(key, value)

    @deprecated(reason="You should use scale_solver as an attribute.")
    def get_scale_solver(self):
        return self._config_parameters['scale_solver']

    @deprecated(reason="You should use bias_solver as an attribute.")
    def get_bias_solver(self):
            return self._config_parameters['bias_solver']
    
    def set_filler(self, filler):
        raise ValueError("Batchnorm doesn't support Filler")

    def set_solver(self, solver):
        """"Set the ``scale`` and ``bias`` solver with the same solver.

        :param solver: Solver object
        :type solver: :py:class:`n2d2.solver.Solver`
        """
        if not isinstance(solver, n2d2.solver.Solver):
            raise n2d2.error_handler.WrongInputType("solver", str(type(solver)), ["n2d2.solver.Solver"])
        self.bias_solver = solver.copy()
        self.scale_solver = solver.copy()

    def has_quantizer(self):
        # BatchNorm objects don't have a quantizer !
        return False


class Activation(NeuralNetworkCell, Datatyped):

    _cell_constructors = {
        'Frame<float>': N2D2.ActivationCell_Frame_float,
    }
    if cuda_compiled:
        _cell_constructors.update({
            'Frame_CUDA<float>': N2D2.ActivationCell_Frame_CUDA_float,
        })

    _parameters = {
    }
    _parameters.update(_cell_frame_parameters)
    
    _convention_converter= n2d2.ConventionConverter(_parameters)

    def __init__(self, activation, **config_parameters):
        """
        :param activation: Activation function
        :type activation: :py:class:`n2d2.activation.ActivationFunction`
        """
        NeuralNetworkCell.__init__(self, **config_parameters)
        # No optional parameter
        self._config_parameters["activation"] = activation
        self._parse_optional_arguments([])

    def __call__(self, inputs):
        super().__call__(inputs)

        if self._N2D2_object is None:
            nb_outputs = inputs.dimZ()

            self._set_N2D2_object(self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                                         self.get_name(),
                                                                         nb_outputs,
                                                                         **self._optional_constructor_arguments))
            """Set and initialize here all complex cells members"""
            for key, value in self._config_parameters.items():
                self.__setattr__(key, value)
            self.load_N2D2_parameters(self.N2D2())

        self._add_to_graph(inputs)

        self._N2D2_object.propagate(self._inference)

        return self.get_outputs()



class Reshape(NeuralNetworkCell, Datatyped):

    _cell_constructors = {
        'Frame<float>': N2D2.ReshapeCell_Frame_float,
    }
    if cuda_compiled:
        _cell_constructors.update({
            'Frame_CUDA<float>': N2D2.ReshapeCell_Frame_CUDA_float,
        })
    _parameters = {
        "dims": "Dims",
    }
    _parameters.update(_cell_frame_parameters)

    _convention_converter= n2d2.ConventionConverter(_parameters)
    def __init__(self, dims, **config_parameters):
        """
        :param dims: dims of the new shape of the layer
        :type dims: list
        """
        if not isinstance(dims, list):

            raise n2d2.error_handler.WrongInputType("dims", str(type(dims)), ["list"])
        NeuralNetworkCell.__init__(self, **config_parameters)

        self._constructor_arguments.update({
            'dims': dims,
        })

        # No optional parameter
        self._parse_optional_arguments([])

    def _load_N2D2_constructor_parameters(self, N2D2_object):
        self._constructor_arguments['dims'] = N2D2_object.getDims()

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

            """Set and initialize here all complex cells members"""
            for key, value in self._config_parameters.items():
                self.__setattr__(key, value)
            self.load_N2D2_parameters(self.N2D2())
        self._add_to_graph(inputs)

        self._N2D2_object.propagate(self._inference)

        return self.get_outputs()


class Resize(NeuralNetworkCell):
    _cell_constructors = {
        'Frame': N2D2.ResizeCell_Frame,
    }
    if cuda_compiled:
        _cell_constructors.update({
            'Frame_CUDA': N2D2.ResizeCell_Frame_CUDA,
        })
    _parameters = {
        "align_corners": "AlignCorners",
    }
    _parameters.update(_cell_frame_parameters)

    _convention_converter = n2d2.ConventionConverter(_parameters)

    def __init__(self, outputs_width, outputs_height, resize_mode, **config_parameters):
        """
        :param outputs_width: outputs_width
        :type outputs_width: int
        :param outputs_height: outputs_height
        :type outputs_height: int
        :param resize_mode: Resize interpolation mode. Can be, ``Bilinear`` or ``BilinearTF`` (TensorFlow implementation)
        :type resize_mode: str
        :param align_corners: Corner alignement mode if ``BilinearTF`` is used as interpolation mode, default=True
        :type align_corners: boolean, optional   
        """
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

    def _load_N2D2_constructor_parameters(self, N2D2_object):
        self._constructor_arguments['outputs_width'] =  N2D2_object.getResizeOutputWidth()
        self._constructor_arguments['outputs_height'] = N2D2_object.getResizeOutputHeight()
        self._constructor_arguments['resize_mode'] = N2D2_object.getMode()


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

            """Set and initialize here all complex cells members"""
            for key, value in self._config_parameters.items():
                self.__setattr__(key, value)
            self.load_N2D2_parameters(self.N2D2())

        self._add_to_graph(inputs)

        self._N2D2_object.propagate(self._inference)

        return self.get_outputs()


class Transpose(NeuralNetworkCell, Datatyped):
    _cell_constructors = {
        'Frame<float>': N2D2.TransposeCell_Frame_float,
    }
    if cuda_compiled:
        _cell_constructors.update({
            'Frame_CUDA<float>': N2D2.TransposeCell_Frame_CUDA_float,
        })
    _parameters = {}
    _parameters.update(_cell_frame_parameters)

    _convention_converter = n2d2.ConventionConverter(_parameters)

    def __init__(self, perm, **config_parameters):
        """
        :param perm: Permutation
        :type perm: list
        """
        if not isinstance(perm, list):
            raise n2d2.error_handler.WrongInputType("outputs_width", type(perm), ["list"])

        NeuralNetworkCell.__init__(self, **config_parameters)

        self._constructor_arguments.update({
            'perm': perm,
        })

        # No optional parameter
        self._parse_optional_arguments([])

    def _load_N2D2_constructor_parameters(self, N2D2_object):
        self._constructor_arguments['perm'] = N2D2_object.getPermutation()
        
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

            """Set and initialize here all complex cells members"""
            for key, value in self._config_parameters.items():
                self.__setattr__(key, value)
            self.load_N2D2_parameters(self.N2D2())

        self._add_to_graph(inputs)

        self._N2D2_object.propagate(self._inference)

        return self.get_outputs()

class Transformation(NeuralNetworkCell, Datatyped):
    _cell_constructors = {
        'Frame': N2D2.TransformationCell_Frame,
    }
    if cuda_compiled:
        _cell_constructors.update({
            'Frame_CUDA': N2D2.TransformationCell_Frame_CUDA,
        })
    _parameters = {
        "nb_outputs": "nbOutputs",
        "transformation": "transformation"
    }
    _parameters.update(_cell_frame_parameters)

    _convention_converter = n2d2.ConventionConverter(_parameters)

    def __init__(self, perm, **config_parameters):
        """
        :param transformation: Transformation to apply
        :type transformation: :py:class:`n2d2.transform.Transformation`
        :param name: Cell name, default=None
        :type name: str, optional
        """
        if not isinstance(perm, list):
            raise n2d2.error_handler.WrongInputType("outputs_width", type(perm), ["list"])

        NeuralNetworkCell.__init__(self, **config_parameters)

        self._constructor_arguments.update({
            "transformation": "transformation",
        })

        # No optional parameter
        self._parse_optional_arguments([])

    def _load_N2D2_constructor_parameters(self, N2D2_object):
        self._constructor_arguments['transformation'] = N2D2_object.getTransformation()
        
    def __call__(self, inputs):
        super().__call__(inputs)

        if self._N2D2_object is None:
            nb_outputs = inputs.dims()[2]

            self._set_N2D2_object(self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                                           self.get_name(),
                                                                           nb_outputs,
                                                                           self._constructor_arguments['transformation'],
                                                                           **self.n2d2_function_argument_parser(
                                                                               self._optional_constructor_arguments)))

            """Set and initialize here all complex cells members"""
            for key, value in self._config_parameters.items():
                self.__setattr__(key, value)
            self.load_N2D2_parameters(self.N2D2())

        self._add_to_graph(inputs)

        self._N2D2_object.propagate(self._inference)

        return self.get_outputs()
