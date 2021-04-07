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

    _type = None

    def __init__(self,  **config_parameters):

        if 'name' in config_parameters:
            self._name = config_parameters.pop('name')
        else:
            self._name = "cell_" + str(n2d2.global_variables.cell_counter)
        n2d2.global_variables.cell_counter += 1

        self._inputs = []

        if 'model' in config_parameters:
            self._model = config_parameters.pop('model')
        else:
            self._model = n2d2.global_variables.default_model
        if 'datatype' in config_parameters:
            self.datatype = config_parameters.pop('datatype')
        else:
            self.datatype = n2d2.global_variables.default_datatype

        self._model_key = self._model + '<' + self.datatype + '>'

        N2D2_Interface.__init__(self, **config_parameters)

        self._connection_parameters = {}

        self._deepnet = None
        self._inference = False

    def learn(self):
        self._inference = False

    def test(self):
        self._inference = True

    """
    def _infer_deepnet(self, inputs):
        if isinstance(inputs, n2d2.deepnet.DeepNet):
            deepnet = inputs
            # TODO: Check if has stimuli provider/dims/output cell?
        elif isinstance(inputs, list):
            if len(inputs) == 0:
                raise RuntimeError("List with 0 elements cannot provide a deepNet")
            else:
                last_deepnet = None
                for ipt in inputs:
                    deepnet = ipt.get_last().get_deepnet()
                    if last_deepnet is not None:
                        if not id(deepnet) == id(last_deepnet):
                            raise RuntimeError("Elements of cell input have different deepnets. "
                                               "Cannot infer implicit deepnet")
                    last_deepnet = deepnet
                deepnet = last_deepnet
        elif isinstance(inputs, Cell):
            deepnet = inputs.get_deepnet()
        elif isinstance(inputs, n2d2.provider.Provider):
            deepnet = n2d2.deepnet.DeepNet(inputs)
        else:
            raise TypeError("Object of type " + str(type(inputs)) + " cannot implicitly provide a deepNet to cell.")
        return deepnet
    """

    def _infer_deepnet(self, inputs):
        if isinstance(inputs, n2d2.tensor.Tensor):
            deepnet = inputs.get_deepnet()
            if deepnet is None:
                deepnet = n2d2.deepnet.DeepNet()
        elif isinstance(inputs, list):
            if len(inputs) == 0:
                raise RuntimeError("List with 0 elements cannot provide a deepNet")
            else:
                last_deepnet = None
                for ipt in inputs:
                    deepnet = self._infer_deepnet(ipt)
                    if last_deepnet is not None:
                        if not id(deepnet) == id(last_deepnet):
                            raise RuntimeError("Elements of cell input have different deepnets. "
                                               "Cannot infer implicit deepnet")
                    last_deepnet = deepnet
                deepnet = last_deepnet
        else:
            raise TypeError("Object of type " + str(type(inputs)) + " cannot implicitly provide a deepNet to cell.")
        return deepnet


    def dims(self):
        return self.get_outputs().dims()

    def get_outputs(self):
        return self._N2D2_object.getOutputs()

    def get_first(self):
        return self

    def get_last(self):
        return self

    # TODO: Add check that final cell
    def get_deepnet(self):
        return self._deepnet


    def get_type(self):
        return self._type

    """
    def add_input(self, inputs):
        if isinstance(inputs, n2d2.deepnet.DeepNet):
            self.add_input(inputs.get_last())
        elif isinstance(inputs, list):
            for cell in inputs:
                self.add_input(cell)
        elif isinstance(inputs, Cell) or isinstance(inputs, n2d2.provider.Provider):
            self._link_N2D2_input(inputs)
            self._inputs.append(inputs)
        else:
            raise TypeError("Cannot add object of type " + str(type(inputs)))
    """

    # TODO: What exactly should be checked? Input identity and/or input dimensions? At the moment we only check dimensions
    # This means a new Cell with same dimensions is will not be connected!
    def _check_tensor(self, inputs):
        if isinstance(inputs, n2d2.cell.Cell) or isinstance(inputs, n2d2.provider.Provider):
            input_dims = inputs.dims()
            if not self.dims(): # If not initialized
                return True
            """"
            # TODO: Does not work for multi inputs, because getInputsDims returns sum of all channel dims
            if not self._N2D2_object.getInputsDims() + [self.dims()[3]] == input_dims: # If input dimesions changed
                raise RuntimeError("Cell '" + self.get_name() + "' was called with input of dim " + str(inputs.dims())
                                   + ", but cell input size is " + str(self._N2D2_object.getInputsDims()+ [self.dims()[3]]) +
                                   ". Inputs dimensions cannot change after first call.")
                #return True
            """
        else:
            raise TypeError("Invalid inputs object of type " + str(type(inputs)))
        # Check if inputs have same deepnet
        #self._infer_deepnet(inputs)
        return False

    def add_input(self, inputs):
        if isinstance(inputs, list):

            pass
        elif isinstance(inputs, n2d2.tensor.Tensor):
            inputs = [inputs]
        else:
            raise TypeError("Cannot add object of type " + str(type(inputs)))

        self._inputs = []
        self._N2D2_object.clearInputTensors()
        initialize = False

        parents = []
        for ipt in inputs:
            cell = ipt.cell
            self._link_N2D2_input(cell)
            if self._check_tensor(cell):
                initialize = True

            if not isinstance(cell, n2d2.provider.Provider):
                parents.append(cell.N2D2())
            self._inputs.append(cell.get_name())

        self._deepnet.N2D2().addCell(self._N2D2_object, parents)
        if initialize:
            self._N2D2_object.initializeDataDependent()

        """
        cell = inputs.cell
        self._N2D2_object.clearInputTensors()
        self._link_N2D2_input(cell)
        if self._check_tensor(cell):
            self._N2D2_object.initializeDataDependent()
        self._inputs.append(cell.get_name())
        # TODO: N2D2 should check here that the cell does not already exist in the DeepNet
        if not isinstance(cell, n2d2.provider.Provider):
            self._deepnet.N2D2().addCell(self._N2D2_object, [cell.N2D2()])
        else:
            self._deepnet.N2D2().addCell(self._N2D2_object, [])
        """


    """
    Links N2D2 cells taking into account cell connection parameters
    """
    # TODO: Simply connection parameters
    def _link_N2D2_input(self, inputs):
        """if 'mapping' in self._connection_parameters:
            if isinstance(inputs, n2d2.cell.Cell):
                dim_z = inputs.N2D2().getNbOutputs()
            elif isinstance(inputs, n2d2.provider.Provider):
                dim_z = inputs.N2D2().getNbChannels()
            else:
                raise ValueError("Incompatible inputs of type " + str(type(inputs)))
            self._connection_parameters['mapping'] = self._connection_parameters['mapping'].create_N2D2_mapping(
                                           dim_z,
                                           self._N2D2_object.getNbOutputs()
                                       ).N2D2()"""
        #self._N2D2_object.clearInputs() #Necessary to reinitialize input dimensions. TODO: Add dimension check
        #self._N2D2_object.clearInputTensors()
        #print(inputs.dims())
        #print(self._N2D2_object.getInputsDims())
        self._N2D2_object.linkInput(inputs.N2D2())

        #if isinstance(inputs, n2d2.provider.Provider):
        #    self._deepnet.set_provider(inputs)


    def _add_to_graph(self, inputs):
        self.add_input(inputs)
        #self._N2D2_object.initialize()
        self._deepnet.add_to_current_group(self)

    def set_activation(self, activation):
        print("Note: Replacing potentially existing activation in cell: " + self.get_name())
        self._config_parameters['activation'] = activation
        self._N2D2_object.setActivation(self._config_parameters['activation'].N2D2())

    def set_activation_quantizer(self, quantizer):
        self._N2D2_object.getActivation().setQuantizer(quantizer.N2D2())
        # TODO: Create n2d2 objects to obtain visibility of objects in API

    def get_inputs(self):
        return self._inputs

    def clear_input(self):
        self._inputs = []
        self._N2D2_object.clearInputs()

    # TODO: Remove these methods to not expose them
    def propagate(self, inference=False):
        self._N2D2_object.propagate(inference)

    def back_propagate(self):
        self._N2D2_object.backPropagate()

    def update(self):
        self._N2D2_object.update()


    def import_free_parameters(self, dir_name, ignoreNotExists=False):
        if self._N2D2_object:
            filename = dir_name + "/" + self.get_name() + ".syntxt"
            print("import " + filename)
            self._N2D2_object.importFreeParameters(filename, ignoreNotExists)

    """
    def import_activation_parameters(self, filename, **kwargs):
        print("import " + filename)
        self._N2D2_object.importActivationParameters(filename, **kwargs)
    """

    def get_name(self):
        return self._name

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
        output = "\'" + self.get_name() + "\' " + self.get_type()+"(" + self._model_key + ")"
        output += N2D2_Interface.__str__(self)
        if len(self.get_inputs()) > 0:
            output += "[inputs="
            for idx, name in enumerate(self.get_inputs()):
                if idx > 0:
                    output += ", "
                output += "'" + name + "'"
            output += "]"
        else:
            output += "[inputs=[]]"
        return output



class Fc(Cell):

    _type = "Fc"

    _cell_constructors = {
            'Frame<float>': N2D2.FcCell_Frame_float,
            'Frame_CUDA<float>': N2D2.FcCell_Frame_CUDA_float,
    }

    def __init__(self, nb_inputs, nb_outputs, from_arguments=True, **config_parameters):
        """
        :param nb_outputs: Number of outputs of the cell.
        :type nb_outputs: int
        :param name: Name fo the cell.
        :type name: str
        :param activation_function: Activation function used by the cell.
        :type activation_function: :py:class:`n2d2.activation.Activation`, optional
        :param weights_solver: Solver for weights
        :type weights_solver: :py:class:`n2d2.solver.Solver`, optional
        :param bias_solver: Solver for biases
        :type bias_solver: :py:class:`n2d2.solver.Solver`, optional
        :param weights_filler: Algorithm used to fill the weights.
        :type weights_filler: :py:class:`n2d2.filler.Filler`, optional
        :param bias_filler: Algorithm used to fill the biases.
        :type bias_filler: :py:class:`n2d2.filler.Filler`, optional
        """

        if not from_arguments and (nb_inputs is not None or nb_outputs is not None or len(config_parameters) > 0):
            raise RuntimeError("N2D2_object argument give to cell but 'nb_inputs' or 'nb_outputs' or 'config parameters' not None")
        if from_arguments:
            self._create_from_arguments(nb_inputs, nb_outputs, **config_parameters)


    def _create_from_arguments(self, nb_inputs, nb_outputs, **config_parameters):
        Cell.__init__(self, **config_parameters)

        self._constructor_arguments.update({
            'nb_inputs': nb_inputs,
            'nb_outputs': nb_outputs,
        })

        self._N2D2_object = self._cell_constructors[self._model_key](N2D2.DeepNet(n2d2.global_variables.default_net),
                                                                     self.get_name(),
                                                                     self._constructor_arguments['nb_outputs'])
        # Set connection and mapping parameters
        for key in self._config_parameters:
            if key is 'inputOffsetX':
                self._connection_parameters['x0'] = self._config_parameters.pop('inputOffsetX')
            elif key is 'inputOffsetY':
                self._connection_parameters['y0'] = self._config_parameters.pop('inputOffsetY')
            elif key is 'inputWidth':
                self._connection_parameters['width'] = self._config_parameters.pop('inputWidth')
            elif key is 'inputHeight':
                self._connection_parameters['height'] = self._config_parameters.pop('inputHeight')
            elif key is 'mapping':
                self._connection_parameters['mapping'] = self._config_parameters.pop('mapping').create_N2D2_mapping(nb_inputs, nb_outputs).N2D2()


        # Set and initialize here all complex cell members
        for key, value in self._config_parameters.items():
            if key is 'activation_function':
                self._N2D2_object.setActivation(value.N2D2())
            elif key is 'weights_solver':
                self._N2D2_object.setWeightsSolver(value.N2D2())
            elif key is 'bias_solver':
                self._N2D2_object.setBiasSolver(value.N2D2())
            elif key is 'weights_filler':
                self._N2D2_object.setWeightsFiller(value.N2D2())
            elif key is 'bias_filler':
                self._N2D2_object.setBiasFiller(value.N2D2())
            elif key is 'quantizer':
                self._N2D2_object.setQuantizer(value.N2D2())
            else:
                self._set_N2D2_parameter(self.python_to_n2d2_convention(key), value)

        # TODO: Does only work for mapping at the moment. Adapt for other connection parameters
        self._N2D2_object.initializeParameters(nb_inputs, 1, **self._connection_parameters)



    @classmethod
    def create_from_N2D2_object(cls, N2D2_object, n2d2_deepnet=None):

        n2d2_cell = cls(None, None, from_arguments=False)

        Cell.__init__(n2d2_cell,
                      N2D2_object.getNbOutputs(),
                      name=N2D2_object.getName(),
                      **N2D2_Interface.load_N2D2_parameters(N2D2_object))

        n2d2_cell._constructor_arguments.update({
            'nb_inputs': N2D2_object.getInputsSize(),
        })

        n2d2_cell._N2D2_object = N2D2_object

        # NOTE: No Fillers because existing cell
        # TODO: Add similar methods to Activation/Solver/Quantizer for nice prints
        n2d2_cell._config_parameters['activation_function'] = n2d2_cell._N2D2_object.getActivation()
        n2d2_cell._config_parameters['weights_solver'] = n2d2_cell._N2D2_object.getWeightsSolver()
        n2d2_cell._config_parameters['bias_solver'] = n2d2_cell._N2D2_object.getBiasSolver()
        n2d2_cell._config_parameters['weights_solver'] = n2d2_cell._N2D2_object.getWeightsSolver()
        n2d2_cell._config_parameters['quantizer'] = n2d2_cell._N2D2_object.getQuantizer()

        if n2d2_deepnet is not None:
            n2d2_cell._deepnet = n2d2_deepnet
            n2d2_cell._sync_inputs_and_parents()
        else:
            n2d2_cell._deepnet = None
            n2d2_cell._N2D2_object.clearInputs()

        return n2d2_cell


    def __call__(self, inputs):
        if inputs.nb_dims() != 4:
            raise ValueError("Input Tensor should have 4 dimensions, " + str(inputs.nb_dims()), " were given.")
        self._deepnet = self._infer_deepnet(inputs)

        self._add_to_graph(inputs)

        self.propagate(self._inference)

        return n2d2.Tensor.from_N2D2(self.get_outputs()).set_cell(self)

    # TODO: This is not working as expected because solvers are copied in a vector at cell initialization.
    #  setWeightsSolver sets only the solver to be copied but does not modify after cell initialization
    """
    def set_weights_solver(self, solver):
        self._config_parameters['weights_solver'] = solver
        self._N2D2_object.setWeightsSolver(self._config_parameters['weights_solver'].N2D2())
    """

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
        :return: list of weight
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


    def set_bias_solver(self, solver):
        self._config_parameters['bias_solver'] = solver
        self._N2D2_object.setBiasSolver(self._config_parameters['bias_solver'].N2D2())

    def set_quantizer(self, quantizer):
        self._config_parameters['quantizer'] = quantizer
        self._N2D2_object.setQuantizer(self._config_parameters['quantizer'].N2D2())


    # TODO: General set_solver that copies solver and does both





# TODO: This is less powerful than the generator, in the sense that it does not accept several formats for the stride, conv, etc.
class Conv(Cell):
    _type = "Conv"

    _cell_constructors = {
        'Frame<float>': N2D2.ConvCell_Frame_float,
        'Frame_CUDA<float>': N2D2.ConvCell_Frame_CUDA_float,
        'Frame<double>': N2D2.ConvCell_Frame_double,
        'Frame_CUDA<double>': N2D2.ConvCell_Frame_CUDA_double,
    }

    def __init__(self,
                 nb_inputs,
                 nb_outputs,
                 kernel_dims,
                 from_arguments=True,
                 **config_parameters):
        """
        :param nb_outputs: Number of outputs of the cell.
        :type nb_outputs: int
        :param name: Name fo the cell.
        :type name: str
        :param kernel_dims: Kernel dimension.
        :type kernel_dims: list
        :param sub_sample_dims: TODO
        :type sub_sample_dims: list, optional
        :param stride_dims: Size of the stride
        :type stride_dims: list, optional
        :param padding_dims: TODO
        :type padding_dims: list, optional
        :param dilation_dims: TODO
        :type dilation_dims: list, optional
        :param noBias: TODO
        :type dilation_dims: list, optional 
        """

        if not from_arguments and (nb_inputs is not None or nb_outputs is not None or kernel_dims is not None or len(config_parameters) > 0):
            raise RuntimeError("N2D2_object argument give to cell but 'nb_inputs' or 'nb_outputs'  or 'kernel_dims' or 'config parameters' not None")
        if from_arguments:
            self._create_from_arguments(nb_inputs, nb_outputs, kernel_dims, **config_parameters)


    def _create_from_arguments(self, nb_inputs, nb_outputs, kernel_dims, **config_parameters):

        Cell.__init__(self, **config_parameters)

        self._constructor_arguments.update({
            'nb_inputs': nb_inputs,
            'nb_outputs': nb_outputs,
            'kernel_dims': kernel_dims,
        })

        self._parse_optional_arguments(['sub_sample_dims', 'stride_dims', 'padding_dims', 'dilation_dims'])

        self._N2D2_object = self._cell_constructors[self._model_key](N2D2.DeepNet(n2d2.global_variables.default_net),
                                                                     self.get_name(),
                                                                     self._constructor_arguments['kernel_dims'],
                                                                     self._constructor_arguments['nb_outputs'],
                                                                     **self.n2d2_function_argument_parser(self._optional_constructor_arguments))


        """Set connection and mapping parameters"""
        if 'mapping' in self._config_parameters:
            if isinstance(self, ConvDepthWise):
                self._connection_parameters['mapping'] = self._config_parameters.pop('mapping').create_N2D2_mapping(nb_inputs, nb_outputs).N2D2()
            else:
                self._connection_parameters['mapping'] = self._config_parameters.pop('mapping').create_N2D2_mapping(nb_inputs, nb_outputs).N2D2()


        # TODO: Add Kernel section of generator

        """Set and initialize here all complex cell members"""
        for key, value in self._config_parameters.items():
            if key is 'activation_function':
                self._N2D2_object.setActivation(value.N2D2())
            elif key is 'weights_solver':
                self._N2D2_object.setWeightsSolver(value.N2D2())
            elif key is 'bias_solver':
                self._N2D2_object.setBiasSolver(value.N2D2())
            elif key is 'weights_filler':
                self._N2D2_object.setWeightsFiller(value.N2D2())
            elif key is 'bias_filler':
                self._N2D2_object.setBiasFiller(value.N2D2())
            elif key is 'quantizer':
                self._N2D2_object.setQuantizer(value.N2D2())
            else:
                self._set_N2D2_parameter(self.python_to_n2d2_convention(key), value)

        if isinstance(self, ConvDepthWise):
            self._N2D2_object.initializeParameters(nb_inputs, 1, **self._connection_parameters)
        else:
            self._N2D2_object.initializeParameters(nb_inputs, 1, **self._connection_parameters)



    @classmethod
    def create_from_N2D2_object(cls, N2D2_object,  n2d2_deepnet=None):

        n2d2_cell = cls(None, None, None, from_arguments=False)

        Cell.__init__(n2d2_cell,
                      N2D2_object.getNbOutputs(),
                      name=N2D2_object.getName(),
                      **N2D2_Interface.load_N2D2_parameters(N2D2_object))


        n2d2_cell._constructor_arguments.update({
            'nb_inputs': N2D2_object.getInputsSize(),
        })

        n2d2_cell._N2D2_object = N2D2_object

        n2d2_cell._constructor_arguments['kernel_dims'] = [n2d2_cell._N2D2_object.getKernelWidth(), n2d2_cell._N2D2_object.getKernelHeight()]
        n2d2_cell._optional_constructor_arguments['sub_sample_dims'] = [n2d2_cell._N2D2_object.getSubSampleX(), n2d2_cell._N2D2_object.getSubSampleY()]
        n2d2_cell._optional_constructor_arguments['stride_dims'] = [n2d2_cell._N2D2_object.getStrideX(), n2d2_cell._N2D2_object.getStrideY()]
        n2d2_cell._optional_constructor_arguments['padding_dims'] = [n2d2_cell._N2D2_object.getPaddingX(), n2d2_cell._N2D2_object.getPaddingY()]
        n2d2_cell._optional_constructor_arguments['dilation_dims'] = [n2d2_cell._N2D2_object.getDilationX(), n2d2_cell._N2D2_object.getDilationY()]


        # NOTE: No Fillers because existing cell
        # TODO: Add similar methods to Activation/Solver/Quantizer for nice prints
        n2d2_cell._config_parameters['activation_function'] = n2d2_cell._N2D2_object.getActivation()
        n2d2_cell._config_parameters['weights_solver'] = n2d2_cell._N2D2_object.getWeightsSolver()
        n2d2_cell._config_parameters['bias_solver'] = n2d2_cell._N2D2_object.getBiasSolver()
        n2d2_cell._config_parameters['weights_solver'] = n2d2_cell._N2D2_object.getWeightsSolver()
        n2d2_cell._config_parameters['quantizer'] = n2d2_cell._N2D2_object.getQuantizer()

        if n2d2_deepnet is not None:
            n2d2_cell._deepnet = n2d2_deepnet
            n2d2_cell._sync_inputs_and_parents()
        else:
            n2d2_cell._deepnet = None
            n2d2_cell._N2D2_object.clearInputs()

        return n2d2_cell

    def __call__(self, inputs):
        if inputs.nb_dims() != 4:
            raise ValueError("Input Tensor should have 4 dimensions, " + str(inputs.nb_dims()), " were given.")
        self._deepnet = self._infer_deepnet(inputs)

        self._add_to_graph(inputs)

        self.propagate(self._inference)

        return n2d2.Tensor.from_N2D2(self.get_outputs()).set_cell(self)


    # TODO: This is not working as expected because solvers are copied in a vector at cell initialization.
    #  setWeightsSolver sets only the solver to be copied but does not modify after cell initialization
    """
    def set_weights_solver(self, solver):
        self._config_parameters['weights_solver'] = solver
        self._N2D2_object.setWeightsSolver(self._config_parameters['weights_solver'].N2D2())
    """

    def set_bias_solver(self, solver):
        self._config_parameters['bias_solver'] = solver
        self._N2D2_object.setBiasSolver(self._config_parameters['bias_solver'].N2D2())

    def set_quantizer(self, quantizer):
        self._config_parameters['quantizer'] = quantizer
        self._N2D2_object.setQuantizer(self._config_parameters['quantizer'].N2D2())

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
        :return: list of weight
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

class ConvDepthWise(Conv):
    _type = 'ConvDepthWise'

    def __init__(self,
                 nbChannels,
                 kernel_dims,
                 **config_parameters):

        if 'mapping' in config_parameters:
            raise RuntimeError('ConvDepthWise does not support custom mappings')
        else:
            config_parameters['mapping'] = n2d2.mapping.Mapping(nb_channels_per_group=1)
        Conv.__init__(self, nbChannels, nbChannels, kernel_dims, **config_parameters)


class ConvPointWise(Conv):
    _type = 'ConvPointWise'

    def __init__(self,
                 nb_inputs,
                 nb_outputs,
                 **config_parameters):
        Conv.__init__(self, nb_inputs, nb_outputs, [1, 1], stride_dims=[1, 1], **config_parameters)




class Softmax(Cell):

    _type = "Softmax"

    _cell_constructors = {
        'Frame<float>': N2D2.SoftmaxCell_Frame_float,
        'Frame_CUDA<float>': N2D2.SoftmaxCell_Frame_CUDA_float,
        'Frame<double>': N2D2.SoftmaxCell_Frame_double,
        'Frame_CUDA<double>': N2D2.SoftmaxCell_Frame_CUDA_double,
    }

    def __init__(self, from_arguments=True, **config_parameters):

        if not from_arguments and len(config_parameters) > 0:
            raise RuntimeError(
                "N2D2_object argument give to cell but 'inputs' or 'nb_outputs' or 'config parameters' not None")
        if from_arguments:
            self._create_from_arguments(**config_parameters)

    def _create_from_arguments(self, **config_parameters):
        Cell.__init__(self, **config_parameters)
        self._parse_optional_arguments(['with_loss', 'group_size'])

    def __call__(self, inputs):
        if inputs.nb_dims() != 4:
            raise ValueError("Input Tensor should have 4 dimensions, " + str(inputs.nb_dims()), " were given.")
        self._deepnet = self._infer_deepnet(inputs)

        if self._N2D2_object is None:
            nb_outputs = inputs.dims()[2]

            self._N2D2_object = self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                                         self.get_name(),
                                                                         nb_outputs,
                                                                         **self.n2d2_function_argument_parser(self._optional_constructor_arguments))

            """Set and initialize here all complex cell members"""
            for key, value in self._config_parameters.items():
                if key is 'activation_function':
                    self._N2D2_object.setActivation(value.N2D2())
                else:
                    self._set_N2D2_parameter(self.python_to_n2d2_convention(key), value)

        self._add_to_graph(inputs)

        self.propagate(self._inference)

        return n2d2.Tensor.from_N2D2(self.get_outputs()).set_cell(self)




    @classmethod
    def create_from_N2D2_object(cls, inputs, N2D2_object, n2d2_deepnet):

        n2d2_cell = cls(inputs, from_arguments=False)

        Cell.__init__(n2d2_cell,
                      inputs,
                      N2D2_object.getNbOutputs(),
                      deepNet=n2d2_deepnet,
                      name=N2D2_object.getName(),
                      **N2D2_Interface.load_N2D2_parameters(N2D2_object))

        n2d2_cell._N2D2_object = N2D2_object

        n2d2_cell._optional_constructor_arguments['with_loss'] = n2d2_cell._N2D2_object.getWithLoss()
        n2d2_cell._optional_constructor_arguments['group_size'] = n2d2_cell._N2D2_object.getGroupSize()

        n2d2_cell._config_parameters['activation_function'] = n2d2_cell._N2D2_object.getActivation()

        n2d2_cell._sync_inputs_and_parents(inputs)

        return n2d2_cell





class Pool(Cell):
    _type = 'Pool'

    _cell_constructors = {
        'Frame<float>': N2D2.PoolCell_Frame_float,
        'Frame_CUDA<float>': N2D2.PoolCell_Frame_CUDA_float,
    }

    def __init__(self,
                 pool_dims,
                 from_arguments=True,
                 **config_parameters):

        if not from_arguments and (pool_dims is not None or len(config_parameters) > 0):
            raise RuntimeError(
                "N2D2_object argument give to cell but 'inputs' or 'nb_outputs' or 'config parameters' not None")
        if from_arguments:
            self._create_from_arguments(pool_dims, **config_parameters)

    def _create_from_arguments(self, pool_dims, **config_parameters):
        Cell.__init__(self, **config_parameters)

        self._constructor_arguments.update({
            'pool_dims': pool_dims,
        })

        # Note: Removed Pooling
        self._parse_optional_arguments(['stride_dims', 'padding_dims', 'pooling'])

        self._optional_constructor_arguments['pooling'] = \
            N2D2.PoolCell.Pooling.__members__[self._optional_constructor_arguments['pooling']]

        """Set connection and mapping parameters"""
        if 'mapping' in self._config_parameters:
            self._connection_parameters['mapping'] = self._config_parameters.pop('mapping').create_N2D2_mapping(nb_inputs, nb_outputs).N2D2()


    @classmethod
    def create_from_N2D2_object(cls, inputs, N2D2_object, n2d2_deepnet):

        n2d2_cell = cls(inputs, None, None, from_arguments=False)

        Cell.__init__(n2d2_cell,
                      inputs,
                      N2D2_object.getNbOutputs(),
                      deepNet=n2d2_deepnet,
                      name=N2D2_object.getName(),
                      **N2D2_Interface.load_N2D2_parameters(N2D2_object))

        n2d2_cell._N2D2_object = N2D2_object

        n2d2_cell._constructor_arguments['pool_dims'] = [n2d2_cell._N2D2_object.getPoolWidth(),
                                                        n2d2_cell._N2D2_object.getPoolHeight()]
        n2d2_cell._optional_constructor_arguments['stride_dims'] = [n2d2_cell._N2D2_object.getStrideX(),
                                                                   n2d2_cell._N2D2_object.getStrideY()]
        n2d2_cell._optional_constructor_arguments['padding_dims'] = [n2d2_cell._N2D2_object.getPaddingX(),
                                                                    n2d2_cell._N2D2_object.getPaddingY()]
        n2d2_cell._optional_constructor_arguments['pooling'] = n2d2_cell._N2D2_object.getPooling()

        n2d2_cell._config_parameters['activation_function'] = n2d2_cell._N2D2_object.getActivation()

        n2d2_cell._sync_inputs_and_parents(inputs)

        return n2d2_cell

    def __call__(self, inputs):
        if inputs.nb_dims() != 4:
            raise ValueError("Input Tensor should have 4 dimensions, " + str(inputs.nb_dims()), " were given.")

        self._deepnet = self._infer_deepnet(inputs)

        if self._N2D2_object is None:

            self._N2D2_object = self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                                         self.get_name(),
                                                                         self._constructor_arguments['pool_dims'],
                                                                         inputs.dims()[2],
                                                                         **self.n2d2_function_argument_parser(self._optional_constructor_arguments))

            """Set and initialize here all complex cell members"""
            for key, value in self._config_parameters.items():
                if key is 'activation_function':
                    self._N2D2_object.setActivation(value.N2D2())
                else:
                    self._set_N2D2_parameter(self.python_to_n2d2_convention(key), value)

            self._N2D2_object.initializeParameters(0, 1, self._connection_parameters['mapping'].create_N2D2_mapping(inputs.dims()[2], inputs.dims()[2]))

        self._add_to_graph(inputs)

        self.propagate(self._inference)

        return n2d2.Tensor.from_N2D2(self.get_outputs()).set_cell(self)     

class Pool2d(Cell):
    _type = 'Pool2d'

    _cell_constructors = {
        'Frame<float>': N2D2.PoolCell_Frame_float,
        'Frame_CUDA<float>': N2D2.PoolCell_Frame_CUDA_float,
    }

    def __init__(self,
                 pool_dims,
                 from_arguments=True,
                 **config_parameters):


        if not from_arguments and (pool_dims is not None or len(config_parameters) > 0):
            raise RuntimeError(
                "N2D2_object argument give to cell but 'inputs' or 'nb_outputs' or 'config parameters' not None")
        if from_arguments:
            self._create_from_arguments(pool_dims, **config_parameters)

    def _create_from_arguments(self, pool_dims, **config_parameters):

        Cell.__init__(self, **config_parameters)

        self._constructor_arguments.update({
            'pool_dims': pool_dims,
        })

        self._parse_optional_arguments(['stride_dims', 'padding_dims', 'pooling'])

        self._optional_constructor_arguments['pooling'] = \
            N2D2.PoolCell.Pooling.__members__[self._optional_constructor_arguments['pooling']]

        if 'mapping' in config_parameters:
            raise RuntimeError('Pool2d does not support custom mappings')
        else:
            self._connection_parameters['mapping'] = n2d2.mapping.Mapping(nb_channels_per_group=1)

    def __call__(self, inputs):
        if inputs.nb_dims() != 4:
            raise ValueError("Input Tensor should have 4 dimensions, " + str(inputs.nb_dims()), " were given.")

        self._deepnet = self._infer_deepnet(inputs)

        if self._N2D2_object is None:

            self._N2D2_object = self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                                         self.get_name(),
                                                                         self._constructor_arguments['pool_dims'],
                                                                         inputs.dims()[2],
                                                                         **self.n2d2_function_argument_parser(self._optional_constructor_arguments))

            """Set and initialize here all complex cell members"""
            for key, value in self._config_parameters.items():
                if key is 'activation_function':
                    self._N2D2_object.setActivation(value.N2D2())
                else:
                    self._set_N2D2_parameter(self.python_to_n2d2_convention(key), value)

            self._N2D2_object.initializeParameters(0, 1, self._connection_parameters['mapping'].create_N2D2_mapping(inputs.dims()[2], inputs.dims()[2]).N2D2())

        self._add_to_graph(inputs)

        self.propagate(self._inference)

        return n2d2.Tensor.from_N2D2(self.get_outputs()).set_cell(self)

class GlobalPool2d(Cell):
    _type = 'GlobalPool2d'

    _cell_constructors = {
        'Frame<float>': N2D2.PoolCell_Frame_float,
        'Frame_CUDA<float>': N2D2.PoolCell_Frame_CUDA_float,
    }

    def __init__(self,
                 from_arguments=True,
                 **config_parameters):

        if not from_arguments and (len(config_parameters) > 0):
            raise RuntimeError(
                "N2D2_object argument give to cell")
        if from_arguments:
            self._create_from_arguments(**config_parameters)

    def _create_from_arguments(self, **config_parameters):
        Cell.__init__(self, **config_parameters)

        self._parse_optional_arguments(['pooling'])

        self._optional_constructor_arguments['pooling'] = \
            N2D2.PoolCell.Pooling.__members__[self._optional_constructor_arguments['pooling']]

        if 'mapping' in config_parameters:
            raise RuntimeError('GlobalPool2d does not support custom mappings')
        else:
            self._connection_parameters['mapping'] = n2d2.mapping.Mapping(nb_channels_per_group=1)


    #@classmethod
    #def create_from_N2D2_object(cls, inputs, N2D2_object, n2d2_deepnet):
    #    return n2d2_cell

    def __call__(self, inputs):
        if inputs.nb_dims() != 4:
            raise ValueError("Input Tensor should have 4 dimensions, " + str(inputs.nb_dims()), " were given.")

        self._deepnet = self._infer_deepnet(inputs)

        if self._N2D2_object is None:

            self._N2D2_object = self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                                         self.get_name(),
                                                                         [inputs.dims()[0], inputs.dims()[1]],
                                                                         inputs.dims()[2],
                                                                         strideDims=[1, 1],
                                                                         **self.n2d2_function_argument_parser(self._optional_constructor_arguments))

            """Set and initialize here all complex cell members"""
            for key, value in self._config_parameters.items():
                if key is 'activation_function':
                    self._N2D2_object.setActivation(value.N2D2())
                else:
                    self._set_N2D2_parameter(self.python_to_n2d2_convention(key), value)

            self._N2D2_object.initializeParameters(0, 1, self._connection_parameters['mapping'].create_N2D2_mapping(inputs.dims()[2], inputs.dims()[2]).N2D2())


        self._add_to_graph(inputs)

        self.propagate(self._inference)

        return n2d2.Tensor.from_N2D2(self.get_outputs()).set_cell(self)




# TODO: This is less powerful than the generator, in the sense that it does not accept several formats for the stride, conv, etc.
class Deconv(Cell):

    _cell_constructors = {
        'Frame<float>': N2D2.DeconvCell_Frame_float,
        'Frame_CUDA<float>': N2D2.DeconvCell_Frame_CUDA_float,
        'Frame<double>': N2D2.DeconvCell_Frame_double,
        'Frame_CUDA<double>': N2D2.DeconvCell_Frame_CUDA_double,
    }

    def __init__(self,
                 inputs,
                 nb_outputs,
                 kernel_dims,
                 from_arguments=True,
                 **config_parameters):
        """
        :param nb_outputs: Number of outputs of the cell.
        :type nb_outputs: int
        :param name: Name fo the cell.
        :type name: str
        :param kernel_dims: Kernel dimension.
        :type kernel_dims: list
        :param sub_sample_dims: TODO
        :type sub_sample_dims: list, optional
        :param stride_dims: TODO
        :type stride_dims: list, optional
        :param padding_dims: TODO
        :type padding_dims: list, optional
        :param dilation_dims: TODO
        :type dilation_dims: list, optional
        """

        if not from_arguments and (nb_outputs is not None or kernel_dims is not None or len(config_parameters) > 0):
            raise RuntimeError("N2D2_object argument give to cell but 'inputs' or 'nb_outputs' or 'config parameters' not None")
        if from_arguments:
            self._create_from_arguments(inputs, nb_outputs, kernel_dims, **config_parameters)
        

    def _create_from_arguments(self, inputs, nb_outputs, kernel_dims, **config_parameters):

        Cell.__init__(self, inputs, nb_outputs, **config_parameters)

        self._constructor_arguments.update({
            'kernel_dims': kernel_dims,
        })

        self._parse_optional_arguments(['stride_dims', 'padding_dims', 'dilation_dims'])

        self._N2D2_object = self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                                     self.get_name(),
                                                                     self._constructor_arguments['kernel_dims'],
                                                                     self._constructor_arguments['nb_outputs'],
                                                                     **self.n2d2_function_argument_parser(self._optional_constructor_arguments))


        """Set connection and mapping parameters"""
        if 'mapping' in self._config_parameters:
            self._connection_parameters['mapping'] = self._config_parameters.pop('mapping')

        # TODO: Add Kernel section of generator

        """Set and initialize here all complex cell members"""
        for key, value in self._config_parameters.items():
            if key is 'activation_function':
                self._N2D2_object.setActivation(value.N2D2())
            elif key is 'weights_solver':
                self._N2D2_object.setWeightsSolver(value.N2D2())
            elif key is 'bias_solver':
                self._N2D2_object.setBiasSolver(value.N2D2())
            elif key is 'weights_filler':
                self._N2D2_object.setWeightsFiller(value.N2D2())
            elif key is 'bias_filler':
                self._N2D2_object.setBiasFiller(value.N2D2())
            else:
                self._set_N2D2_parameter(self.python_to_n2d2_convention(key), value)

        self._add_to_graph(inputs)


    @classmethod
    def create_from_N2D2_object(cls, inputs, N2D2_object, n2d2_deepnet):

        n2d2_cell = cls(inputs, None, None, from_arguments=False)

        Cell.__init__(n2d2_cell,
                      inputs,
                      N2D2_object.getNbOutputs(),
                      deepNet=n2d2_deepnet,
                      name=N2D2_object.getName(),
                      **N2D2_Interface.load_N2D2_parameters(N2D2_object))

        n2d2_cell._N2D2_object = N2D2_object

        n2d2_cell._constructor_arguments['kernel_dims'] = [n2d2_cell._N2D2_object.getKernelWidth(), n2d2_cell._N2D2_object.getKernelHeight()]
        n2d2_cell._optional_constructor_arguments['stride_dims'] = [n2d2_cell._N2D2_object.getStrideX(), n2d2_cell._N2D2_object.getStrideY()]
        n2d2_cell._optional_constructor_arguments['padding_dims'] = [n2d2_cell._N2D2_object.getPaddingX(), n2d2_cell._N2D2_object.getPaddingY()]
        n2d2_cell._optional_constructor_arguments['dilation_dims'] = [n2d2_cell._N2D2_object.getDilationX(), n2d2_cell._N2D2_object.getDilationY()]


        # NOTE: No Fillers because existing cell
        # TODO: Add similar methods to Activation/Solver/Quantizer for nice prints
        n2d2_cell._config_parameters['activation_function'] = n2d2_cell._N2D2_object.getActivation()
        n2d2_cell._config_parameters['weights_solver'] = n2d2_cell._N2D2_object.getWeightsSolver()
        n2d2_cell._config_parameters['bias_solver'] = n2d2_cell._N2D2_object.getBiasSolver()
        n2d2_cell._config_parameters['weights_solver'] = n2d2_cell._N2D2_object.getWeightsSolver()

        n2d2_cell._sync_inputs_and_parents(inputs)

        return n2d2_cell

    # TODO: This is not working as expected because solvers are copied in a vector at cell initialization.
    #  setWeightsSolver sets only the solver to be copied but does not modify after cell initialization
    """
    def set_weights_solver(self, solver):
        self._config_parameters['weights_solver'] = solver
        self._N2D2_object.setWeightsSolver(self._config_parameters['weights_solver'].N2D2())
    """

    def set_bias_solver(self, solver):
        self._config_parameters['bias_solver'] = solver
        self._N2D2_object.setBiasSolver(self._config_parameters['bias_solver'].N2D2())



class ElemWise(Cell):

    _cell_constructors = {
        'Frame': N2D2.ElemWiseCell_Frame,
        'Frame_CUDA': N2D2.ElemWiseCell_Frame_CUDA,
    }

    def __init__(self, from_arguments=True, **config_parameters):

        if not from_arguments and (len(config_parameters) > 0):
            raise RuntimeError(
                "N2D2_object argument give to cell but 'inputs' or 'nb_outputs' or 'config parameters' not None")
        if from_arguments:
            self._create_from_arguments(**config_parameters)
        


    def _create_from_arguments(self, **config_parameters):
        Cell.__init__(self, **config_parameters)

        self._parse_optional_arguments(['operation', 'weights', 'shifts'])

        self._optional_constructor_arguments['operation'] = \
            N2D2.ElemWiseCell.Operation.__members__[self._optional_constructor_arguments['operation']]



    @classmethod
    def create_from_N2D2_object(cls, inputs, N2D2_object, n2d2_deepnet):

        n2d2_cell = cls(inputs, None, from_arguments=False)

        Cell.__init__(n2d2_cell,
                      inputs,
                      N2D2_object.getNbOutputs(),
                      deepNet=n2d2_deepnet,
                      name=N2D2_object.getName(),
                      **N2D2_Interface.load_N2D2_parameters(N2D2_object))

        n2d2_cell._N2D2_object = N2D2_object

        n2d2_cell._optional_constructor_arguments['operation'] = n2d2_cell._N2D2_object.getOperation()
        n2d2_cell._optional_constructor_arguments['weights'] = n2d2_cell._N2D2_object.getWeights()
        n2d2_cell._optional_constructor_arguments['shifts'] = n2d2_cell._N2D2_object.getShifts()

        # NOTE: No Fillers because existing cell
        # TODO: Add similar methods to Activation/Solver/Quantizer for nice prints
        n2d2_cell._config_parameters['activation_function'] = n2d2_cell._N2D2_object.getActivation()

        n2d2_cell._sync_inputs_and_parents(inputs)

        return n2d2_cell

    def __call__(self, inputs):

        for elem in inputs:
            if elem.nb_dims() != 4:
                raise ValueError("Input Tensor should have 4 dimensions, " + str(inputs.nb_dims()), " were given.")

        self._deepnet = self._infer_deepnet(inputs)

        if self._N2D2_object is None:

            nb_outputs = inputs[0].dims()[2]

            self._N2D2_object = self._cell_constructors[self._model](self._deepnet.N2D2(),
                                                                     self.get_name(),
                                                                     nb_outputs,
                                                                     **self.n2d2_function_argument_parser(
                                                                         self._optional_constructor_arguments))

            """Set and initialize here all complex cell members"""
            for key, value in self._config_parameters.items():
                if key is 'activation_function':
                    self._N2D2_object.setActivation(value.N2D2())
                else:
                    self._set_N2D2_parameter(self.python_to_n2d2_convention(key), value)

        self._add_to_graph(inputs)

        self.propagate(self._inference)

        return n2d2.Tensor.from_N2D2(self.get_outputs()).set_cell(self)



class Dropout(Cell):
    _cell_constructors = {
        'Frame<float>': N2D2.DropoutCell_Frame_float,
        'Frame_CUDA<float>': N2D2.DropoutCell_Frame_CUDA_float,
        'Frame<double>': N2D2.DropoutCell_Frame_double,
        'Frame_CUDA<double>': N2D2.DropoutCell_Frame_CUDA_double,
    }
    def __init__(self, inputs, from_arguments=True, **config_parameters):

        if not from_arguments and  len(config_parameters) > 0:
            raise RuntimeError(
                "N2D2_object argument give to cell but 'inputs' or 'nb_outputs' or 'config parameters' not None")
        if from_arguments:
            self._create_from_arguments(inputs, **config_parameters)
        

    def _create_from_arguments(self, inputs, **config_parameters):
        Cell.__init__(self, inputs, inputs.get_nb_outputs(), **config_parameters)
        # No optionnal args
        self._parse_optional_arguments([])
        self._N2D2_object = self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                        self.get_name(),
                                                        self._constructor_arguments['nb_outputs'],
                                                        **self.n2d2_function_argument_parser(self._optional_constructor_arguments))

        # Delete to avoid print
        del self._constructor_arguments['nb_outputs']

        """Set and initialize here all complex cell members"""
        for key, value in self._config_parameters.items():
            if key is 'activation_function':
                self._N2D2_object.setActivation(value.N2D2())
            else:
                self._set_N2D2_parameter(self.python_to_n2d2_convention(key), value)

        self._add_to_graph(inputs)


    @classmethod
    def create_from_N2D2_object(cls, inputs, N2D2_object, n2d2_deepnet):

        n2d2_cell = cls(inputs, from_arguments=False)

        Cell.__init__(n2d2_cell,
                      inputs,
                      N2D2_object.getNbOutputs(),
                      deepNet=n2d2_deepnet,
                      name=N2D2_object.getName(),
                      **N2D2_Interface.load_N2D2_parameters(N2D2_object))

        n2d2_cell._N2D2_object = N2D2_object

        n2d2_cell._config_parameters['activation_function'] = n2d2_cell._N2D2_object.getActivation()

        n2d2_cell._sync_inputs_and_parents(inputs)

        return n2d2_cell


class Padding(Cell):

    _cell_constructors = {
        'Frame': N2D2.PaddingCell_Frame,
        'Frame_CUDA': N2D2.PaddingCell_Frame_CUDA,
    }

    def __init__(self,
                 inputs,
                 nb_outputs,
                 top_pad,
                 bot_pad,
                 left_pad,
                 right_pad,
                 from_arguments=True,
                 **config_parameters):

        if not from_arguments and (nb_outputs is not None
                                        or top_pad is not None
                                        or bot_pad is not None
                                        or left_pad is not None
                                        or right_pad is not None
                                        or len(config_parameters) > 0):
            raise RuntimeError(
                "N2D2_object argument give to cell but 'inputs' or 'nb_outputs' or 'config parameters' not None")
        if from_arguments:
            self._create_from_arguments(inputs, nb_outputs, top_pad, bot_pad, left_pad, right_pad, **config_parameters)
        

    def _create_from_arguments(self, inputs, nb_outputs, top_pad, bot_pad, left_pad, right_pad, **config_parameters):

        Cell.__init__(self, inputs, nb_outputs, **config_parameters)

        self._constructor_arguments.update({
                 'top_pad': top_pad,
                 'bot_pad': bot_pad,
                 'left_pad': left_pad,
                 'right_pad': right_pad
        })

        # No optional args
        self._parse_optional_arguments([])

        self._N2D2_object = self._cell_constructors[self._model](self._deepnet.N2D2(),
                                                                     self.get_name(),
                                                                     self._constructor_arguments['nb_outputs'],
                                                                     self._constructor_arguments['top_pad'],
                                                                     self._constructor_arguments['bot_pad'],
                                                                     self._constructor_arguments['left_pad'],
                                                                     self._constructor_arguments['right_pad'],
                                                                     **self.n2d2_function_argument_parser(self._optional_constructor_arguments))
        """Set and initialize here all complex cell members"""
        for key, value in self._config_parameters.items():
            if key is 'activation_function':
                self._N2D2_object.setActivation(value.N2D2())
            else:
                self._set_N2D2_parameter(self.python_to_n2d2_convention(key), value)

        self._add_to_graph(inputs)

    @classmethod
    def create_from_N2D2_object(cls, inputs, N2D2_object, n2d2_deepnet):

        n2d2_cell = cls(inputs, None, None, None, None, None, from_arguments=False)

        Cell.__init__(n2d2_cell,
                      inputs,
                      N2D2_object.getNbOutputs(),
                      deepNet=n2d2_deepnet,
                      name=N2D2_object.getName(),
                      **N2D2_Interface.load_N2D2_parameters(N2D2_object))

        n2d2_cell._N2D2_object = N2D2_object

        n2d2_cell._constructor_arguments['top_pad'] = n2d2_cell._N2D2_object.getTopPad()
        n2d2_cell._constructor_arguments['bot_pad'] = n2d2_cell._N2D2_object.getBotPad()
        n2d2_cell._constructor_arguments['left_pad'] = n2d2_cell._N2D2_object.getLeftPad()
        n2d2_cell._constructor_arguments['right_pad'] = n2d2_cell._N2D2_object.getRightPad()

        n2d2_cell._config_parameters['activation_function'] = n2d2_cell._N2D2_object.getActivation()

        n2d2_cell._sync_inputs_and_parents(inputs)

        return n2d2_cell

class LRN(Cell):
    _cell_constructors = {
        'Frame<float>': N2D2.LRNCell_Frame_float,
        'Frame_CUDA<float>': N2D2.LRNCell_Frame_CUDA_float,
    }


    def __init__(self, inputs, nb_outputs, from_arguments=True, **config_parameters):

        if not from_arguments and (nb_outputs is not None or len(config_parameters) > 0):
            raise RuntimeError(
                "N2D2_object argument give to cell but 'inputs' or 'nb_outputs' or 'config parameters' not None")
        if from_arguments:
            self._create_from_arguments(inputs, nb_outputs, **config_parameters)
        

    def _create_from_arguments(self, inputs, nb_outputs, **config_parameters):
        Cell.__init__(self, inputs, nb_outputs, **config_parameters)

        # No optional args
        self._parse_optional_arguments([])
        self._N2D2_object = self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                self.get_name(),
                                                self._constructor_arguments['nb_outputs'],
                                                **self.n2d2_function_argument_parser(self._optional_constructor_arguments))

        """Set and initialize here all complex cell members"""
        for key, value in self._config_parameters.items():
            if key is 'activation_function':
                self._N2D2_object.setActivation(value.N2D2())
            else:
                self._set_N2D2_parameter(self.python_to_n2d2_convention(key), value)

        self._add_to_graph(inputs)

    @classmethod
    def create_from_N2D2_object(cls, inputs, N2D2_object, n2d2_deepnet):

        n2d2_cell = cls(inputs, None, from_arguments=False)

        Cell.__init__(n2d2_cell,
                      inputs,
                      N2D2_object.getNbOutputs(),
                      deepNet=n2d2_deepnet,
                      name=N2D2_object.getName(),
                      **N2D2_Interface.load_N2D2_parameters(N2D2_object))

        n2d2_cell._N2D2_object = N2D2_object

        n2d2_cell._config_parameters['activation_function'] = n2d2_cell._N2D2_object.getActivation()

        n2d2_cell._sync_inputs_and_parents(inputs)

        return n2d2_cell

class BatchNorm2d(Cell):
    _cell_constructors = {
        'Frame<float>': N2D2.BatchNormCell_Frame_float,
        'Frame_CUDA<float>': N2D2.BatchNormCell_Frame_CUDA_float,
    }
    def __init__(self, nb_inputs, from_arguments=True, **config_parameters):
        if not from_arguments and len(config_parameters) > 0:
            raise RuntimeError(
                "N2D2_object argument give to cell but 'inputs' or 'nb_outputs' or 'config parameters' not None")
        if from_arguments:
            self._create_from_arguments(nb_inputs, **config_parameters)
        

    def _create_from_arguments(self, nb_inputs, **config_parameters):
        Cell.__init__(self, **config_parameters)
        # No optional parameter
        self._parse_optional_arguments([])
        self._N2D2_object = self._cell_constructors[self._model_key](N2D2.DeepNet(n2d2.global_variables.default_net),
                                                self.get_name(),
                                                nb_inputs,
                                                **self.n2d2_function_argument_parser(self._optional_constructor_arguments))

        """Set and initialize here all complex cell members"""
        for key, value in self._config_parameters.items():
            if key is 'activation_function':
                self._N2D2_object.setActivation(value.N2D2())
            elif key is 'scale_solver':
                self._N2D2_object.setScaleSolver(value.N2D2())
            elif key is 'bias_solver':
                self._N2D2_object.setBiasSolver(value.N2D2())
            else:
                self._set_N2D2_parameter(self.python_to_n2d2_convention(key), value)

        self._N2D2_object.initializeParameters(nb_inputs, 1)


    @classmethod
    def create_from_N2D2_object(cls, inputs, N2D2_object, n2d2_deepnet):

        n2d2_cell = cls(inputs, from_arguments=False)

        Cell.__init__(n2d2_cell,
                      inputs,
                      N2D2_object.getNbOutputs(),
                      deepNet=n2d2_deepnet,
                      name=N2D2_object.getName(),
                      **N2D2_Interface.load_N2D2_parameters(N2D2_object))

        n2d2_cell._N2D2_object = N2D2_object

        # TODO: Add similar methods to Activation/Solver/Quantizer for nice prints and access
        n2d2_cell._config_parameters['activation_function'] = n2d2_cell._N2D2_object.getActivation()
        n2d2_cell._config_parameters['scale_solver'] = n2d2_cell._N2D2_object.getScaleSolver()
        n2d2_cell._config_parameters['bias_solver'] = n2d2_cell._N2D2_object.getBiasSolver()

        n2d2_cell._sync_inputs_and_parents(inputs)

        return n2d2_cell

    def __call__(self, inputs):
        if inputs.nb_dims() != 4:
            raise ValueError("Input Tensor should have 4 dimensions, " + str(inputs.nb_dims()), " were given.")

        self._deepnet = self._infer_deepnet(inputs)

        self._add_to_graph(inputs)

        self.propagate(self._inference)

        return n2d2.Tensor.from_N2D2(self.get_outputs()).set_cell(self)     

    def set_scale_solver(self, solver):
        self._config_parameters['scale_solver'] = solver
        self._N2D2_object.setScaleSolver(self._config_parameters['scale_solver'].N2D2())

    def set_bias_solver(self, solver):
        self._config_parameters['bias_solver'] = solver
        self._N2D2_object.setBiasSolver(self._config_parameters['bias_solver'].N2D2())



class Activation(Cell):

    _type = "Activation" # TODO : used ?

    _cell_constructors = {
        'Frame<float>': N2D2.ActivationCell_Frame_float,
        'Frame_CUDA<float>': N2D2.ActivationCell_Frame_CUDA_float,
    }


    def __init__(self, from_arguments=True, **config_parameters):

        if not from_arguments and len(config_parameters) > 0:
            raise RuntimeError(
                "N2D2_object argument give to cell but 'config parameters' not None")

        if from_arguments:
            self._create_from_arguments(**config_parameters)
        

    def _create_from_arguments(self, **config_parameters):
        Cell.__init__(self,  **config_parameters)
        # No optional parameter
        self._parse_optional_arguments([])

    def __call__(self, inputs):
        if inputs.nb_dims() != 4:
            raise ValueError("Input Tensor should have 4 dimensions, " + str(inputs.nb_dims()), " were given.")

        self._deepnet = self._infer_deepnet(inputs)

        if self._N2D2_object is None:
            nb_outputs = inputs.dims()[2]

            self._N2D2_object = self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                                         self.get_name(),
                                                                         nb_outputs,
                                                                         **self._optional_constructor_arguments)

            """Set and initialize here all complex cell members"""
            for key, value in self._config_parameters.items():
                if key is 'activationFunction':
                    self._N2D2_object.setActivation(value.N2D2())
                else:
                    self._set_N2D2_parameter(self._param_to_INI_convention(key), value)


        self._add_to_graph(inputs)

        self.propagate(self._inference)

        return n2d2.Tensor.from_N2D2(self.get_outputs()).set_cell(self)     

    @classmethod
    def create_from_N2D2_object(cls, inputs, N2D2_object, n2d2_deepnet):

        n2d2_cell = cls(inputs, from_arguments=False)

        Cell.__init__(n2d2_cell,
                      inputs,
                      N2D2_object.getNbOutputs(),
                      deepNet=n2d2_deepnet,
                      name=N2D2_object.getName(),
                      **N2D2_Interface.load_N2D2_parameters(N2D2_object))

        n2d2_cell._N2D2_object = N2D2_object

        n2d2_cell._config_parameters['activation_function'] = n2d2_cell._N2D2_object.getActivation()

        n2d2_cell._sync_inputs_and_parents(inputs)

        return n2d2_cell



class Reshape(Cell):
    _cell_constructors = {
        'Frame<float>': N2D2.ReshapeCell_Frame_float,
        'Frame_CUDA<float>': N2D2.ReshapeCell_Frame_CUDA_float,
    }
    def __init__(self, inputs, nb_outputs, dims, from_arguments=True, **config_parameters):

        if not from_arguments and (nb_outputs is not None or dims is not None or len(config_parameters) > 0):
            raise RuntimeError(
                "N2D2_object argument give to cell but 'inputs' or 'nb_outputs' or 'dims' or 'config parameters' not None")
        if from_arguments:
            self._create_from_arguments(inputs, nb_outputs, dims, **config_parameters)
        

    def _create_from_arguments(self, inputs, nb_outputs, dims, **config_parameters):
        Cell.__init__(self, inputs, nb_outputs, **config_parameters)

        self._constructor_arguments.update({
            'dims': dims,
        })

        # No optional parameter
        self._parse_optional_arguments([])
        self._N2D2_object = self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                self.get_name(),
                                                self._constructor_arguments['nb_outputs'],
                                                self._constructor_arguments['dims'],
                                                **self.n2d2_function_argument_parser(self._optional_constructor_arguments))

        """Set and initialize here all complex cell members"""
        for key, value in self._config_parameters.items():
            if key is 'activation_function':
                self._N2D2_object.setActivation(value.N2D2())
            else:
                self._set_N2D2_parameter(self.python_to_n2d2_convention(key), value)

        self._add_to_graph(inputs)

    @classmethod
    def create_from_N2D2_object(cls, inputs, N2D2_object, n2d2_deepnet):

        n2d2_cell = cls(inputs, None, None, from_arguments=False)

        Cell.__init__(n2d2_cell,
                      inputs,
                      N2D2_object.getNbOutputs(),
                      deepNet=n2d2_deepnet,
                      name=N2D2_object.getName(),
                      **N2D2_Interface.load_N2D2_parameters(N2D2_object))

        n2d2_cell._N2D2_object = N2D2_object

        n2d2_cell._constructor_arguments['dims'] = n2d2_cell._N2D2_object.getDims()

        n2d2_cell._config_parameters['activation_function'] = n2d2_cell._N2D2_object.getActivation()

        n2d2_cell._sync_inputs_and_parents(inputs)

        return n2d2_cell
