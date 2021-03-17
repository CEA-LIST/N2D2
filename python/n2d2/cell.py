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

    def __init__(self, inputs, nbOutputs, **config_parameters):

        if 'name' in config_parameters:
            self._name = config_parameters.pop('name')
        else:
            self._name = "cell_" + str(n2d2.global_variables.cell_counter)
        n2d2.global_variables.cell_counter += 1

        self._inputs = []

        if 'deepNet' in config_parameters:
            self._deepnet = config_parameters.pop('deepNet')
        else:
            if isinstance(inputs, n2d2.deepnet.Layer) or isinstance(inputs, list):
                if isinstance(inputs, n2d2.deepnet.Layer):
                    inputs = inputs.get_elements()
                last_deepnet = None
                for ipt in inputs.get_elements():
                    deepnet = ipt.get_last().get_deepnet()
                    if last_deepnet is not None:
                        if not id(deepnet) == id(last_deepnet):
                            raise RuntimeError("Elements of cell input have different deepnets. "
                                               "Cannot infer implicit deepnet")
                    last_deepnet = deepnet
                self._deepnet = last_deepnet
            elif isinstance(inputs, n2d2.deepnet.Sequence) or isinstance(inputs, Cell):
                self._deepnet = inputs.get_last().get_deepnet()
            elif isinstance(inputs, n2d2.provider.Provider):
                self._deepnet = n2d2.deepnet.DeepNet()
            else:
                raise TypeError("Object of type " + str(type(inputs)) + " cannot implicitly provide a deepNet to cell.")

        if 'model' in config_parameters:
            self._model = config_parameters.pop('model')
        else:
            self._model = n2d2.global_variables.default_model
        if 'dataType' in config_parameters:
            self._datatype = config_parameters.pop('dataType')
        else:
            self._datatype = n2d2.global_variables.default_dataType

        self._model_key = self._model + '<' + self._datatype + '>'

        N2D2_Interface.__init__(self, **config_parameters)

        self._constructor_arguments.update({
            'nbOutputs': nbOutputs,
        })

        self._connection_parameters = {}

    def initialize(self):
        self._N2D2_object.initialize()

    def get_outputs(self):
        return self._N2D2_object.getOutputs()

    def get_first(self):
        return self

    def get_last(self):
        return self

    def get_deepnet(self):
        return self._deepnet


    def get_type(self):
        if self._type is None:
            return self._N2D2_object.getType()
        else:
            return self._type

    def add_input(self, inputs):
        if isinstance(inputs, list):
            for cell in inputs:
                self.add_input(cell)
        elif isinstance(inputs, n2d2.deepnet.Sequence):
            self.add_input(inputs.get_last())
        elif isinstance(inputs, n2d2.deepnet.Layer):
            for cell in inputs.get_elements():
                self.add_input(cell)
        elif isinstance(inputs, Cell) or isinstance(inputs, n2d2.provider.Provider) or isinstance(inputs, n2d2.tensor.Tensor):
            self._link_N2D2_input(inputs)
            self._inputs.append(inputs)
        else:
            raise TypeError("Cannot add object of type " + str(type(inputs)))

    """
    Links N2D2 cells tacking into account cell connection parameters
    """
    # TODO: Simply connection parameters
    # TODO: Add to add_input
    def _link_N2D2_input(self, inputs):
        if isinstance(inputs, n2d2.tensor.Tensor):
            # @Johannes : The method to call is addInputBis (to rename ?).
            # I have renamed this method in the binding because it's not defined in the same file as the other addInput method. 
            # Thus Pybind overwrite it.
            # I also changed the the diffOutput Tensor so that it has the good dimensions !
            # Reminder : shape get the pythonic dimensions and dims give the N2D2 dimensions.
            # Here you called the n2d2 constructor so you want to have the pythonic dimensions.
            # If you called the N2D2 constructor you want to use the dims() method. 
            self._N2D2_object.addInputBis(inputs.N2D2(), n2d2.tensor.Tensor(inputs.shape()).N2D2(), value=0) # TODO: Tested
        else:
            if 'mapping' in self._connection_parameters:
                if isinstance(inputs, n2d2.cell.Cell):
                    dim_z = inputs.N2D2().getNbOutputs()
                elif isinstance(inputs, n2d2.provider.Provider):
                    dim_z = inputs.N2D2().getNbChannels()
                else:
                    raise ValueError("Incompatible inputs of type " + str(type(inputs)))
                self._connection_parameters['mapping'] = self._connection_parameters['mapping'].create_N2D2_mapping(
                                               dim_z,
                                               self._N2D2_object.getNbOutputs()
                                           ).N2D2()
            self._N2D2_object.addInput(inputs.N2D2(), **self._connection_parameters)

            if isinstance(inputs, n2d2.provider.Provider):
                self._deepnet.add_provider(inputs)


    def _link_to_N2D2_deepnet(self):
        parents = []
        for ipt in self.get_inputs():
            if not isinstance(ipt, n2d2.provider.Provider):
                parents.append(ipt.get_last().N2D2())
        self._deepnet.N2D2().addCell(self._N2D2_object, parents)


    def _add_to_graph(self, inputs):
        self.add_input(inputs)
        self._link_to_N2D2_deepnet()
        self._N2D2_object.initialize()

    """
    def initialize(self):
        #self._N2D2_object.clearInputs()
        #for cell in self._inputs:
        #    self._link_N2D2_input(cell)
        self._N2D2_object.initialize()
    """

    def set_activation(self, activation):
        print("Note: Replacing potentially existing activation in cell: " + self.get_name())
        self._config_parameters['activation'] = activation
        self._N2D2_object.setActivation(self._config_parameters['activation'].N2D2())

    def set_activation_quantizer(self, quantizer):
        self._N2D2_object.getActivation().setQuantizer(quantizer.N2D2())
        # TODO: Create n2d2 objects to obtain visibility of objects in API


    def get_deepnet(self):
        return self._deepnet

    def get_inputs(self):
        return self._inputs

    def clear_input(self):
        self._inputs = []
        self._N2D2_object.clearInputs()

    def propagate(self, inference=False):
        self._N2D2_object.propagate(inference)

    def back_propagate(self):
        self._N2D2_object.backPropagate()

    def update(self):
        self._N2D2_object.update()

    def import_free_parameters(self, fileName, **kwargs):
        print("import " + fileName)
        self._N2D2_object.importFreeParameters(fileName, **kwargs)

    def import_activation_parameters(self, fileName, **kwargs):
        print("import " + fileName)
        self._N2D2_object.importActivationParameters(fileName, **kwargs)

    def get_name(self):
        return self._name

    def get_nb_outputs(self):
        return self._N2D2_object.getNbOutputs()

    def _sync_inputs_and_parents(self, inputs):
        parents = self._N2D2_object.getParentsCells()
        # Necessary because N2D2 returns [None] if no parents
        if parents[0] is None:
            parents = []
        if not len(inputs) == len(parents):
            raise RuntimeError("Number of given inputs " + str(len(inputs)) +
                               " is different than number of N2D2 parent cells " +
                               str(len(self._N2D2_object.getParentsCells())) +
                               ". Did you provide the correct input cells?")

        for idx, ipt in enumerate(inputs):
            if not ipt.get_name() == parents[idx].getName():
                print(
                    "Warning: input cell name and N2D2 corresponding parent cell name do not match. Are you connecting the right cell?")
            self._inputs.append(ipt)

    def __str__(self):
        output = "\'" + self.get_name() + "\' " + self.get_type()+"(" + self._model_key + ")"
        output += N2D2_Interface.__str__(self)
        if len(self.get_inputs()) > 0:
            output += "[inputs="
            for idx, cell in enumerate(self.get_inputs()):
                if idx > 0:
                    output += ", "
                output += "'" + cell.get_name() + "'"
            output += "]"
        else:
            output += "[inputs=[]]"
        return output



class Fc(Cell):

    _cell_constructors = {
            'Frame<float>': N2D2.FcCell_Frame_float,
            'Frame_CUDA<float>': N2D2.FcCell_Frame_CUDA_float,
    }

    def __init__(self, inputs, nbOutputs, N2D2_object=None, **config_parameters):
        # TODO : Add description for filler and solver.
        """
        :param nbOutputs: Number of outputs of the cell.
        :type nbOutputs: int
        :param name: Name fo the cell.
        :type name: str
        :param activationFunction: Activation function used by the cell.
        :type activationFunction: :py:class:`n2d2.activation.Activation`, optional
        :param weightsSolver: TODO
        :type weightsSolver: :py:class:`n2d2.solver.Solver`, optional
        :param biasSolver: TODO
        :type biasSolver: :py:class:`n2d2.solver.Solver`, optional
        :param weightsFiller: TODO
        :type weightsFiller: :py:class:`n2d2.filler.Filler`, optional
        :param biasFiller: TODO
        :type biasFiller: :py:class:`n2d2.filler.Filler`, optional
        """

        if N2D2_object is not None and (nbOutputs is not None or len(config_parameters) > 0):
            raise RuntimeError("N2D2_object argument give to cell but 'inputs' or 'nbOutputs' or 'config parameters' not None")
        if N2D2_object is None:
            self._create_from_arguments(inputs, nbOutputs, **config_parameters)
        else:
            self._create_from_N2D2_object(inputs, N2D2_object)


    def _create_from_arguments(self, inputs, nbOutputs, **config_parameters):
        Cell.__init__(self, inputs, nbOutputs, **config_parameters)

        self._N2D2_object = self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                                     self.get_name(),
                                                                     self._constructor_arguments['nbOutputs'])
        """Set connection and mapping parameters"""
        for key in self._config_parameters:
            if key is 'inputOffsetX':
                self._connection_parameters['x0'] = self._config_parameters.pop('inputOffsetX')
            elif key is 'inputOffsetY':
                self._connection_parameters['y0'] = self._config_parameters.pop('inputOffsetY')
            elif key is 'inputWidth':
                self._connection_parameters['width'] = self._config_parameters.pop('inputWidth')
            elif key is 'inputHeight':
                self._connection_parameters['height'] = self._config_parameters.pop('inputHeight')

        """Set and initialize here all complex cell members"""
        for key, value in self._config_parameters.items():
            if key is 'activationFunction':
                self._N2D2_object.setActivation(value.N2D2())
            elif key is 'weightsSolver':
                self._N2D2_object.setWeightsSolver(value.N2D2())
            elif key is 'biasSolver':
                self._N2D2_object.setBiasSolver(value.N2D2())
            elif key is 'weightsFiller':
                self._N2D2_object.setWeightsFiller(value.N2D2())
            elif key is 'biasFiller':
                self._N2D2_object.setBiasFiller(value.N2D2())
            elif key is 'quantizer':
                self._N2D2_object.setQuantizer(value.N2D2())
            else:
                self._set_N2D2_parameter(self._param_to_INI_convention(key), value)

        self._add_to_graph(inputs)

    def _create_from_N2D2_object(self, inputs, N2D2_object):

        Cell.__init__(self, inputs,
                      N2D2_object.getNbOutputs(),
                      deepNet=n2d2.deepnet.DeepNet(N2D2_object=N2D2_object.getAssociatedDeepNet()),
                      name=N2D2_object.getName(),
                      **self._load_N2D2_parameters(N2D2_object))

        self._N2D2_object = N2D2_object

        # NOTE: No Fillers because existing cell
        # TODO: Add similar methods to Activation/Solver/Quantizer for nice prints
        self._config_parameters['activationFunction'] = self._N2D2_object.getActivation()
        self._config_parameters['weightsSolver'] = self._N2D2_object.getWeightsSolver()
        self._config_parameters['biasSolver'] = self._N2D2_object.getBiasSolver()
        self._config_parameters['weightsSolver'] = self._N2D2_object.getWeightsSolver()
        self._config_parameters['quantizer'] = self._N2D2_object.getQuantizer()

        self._sync_inputs_and_parents(inputs)

    # TODO: This is not working as expected because solvers are copied in a vector at cell initialization.
    #  setWeightsSolver sets only the solver to be copied but does not modify after cell initialization
    """
    def set_weights_solver(self, solver):
        self._config_parameters['weightsSolver'] = solver
        self._N2D2_object.setWeightsSolver(self._config_parameters['weightsSolver'].N2D2())
    """

    def set_bias_solver(self, solver):
        self._config_parameters['biasSolver'] = solver
        self._N2D2_object.setBiasSolver(self._config_parameters['biasSolver'].N2D2())

    def set_quantizer(self, quantizer):
        self._config_parameters['quantizer'] = quantizer
        self._N2D2_object.setQuantizer(self._config_parameters['quantizer'].N2D2())


    # TODO: General set_solver that copies solver and does both




# TODO: This is less powerful than the generator, in the sense that it does not accept several formats for the stride, conv, etc.
class Conv(Cell):

    _cell_constructors = {
        'Frame<float>': N2D2.ConvCell_Frame_float,
        'Frame_CUDA<float>': N2D2.ConvCell_Frame_CUDA_float,
        'Frame<double>': N2D2.ConvCell_Frame_double,
        'Frame_CUDA<double>': N2D2.ConvCell_Frame_CUDA_double,
    }

    def __init__(self,
                 inputs,
                 nbOutputs,
                 kernelDims,
                 N2D2_object=None,
                 **config_parameters):
        """
        :param nbOutputs: Number of outputs of the cell.
        :type nbOutputs: int
        :param name: Name fo the cell.
        :type name: str
        :param kernelDims: Kernel dimension.
        :type kernelDims: list
        :param subSampleDims: TODO
        :type subSampleDims: list, optional
        :param strideDims: TODO
        :type strideDims: list, optional
        :param paddingDims: TODO
        :type paddingDims: list, optional
        :param dilationDims: TODO
        :type dilationDims: list, optional     
        """

        if N2D2_object is not None and (nbOutputs is not None or kernelDims is not None or len(config_parameters) > 0):
            raise RuntimeError("N2D2_object argument give to cell but 'inputs' or 'nbOutputs'  or 'kernelDims' or 'config parameters' not None")
        if N2D2_object is None:
            self._create_from_arguments(inputs, nbOutputs, kernelDims, **config_parameters)
        else:
            self._create_from_N2D2_object(inputs, N2D2_object)

    def _create_from_arguments(self, inputs, nbOutputs, kernelDims, **config_parameters):

        Cell.__init__(self, inputs, nbOutputs, **config_parameters)

        self._constructor_arguments.update({
            'kernelDims': kernelDims,
        })

        self._parse_optional_arguments(['subSampleDims', 'strideDims', 'paddingDims', 'dilationDims'])

        self._N2D2_object = self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                                     self.get_name(),
                                                                     self._constructor_arguments['kernelDims'],
                                                                     self._constructor_arguments['nbOutputs'],
                                                                     **self._optional_constructor_arguments)


        """Set connection and mapping parameters"""
        if 'mapping' in self._config_parameters:
            self._connection_parameters['mapping'] = self._config_parameters.pop('mapping')

        # TODO: Add Kernel section of generator

        """Set and initialize here all complex cell members"""
        for key, value in self._config_parameters.items():
            if key is 'activationFunction':
                self._N2D2_object.setActivation(value.N2D2())
            elif key is 'weightsSolver':
                self._N2D2_object.setWeightsSolver(value.N2D2())
            elif key is 'biasSolver':
                self._N2D2_object.setBiasSolver(value.N2D2())
            elif key is 'weightsFiller':
                self._N2D2_object.setWeightsFiller(value.N2D2())
            elif key is 'biasFiller':
                self._N2D2_object.setBiasFiller(value.N2D2())
            elif key is 'quantizer':
                self._N2D2_object.setQuantizer(value.N2D2())
            else:
                self._set_N2D2_parameter(self._param_to_INI_convention(key), value)

        self._add_to_graph(inputs)


    def _create_from_N2D2_object(self, inputs, N2D2_object):

        Cell.__init__(self, inputs,
                      N2D2_object.getNbOutputs(),
                      deepNet=n2d2.deepnet.DeepNet(N2D2_object=N2D2_object.getAssociatedDeepNet()),
                      name=N2D2_object.getName(),
                      **self._load_N2D2_parameters(N2D2_object))

        self._N2D2_object = N2D2_object

        self._constructor_arguments['kernelDims'] = [self._N2D2_object.getKernelWidth(), self._N2D2_object.getKernelHeight()]
        self._optional_constructor_arguments['subSampleDims'] = [self._N2D2_object.getSubSampleX(), self._N2D2_object.getSubSampleY()]
        self._optional_constructor_arguments['strideDims'] = [self._N2D2_object.getStrideX(), self._N2D2_object.getStrideY()]
        self._optional_constructor_arguments['paddingDims'] = [self._N2D2_object.getPaddingX(), self._N2D2_object.getPaddingY()]
        self._optional_constructor_arguments['dilationDims'] = [self._N2D2_object.getDilationX(), self._N2D2_object.getDilationY()]


        # NOTE: No Fillers because existing cell
        # TODO: Add similar methods to Activation/Solver/Quantizer for nice prints
        self._config_parameters['activationFunction'] = self._N2D2_object.getActivation()
        self._config_parameters['weightsSolver'] = self._N2D2_object.getWeightsSolver()
        self._config_parameters['biasSolver'] = self._N2D2_object.getBiasSolver()
        self._config_parameters['weightsSolver'] = self._N2D2_object.getWeightsSolver()
        self._config_parameters['quantizer'] = self._N2D2_object.getQuantizer()

        self._sync_inputs_and_parents(inputs)

    # TODO: This is not working as expected because solvers are copied in a vector at cell initialization.
    #  setWeightsSolver sets only the solver to be copied but does not modify after cell initialization
    """
    def set_weights_solver(self, solver):
        self._config_parameters['weightsSolver'] = solver
        self._N2D2_object.setWeightsSolver(self._config_parameters['weightsSolver'].N2D2())
    """

    def set_bias_solver(self, solver):
        self._config_parameters['biasSolver'] = solver
        self._N2D2_object.setBiasSolver(self._config_parameters['biasSolver'].N2D2())

    def set_quantizer(self, quantizer):
        self._config_parameters['quantizer'] = quantizer
        self._N2D2_object.setQuantizer(self._config_parameters['quantizer'].N2D2())



class ConvDepthWise(Conv):
    _type = 'ConvDepthWise'

    def __init__(self,
                 inputs,
                 kernelDims,
                 **config_parameters):

        if 'mapping' in config_parameters:
            raise RuntimeError('ConvDepthWise does not support custom mappings')
        else:
            config_parameters['mapping'] = n2d2.mapping.Mapping(nbChannelsPerGroup=1)
        Conv.__init__(self, inputs, inputs.get_outputs().dimZ(), kernelDims, **config_parameters)


class ConvPointWise(Conv):
    _type = 'ConvPointWise'

    def __init__(self,
                 inputs,
                 nbOutputs,
                 **config_parameters):

        if 'mapping' in config_parameters:
            raise RuntimeError('ConvDepthWise does not support custom mappings')
        Conv.__init__(self, inputs, nbOutputs, [1, 1], strideDims=[1, 1], **config_parameters)


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
                 nbOutputs,
                 kernelDims,
                 N2D2_object=None,
                 **config_parameters):
        """
        :param nbOutputs: Number of outputs of the cell.
        :type nbOutputs: int
        :param name: Name fo the cell.
        :type name: str
        :param kernelDims: Kernel dimension.
        :type kernelDims: list
        :param subSampleDims: TODO
        :type subSampleDims: list, optional
        :param strideDims: TODO
        :type strideDims: list, optional
        :param paddingDims: TODO
        :type paddingDims: list, optional
        :param dilationDims: TODO
        :type dilationDims: list, optional
        """

        if N2D2_object is not None and (nbOutputs is not None or kernelDims is not None or len(config_parameters) > 0):
            raise RuntimeError("N2D2_object argument give to cell but 'inputs' or 'nbOutputs' or 'config parameters' not None")
        if N2D2_object is None:
            self._create_from_arguments(inputs, nbOutputs, kernelDims, **config_parameters)
        else:
            self._create_from_N2D2_object(inputs, N2D2_object)

    def _create_from_arguments(self, inputs, nbOutputs, kernelDims, **config_parameters):

        Cell.__init__(self, inputs, nbOutputs, **config_parameters)

        self._constructor_arguments.update({
            'kernelDims': kernelDims,
        })

        self._parse_optional_arguments(['strideDims', 'paddingDims', 'dilationDims'])

        self._N2D2_object = self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                                     self.get_name(),
                                                                     self._constructor_arguments['kernelDims'],
                                                                     self._constructor_arguments['nbOutputs'],
                                                                     **self._optional_constructor_arguments)


        """Set connection and mapping parameters"""
        if 'mapping' in self._config_parameters:
            self._connection_parameters['mapping'] = self._config_parameters.pop('mapping')

        # TODO: Add Kernel section of generator

        """Set and initialize here all complex cell members"""
        for key, value in self._config_parameters.items():
            if key is 'activationFunction':
                self._N2D2_object.setActivation(value.N2D2())
            elif key is 'weightsSolver':
                self._N2D2_object.setWeightsSolver(value.N2D2())
            elif key is 'biasSolver':
                self._N2D2_object.setBiasSolver(value.N2D2())
            elif key is 'weightsFiller':
                self._N2D2_object.setWeightsFiller(value.N2D2())
            elif key is 'biasFiller':
                self._N2D2_object.setBiasFiller(value.N2D2())
            else:
                self._set_N2D2_parameter(self._param_to_INI_convention(key), value)

        self._add_to_graph(inputs)


    def _create_from_N2D2_object(self, inputs, N2D2_object):

        Cell.__init__(self, inputs,
                      N2D2_object.getNbOutputs(),
                      deepNet=n2d2.deepnet.DeepNet(N2D2_object=N2D2_object.getAssociatedDeepNet()),
                      name=N2D2_object.getName(),
                      **self._load_N2D2_parameters(N2D2_object))

        self._N2D2_object = N2D2_object

        self._constructor_arguments['kernelDims'] = [self._N2D2_object.getKernelWidth(), self._N2D2_object.getKernelHeight()]
        self._optional_constructor_arguments['strideDims'] = [self._N2D2_object.getStrideX(), self._N2D2_object.getStrideY()]
        self._optional_constructor_arguments['paddingDims'] = [self._N2D2_object.getPaddingX(), self._N2D2_object.getPaddingY()]
        self._optional_constructor_arguments['dilationDims'] = [self._N2D2_object.getDilationX(), self._N2D2_object.getDilationY()]


        # NOTE: No Fillers because existing cell
        # TODO: Add similar methods to Activation/Solver/Quantizer for nice prints
        self._config_parameters['activationFunction'] = self._N2D2_object.getActivation()
        self._config_parameters['weightsSolver'] = self._N2D2_object.getWeightsSolver()
        self._config_parameters['biasSolver'] = self._N2D2_object.getBiasSolver()
        self._config_parameters['weightsSolver'] = self._N2D2_object.getWeightsSolver()

        self._sync_inputs_and_parents(inputs)

    # TODO: This is not working as expected because solvers are copied in a vector at cell initialization.
    #  setWeightsSolver sets only the solver to be copied but does not modify after cell initialization
    """
    def set_weights_solver(self, solver):
        self._config_parameters['weightsSolver'] = solver
        self._N2D2_object.setWeightsSolver(self._config_parameters['weightsSolver'].N2D2())
    """

    def set_bias_solver(self, solver):
        self._config_parameters['biasSolver'] = solver
        self._N2D2_object.setBiasSolver(self._config_parameters['biasSolver'].N2D2())



class ElemWise(Cell):

    _cell_constructors = {
        'Frame': N2D2.ElemWiseCell_Frame,
        'Frame_CUDA': N2D2.ElemWiseCell_Frame_CUDA,
    }

    def __init__(self, inputs, nbOutputs,  N2D2_object=None, **config_parameters):

        if N2D2_object is not None and (nbOutputs is not None or len(config_parameters) > 0):
            raise RuntimeError(
                "N2D2_object argument give to cell but 'inputs' or 'nbOutputs' or 'config parameters' not None")
        if N2D2_object is None:
            self._create_from_arguments(inputs, nbOutputs, **config_parameters)
        else:
            self._create_from_N2D2_object(inputs, N2D2_object)


    def _create_from_arguments(self, inputs, nbOutputs, **config_parameters):
        Cell.__init__(self, inputs, nbOutputs, **config_parameters)

        self._parse_optional_arguments(['operation', 'weights', 'shifts'])

        self._optional_constructor_arguments['operation'] = \
            N2D2.ElemWiseCell.Operation.__members__[self._optional_constructor_arguments['operation']]

        self._N2D2_object = self._cell_constructors[self._model](self._deepnet.N2D2(),
                                                self.get_name(),
                                                self._constructor_arguments['nbOutputs'],
                                                **self._optional_constructor_arguments)

        """Set and initialize here all complex cell members"""
        for key, value in self._config_parameters.items():
            if key is 'activationFunction':
                self._N2D2_object.setActivation(value.N2D2())
            else:
                self._set_N2D2_parameter(self._param_to_INI_convention(key), value)

        self._add_to_graph(inputs)

    def _create_from_N2D2_object(self, inputs, N2D2_object):

        Cell.__init__(self, inputs,
                      N2D2_object.getNbOutputs(),
                      deepNet=n2d2.deepnet.DeepNet(N2D2_object=N2D2_object.getAssociatedDeepNet()),
                      name=N2D2_object.getName(),
                      **self._load_N2D2_parameters(N2D2_object))

        self._N2D2_object = N2D2_object

        self._optional_constructor_arguments['operation'] = self._N2D2_object.getOperation()
        self._optional_constructor_arguments['weights'] = self._N2D2_object.getWeights()
        self._optional_constructor_arguments['shifts'] = self._N2D2_object.getShifts()

        # NOTE: No Fillers because existing cell
        # TODO: Add similar methods to Activation/Solver/Quantizer for nice prints
        self._config_parameters['activationFunction'] = self._N2D2_object.getActivation()

        self._sync_inputs_and_parents(inputs)


class Softmax(Cell):

    _cell_constructors = {
        'Frame<float>': N2D2.SoftmaxCell_Frame_float,
        'Frame_CUDA<float>': N2D2.SoftmaxCell_Frame_CUDA_float,
        'Frame<double>': N2D2.SoftmaxCell_Frame_double,
        'Frame_CUDA<double>': N2D2.SoftmaxCell_Frame_CUDA_double,
    }

    def __init__(self, inputs,  N2D2_object=None, **config_parameters):

        if N2D2_object is not None and len(config_parameters) > 0:
            raise RuntimeError(
                "N2D2_object argument give to cell but 'inputs' or 'nbOutputs' or 'config parameters' not None")
        if N2D2_object is None:
            self._create_from_arguments(inputs, **config_parameters)
        else:
            self._create_from_N2D2_object(inputs, N2D2_object)

    def _create_from_arguments(self, inputs, **config_parameters):
        Cell.__init__(self, inputs, inputs.get_nb_outputs(), **config_parameters)

        self._parse_optional_arguments(['withLoss', 'groupSize'])
        self._N2D2_object = self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                                     self.get_name(),
                                                                     self._constructor_arguments['nbOutputs'],
                                                                     **self._optional_constructor_arguments)

        # Delete to avoid print
        del self._constructor_arguments['nbOutputs']

        """Set and initialize here all complex cell members"""
        for key, value in self._config_parameters.items():
            if key is 'activationFunction':
                self._N2D2_object.setActivation(value.N2D2())
            else:
                self._set_N2D2_parameter(self._param_to_INI_convention(key), value)

        self._add_to_graph(inputs)


    def _create_from_N2D2_object(self, inputs, N2D2_object):

        Cell.__init__(self, inputs,
                      N2D2_object.getNbOutputs(),
                      deepNet=n2d2.deepnet.DeepNet(N2D2_object=N2D2_object.getAssociatedDeepNet()),
                      name=N2D2_object.getName(),
                      **self._load_N2D2_parameters(N2D2_object))

        self._N2D2_object = N2D2_object

        self._optional_constructor_arguments['withLoss'] = self._N2D2_object.getWithLoss()
        self._optional_constructor_arguments['groupSize'] = self._N2D2_object.getGroupSize()

        self._config_parameters['activationFunction'] = self._N2D2_object.getActivation()

        self._sync_inputs_and_parents(inputs)



class Dropout(Cell):
    _cell_constructors = {
        'Frame<float>': N2D2.DropoutCell_Frame_float,
        'Frame_CUDA<float>': N2D2.DropoutCell_Frame_CUDA_float,
        'Frame<double>': N2D2.DropoutCell_Frame_double,
        'Frame_CUDA<double>': N2D2.DropoutCell_Frame_CUDA_double,
    }
    def __init__(self, inputs, N2D2_object=None, **config_parameters):

        if N2D2_object is not None and  len(config_parameters) > 0:
            raise RuntimeError(
                "N2D2_object argument give to cell but 'inputs' or 'nbOutputs' or 'config parameters' not None")
        if N2D2_object is None:
            self._create_from_arguments(inputs, **config_parameters)
        else:
            self._create_from_N2D2_object(inputs, N2D2_object)

    def _create_from_arguments(self, inputs, **config_parameters):
        Cell.__init__(self, inputs, inputs.get_nb_outputs(), **config_parameters)
        # No optionnal args
        self._parse_optional_arguments([])
        self._N2D2_object = self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                        self.get_name(),
                                                        self._constructor_arguments['nbOutputs'],
                                                        **self._optional_constructor_arguments)

        # Delete to avoid print
        del self._constructor_arguments['nbOutputs']

        """Set and initialize here all complex cell members"""
        for key, value in self._config_parameters.items():
            if key is 'activationFunction':
                self._N2D2_object.setActivation(value.N2D2())
            else:
                self._set_N2D2_parameter(self._param_to_INI_convention(key), value)

        self._add_to_graph(inputs)


    def _create_from_N2D2_object(self, inputs, N2D2_object):
        Cell.__init__(self, inputs,
                      N2D2_object.getNbOutputs(),
                      deepNet=n2d2.deepnet.DeepNet(N2D2_object=N2D2_object.getAssociatedDeepNet()),
                      name=N2D2_object.getName(),
                      **self._load_N2D2_parameters(N2D2_object))

        self._N2D2_object = N2D2_object

        self._config_parameters['activationFunction'] = self._N2D2_object.getActivation()

        self._sync_inputs_and_parents(inputs)


class Padding(Cell):

    _cell_constructors = {
        'Frame': N2D2.PaddingCell_Frame,
        'Frame_CUDA': N2D2.PaddingCell_Frame_CUDA,
    }

    def __init__(self,
                 inputs,
                 nbOutputs,
                 topPad,
                 botPad,
                 leftPad,
                 rightPad,
                 N2D2_object=None,
                 **config_parameters):

        if N2D2_object is not None and (nbOutputs is not None
                                        or topPad is not None
                                        or botPad is not None
                                        or leftPad is not None
                                        or rightPad is not None
                                        or len(config_parameters) > 0):
            raise RuntimeError(
                "N2D2_object argument give to cell but 'inputs' or 'nbOutputs' or 'config parameters' not None")
        if N2D2_object is None:
            self._create_from_arguments(inputs, nbOutputs, topPad, botPad, leftPad, rightPad, **config_parameters)
        else:
            self._create_from_N2D2_object(inputs, N2D2_object)

    def _create_from_arguments(self, inputs, nbOutputs, topPad, botPad, leftPad, rightPad, **config_parameters):

        Cell.__init__(self, inputs, nbOutputs, **config_parameters)

        self._constructor_arguments.update({
                 'topPad': topPad,
                 'botPad': botPad,
                 'leftPad': leftPad,
                 'rightPad': rightPad
        })

        # No optional args
        self._parse_optional_arguments([])

        self._N2D2_object = self._cell_constructors[self._model](self._deepnet.N2D2(),
                                                                     self.get_name(),
                                                                     self._constructor_arguments['nbOutputs'],
                                                                     self._constructor_arguments['topPad'],
                                                                     self._constructor_arguments['botPad'],
                                                                     self._constructor_arguments['leftPad'],
                                                                     self._constructor_arguments['rightPad'],
                                                                     **self._optional_constructor_arguments)
        """Set and initialize here all complex cell members"""
        for key, value in self._config_parameters.items():
            if key is 'activationFunction':
                self._N2D2_object.setActivation(value.N2D2())
            else:
                self._set_N2D2_parameter(self._param_to_INI_convention(key), value)

        self._add_to_graph(inputs)

    def _create_from_N2D2_object(self, inputs, N2D2_object):
        Cell.__init__(self, inputs,
                      N2D2_object.getNbOutputs(),
                      deepNet=n2d2.deepnet.DeepNet(N2D2_object=N2D2_object.getAssociatedDeepNet()),
                      name=N2D2_object.getName(),
                      **self._load_N2D2_parameters(N2D2_object))

        self._N2D2_object = N2D2_object

        self._constructor_arguments['topPad'] = self._N2D2_object.getTopPad()
        self._constructor_arguments['botPad'] = self._N2D2_object.getBotPad()
        self._constructor_arguments['leftPad'] = self._N2D2_object.getLeftPad()
        self._constructor_arguments['rightPad'] = self._N2D2_object.getRightPad()

        self._config_parameters['activationFunction'] = self._N2D2_object.getActivation()

        self._sync_inputs_and_parents(inputs)


class Pool(Cell):
    _type = 'Pool'

    _cell_constructors = {
        'Frame<float>': N2D2.PoolCell_Frame_float,
        'Frame_CUDA<float>': N2D2.PoolCell_Frame_CUDA_float,
    }
    
    def __init__(self,
                 inputs,
                 nbOutputs,
                 poolDims,
                 N2D2_object=None,
                 **config_parameters):

        if N2D2_object is not None and (nbOutputs is not None or poolDims is not None or len(config_parameters) > 0):
            raise RuntimeError(
                "N2D2_object argument give to cell but 'inputs' or 'nbOutputs' or 'config parameters' not None")
        if N2D2_object is None:
            self._create_from_arguments(inputs, nbOutputs, poolDims, **config_parameters)
        else:
            self._create_from_N2D2_object(inputs, N2D2_object)

    def _create_from_arguments(self, inputs, nbOutputs, poolDims, **config_parameters):
        Cell.__init__(self, inputs, nbOutputs, **config_parameters)

        self._constructor_arguments.update({
            'poolDims': poolDims,
        })


        # Note: Removed Pooling
        self._parse_optional_arguments(['strideDims', 'paddingDims', 'pooling'])

        self._optional_constructor_arguments['pooling'] = \
            N2D2.PoolCell.Pooling.__members__[self._optional_constructor_arguments['pooling']]


        """Set connection and mapping parameters"""
        if 'mapping' in self._config_parameters:
            self._connection_parameters['mapping'] = self._config_parameters.pop('mapping')


        self._N2D2_object = self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                                     self.get_name(),
                                                                     self._constructor_arguments['poolDims'],
                                                                     self._constructor_arguments['nbOutputs'],
                                                                     **self._optional_constructor_arguments)

        """Set and initialize here all complex cell members"""
        for key, value in self._config_parameters.items():
            if key is 'activationFunction':
                self._N2D2_object.setActivation(value.N2D2())
            else:
                self._set_N2D2_parameter(self._param_to_INI_convention(key), value)

        self._add_to_graph(inputs)

    def _create_from_N2D2_object(self, inputs, N2D2_object):
        Cell.__init__(self, inputs,
                      N2D2_object.getNbOutputs(),
                      deepNet=n2d2.deepnet.DeepNet(N2D2_object=N2D2_object.getAssociatedDeepNet()),
                      name=N2D2_object.getName(),
                      **self._load_N2D2_parameters(N2D2_object))

        self._N2D2_object = N2D2_object

        self._constructor_arguments['poolDims'] = [self._N2D2_object.getPoolWidth(),
                                                     self._N2D2_object.getPoolHeight()]
        self._optional_constructor_arguments['strideDims'] = [self._N2D2_object.getStrideX(),
                                                              self._N2D2_object.getStrideY()]
        self._optional_constructor_arguments['paddingDims'] = [self._N2D2_object.getPaddingX(),
                                                               self._N2D2_object.getPaddingY()]
        self._optional_constructor_arguments['pooling'] = self._N2D2_object.getPooling()

        self._config_parameters['activationFunction'] = self._N2D2_object.getActivation()

        self._sync_inputs_and_parents(inputs)


class Pool2D(Pool):
    _type = 'Pool2D'
    def __init__(self,
                 inputs,
                 poolDims,
                 **config_parameters):
        if 'mapping' in config_parameters:
            raise RuntimeError('Pool2D does not support custom mappings')
        else:
            config_parameters['mapping'] = n2d2.mapping.Mapping(nbChannelsPerGroup=1)
        Pool.__init__(self, inputs, inputs.get_outputs().dimZ(), poolDims, **config_parameters)


class GlobalPool2D(Pool2D):
    _type = 'GlobalPool2D'
    def __init__(self,
                 inputs,
                 **config_parameters):
        Pool2D.__init__(self, inputs, [inputs.get_outputs().dimX(), inputs.get_outputs().dimY()],
                        strideDims=[1, 1], **config_parameters)



class LRN(Cell):
    _cell_constructors = {
        'Frame<float>': N2D2.LRNCell_Frame_float,
        'Frame_CUDA<float>': N2D2.LRNCell_Frame_CUDA_float,
    }


    def __init__(self, inputs, nbOutputs, N2D2_object=None, **config_parameters):

        if N2D2_object is not None and (nbOutputs is not None or len(config_parameters) > 0):
            raise RuntimeError(
                "N2D2_object argument give to cell but 'inputs' or 'nbOutputs' or 'config parameters' not None")
        if N2D2_object is None:
            self._create_from_arguments(inputs, nbOutputs, **config_parameters)
        else:
            self._create_from_N2D2_object(inputs, N2D2_object)

    def _create_from_arguments(self, inputs, nbOutputs, **config_parameters):
        Cell.__init__(self, inputs, nbOutputs, N2D2_object=None, **config_parameters)

        # No optional args
        self._parse_optional_arguments([])
        self._N2D2_object = self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                self.get_name(),
                                                self._constructor_arguments['nbOutputs'],
                                                **self._optional_constructor_arguments)

        """Set and initialize here all complex cell members"""
        for key, value in self._config_parameters.items():
            if key is 'activationFunction':
                self._N2D2_object.setActivation(value.N2D2())
            else:
                self._set_N2D2_parameter(self._param_to_INI_convention(key), value)

        self._add_to_graph(inputs)

    def _create_from_N2D2_object(self, inputs, N2D2_object):
        Cell.__init__(self, inputs,
                      N2D2_object.getNbOutputs(),
                      deepNet=n2d2.deepnet.DeepNet(N2D2_object=N2D2_object.getAssociatedDeepNet()),
                      name=N2D2_object.getName(),
                      **self._load_N2D2_parameters(N2D2_object))

        self._N2D2_object = N2D2_object

        self._config_parameters['activationFunction'] = self._N2D2_object.getActivation()

        self._sync_inputs_and_parents(inputs)

# TODO: make nbOutputs implicit on input
class BatchNorm(Cell):
    _cell_constructors = {
        'Frame<float>': N2D2.BatchNormCell_Frame_float,
        'Frame_CUDA<float>': N2D2.BatchNormCell_Frame_CUDA_float,
    }
    def __init__(self, inputs, N2D2_object=None, **config_parameters):
        if N2D2_object is not None and len(config_parameters) > 0:
            raise RuntimeError(
                "N2D2_object argument give to cell but 'inputs' or 'nbOutputs' or 'config parameters' not None")
        if N2D2_object is None:
            self._create_from_arguments(inputs, **config_parameters)
        else:
            self._create_from_N2D2_object(inputs, N2D2_object)

    def _create_from_arguments(self, inputs, **config_parameters):
        Cell.__init__(self, inputs, inputs.get_nb_outputs(), **config_parameters)
        # No optional parameter
        self._parse_optional_arguments([])
        self._N2D2_object = self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                self.get_name(),
                                                self._constructor_arguments['nbOutputs'],
                                                **self._optional_constructor_arguments)

        # Delete to avoid print
        del self._constructor_arguments['nbOutputs']

        """Set and initialize here all complex cell members"""
        for key, value in self._config_parameters.items():
            if key is 'activationFunction':
                self._N2D2_object.setActivation(value.N2D2())
            elif key is 'scaleSolver':
                self._N2D2_object.setScaleSolver(value.N2D2())
            elif key is 'biasSolver':
                self._N2D2_object.setBiasSolver(value.N2D2())
            else:
                self._set_N2D2_parameter(self._param_to_INI_convention(key), value)

        self._add_to_graph(inputs)

    def _create_from_N2D2_object(self, inputs, N2D2_object):
        Cell.__init__(self, inputs,
                      N2D2_object.getNbOutputs(),
                      deepNet=n2d2.deepnet.DeepNet(N2D2_object=N2D2_object.getAssociatedDeepNet()),
                      name=N2D2_object.getName(),
                      **self._load_N2D2_parameters(N2D2_object))

        self._N2D2_object = N2D2_object

        # TODO: Add similar methods to Activation/Solver/Quantizer for nice prints and access
        self._config_parameters['activationFunction'] = self._N2D2_object.getActivation()
        self._config_parameters['scaleSolver'] = self._N2D2_object.getScaleSolver()
        self._config_parameters['biasSolver'] = self._N2D2_object.getBiasSolver()

        self._sync_inputs_and_parents(inputs)

    def set_scale_solver(self, solver):
        self._config_parameters['scaleSolver'] = solver
        self._N2D2_object.setScaleSolver(self._config_parameters['scaleSolver'].N2D2())

    def set_bias_solver(self, solver):
        self._config_parameters['biasSolver'] = solver
        self._N2D2_object.setBiasSolver(self._config_parameters['biasSolver'].N2D2())




class Activation(Cell):
    _cell_constructors = {
        'Frame<float>': N2D2.ActivationCell_Frame_float,
        'Frame_CUDA<float>': N2D2.ActivationCell_Frame_CUDA_float,
    }
    def __init__(self, inputs, nbOutputs, N2D2_object=None, **config_parameters):

        if N2D2_object is not None and (nbOutputs is not None or len(config_parameters) > 0):
            raise RuntimeError(
                "N2D2_object argument give to cell but 'inputs' or 'nbOutputs' or 'config parameters' not None")
        if N2D2_object is None:
            self._create_from_arguments(inputs, nbOutputs, **config_parameters)
        else:
            self._create_from_N2D2_object(inputs, N2D2_object)

    def _create_from_arguments(self, inputs, nbOutputs, **config_parameters):
        Cell.__init__(self, inputs, nbOutputs, **config_parameters)
        # No optional parameter
        self._parse_optional_arguments([])
        self._N2D2_object = self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                self.get_name(),
                                                self._constructor_arguments['nbOutputs'],
                                                **self._optional_constructor_arguments)

        """Set and initialize here all complex cell members"""
        for key, value in self._config_parameters.items():
            if key is 'activationFunction':
                self._N2D2_object.setActivation(value.N2D2())
            else:
                self._set_N2D2_parameter(self._param_to_INI_convention(key), value)

        self._add_to_graph(inputs)

    def _create_from_N2D2_object(self, inputs, N2D2_object):
        Cell.__init__(self, inputs,
                      N2D2_object.getNbOutputs(),
                      deepNet=n2d2.deepnet.DeepNet(N2D2_object=N2D2_object.getAssociatedDeepNet()),
                      name=N2D2_object.getName(),
                      **self._load_N2D2_parameters(N2D2_object))

        self._N2D2_object = N2D2_object

        self._config_parameters['activationFunction'] = self._N2D2_object.getActivation()

        self._sync_inputs_and_parents(inputs)



class Reshape(Cell):
    _cell_constructors = {
        'Frame<float>': N2D2.ReshapeCell_Frame_float,
        'Frame_CUDA<float>': N2D2.ReshapeCell_Frame_CUDA_float,
    }
    def __init__(self, inputs, nbOutputs, dims, N2D2_object=None, **config_parameters):

        if N2D2_object is not None and (nbOutputs is not None or dims is not None or len(config_parameters) > 0):
            raise RuntimeError(
                "N2D2_object argument give to cell but 'inputs' or 'nbOutputs' or 'dims' or 'config parameters' not None")
        if N2D2_object is None:
            self._create_from_arguments(inputs, nbOutputs, dims, **config_parameters)
        else:
            self._create_from_N2D2_object(inputs, N2D2_object)

    def _create_from_arguments(self, inputs, nbOutputs, dims, **config_parameters):
        Cell.__init__(self, inputs, nbOutputs, **config_parameters)

        self._constructor_arguments.update({
            'dims': dims,
        })

        # No optional parameter
        self._parse_optional_arguments([])
        self._N2D2_object = self._cell_constructors[self._model_key](self._deepnet.N2D2(),
                                                self.get_name(),
                                                self._constructor_arguments['nbOutputs'],
                                                self._constructor_arguments['dims'],
                                                **self._optional_constructor_arguments)

        """Set and initialize here all complex cell members"""
        for key, value in self._config_parameters.items():
            if key is 'activationFunction':
                self._N2D2_object.setActivation(value.N2D2())
            else:
                self._set_N2D2_parameter(self._param_to_INI_convention(key), value)

        self._add_to_graph(inputs)

    def _create_from_N2D2_object(self, inputs, N2D2_object):
        Cell.__init__(self, inputs,
                      N2D2_object.getNbOutputs(),
                      deepNet=n2d2.deepnet.DeepNet(N2D2_object=N2D2_object.getAssociatedDeepNet()),
                      name=N2D2_object.getName(),
                      **self._load_N2D2_parameters(N2D2_object))

        self._N2D2_object = N2D2_object

        self._constructor_arguments['dims'] = self._N2D2_object.getDims()

        self._config_parameters['activationFunction'] = self._N2D2_object.getActivation()

        self._sync_inputs_and_parents(inputs)
