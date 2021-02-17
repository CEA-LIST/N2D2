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

        #if isinstance(inputs, list):
        #    self._inputs = inputs
        #else:
        #    self._inputs = [inputs]
        self._inputs = []

        if 'deepNet' in config_parameters:
            self._deepnet = config_parameters.pop('deepNet')
        else:
            # TODO: This could be more elegant
            if isinstance(inputs, list):
                self._deepnet = inputs[0].get_deepnet()
            elif isinstance(inputs, n2d2.deepnet.Sequence):
                self._deepnet = inputs.get_last().get_deepnet()
            elif isinstance(inputs, n2d2.deepnet.Layer):
                self._deepnet = inputs.get_elements()[0].get_deepnet()
            elif isinstance(inputs, Cell):
                self._deepnet = inputs.get_deepnet()
            else:
                raise TypeError("Object of type " + str(type(inputs)) + " cannot provide a deepnet to cell.")

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
        elif isinstance(inputs, Cell) or isinstance(inputs, n2d2.provider.DataProvider) or isinstance(inputs, n2d2.tensor.Tensor):
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
            self._N2D2_object.addInput(inputs.N2D2(), n2d2.tensor.Tensor(dims=[]).N2D2()) # TODO: Tested
        else:
            if 'mapping' in self._connection_parameters:
                self._connection_parameters['mapping'] = self._connection_parameters['mapping'].create_N2D2_mapping(
                                               inputs.N2D2().getNbOutputs(),
                                               self._N2D2_object.getNbOutputs()
                                           ).N2D2()
            self._N2D2_object.addInput(inputs.N2D2(), **self._connection_parameters)

            if isinstance(inputs, n2d2.provider.DataProvider):
                self._deepnet.add_provider(inputs)


    def _link_to_N2D2_deepnet(self):
        parents = []
        for ipt in self.get_inputs():
            if not isinstance(ipt, n2d2.provider.DataProvider):
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

    def get_name(self):
        return self._name

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
                    output += ","
                output += "'" + cell.get_name() + "'"
            output += "]"
        else:
            output += "[inputs=[]]"
        return output


    def convert_to_INI_section(self):
        """Possible to create section without name"""
        #if self._constructor_arguments['Name'] is not None:
        output = "[" + self.get_name() + "]\n"
        output += "Input="
        for idx, cell in enumerate(self._inputs):
            if idx > 0:
                output += ","
            output += cell.get_name()
        output += "\n"
        output += "Type=" + self.get_type() + "\n"
        output += "nbOutputs=" + str(self._constructor_arguments['nbOutputs']) + "\n"
        return output




class Fc(Cell):

    _cell_constructors = {
            'Frame<float>': N2D2.FcCell_Frame_float,
            'Frame_CUDA<float>': N2D2.FcCell_Frame_CUDA_float,
    }

    def __init__(self, inputs, nbOutputs, N2D2_object=None, **config_parameters):
        # TODO : Add description for filler and solver.
        """s
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


    def set_weights_solver(self, solver):
        print("Note: Replacing existing solver in cell: " + self.get_name())
        self._config_parameters['weightsSolver'] = solver
        self._N2D2_object.setWeightsSolver(self._config_parameters['weightsSolver'].N2D2())

    def set_bias_solver(self, solver):
        print("Note: Replacing existing solver in cell: " + self.get_name())
        self._config_parameters['biasSolver'] = solver
        self._N2D2_object.setBiasSolver(self._config_parameters['biasSolver'].N2D2())


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


    def set_weights_solver(self, solver):
        print("Note: Replacing existing solver in cell: " + self.get_name())
        self._config_parameters['weightsSolver'] = solver
        self._N2D2_object.setWeightsSolver(self._config_parameters['weightsSolver'].N2D2())

    def set_bias_solver(self, solver):
        print("Note: Replacing existing solver in cell: " + self.get_name())
        self._config_parameters['biasSolver'] = solver
        self._N2D2_object.setBiasSolver(self._config_parameters['biasSolver'].N2D2())




class Conv2D(Conv):
    _type = 'Conv2D'

    def __init__(self,
                 inputs,
                 nbOutputs,
                 kernelDims,
                 **config_parameters):

        if 'mapping' in config_parameters:
            raise RuntimeError('Conv2D does not support custom mappings')
        else:
            config_parameters['mapping'] = n2d2.mapping.Mapping(nbChannelsPerGroup=1)
        Conv.__init__(self, inputs, nbOutputs, kernelDims, **config_parameters)


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

        self._parse_optional_arguments(['withLoss', 'groupSize'])
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
        # No optionnal args
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
                 N2D2_object,
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
    _type = 'Pool2D'

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
    def __init__(self,
                 inputs,
                 nbOutputs,
                 poolDims,
                 **config_parameters):
        if 'mapping' in config_parameters:
            raise RuntimeError('Pool2D does not support custom mappings')
        else:
            config_parameters['mapping'] = n2d2.mapping.Mapping(nbChannelsPerGroup=1)
        Pool.__init__(self, inputs, nbOutputs, poolDims, **config_parameters)



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


class BatchNorm(Cell):
    _cell_constructors = {
        'Frame<float>': N2D2.BatchNormCell_Frame_float,
        'Frame_CUDA<float>': N2D2.BatchNormCell_Frame_CUDA_float,
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
        print("Note: Replacing existing solver in cell: " + self.get_name())
        self._config_parameters['scaleSolver'] = solver
        self._N2D2_object.setScaleSolver(self._config_parameters['scaleSolver'].N2D2())

    def set_bias_solver(self, solver):
        print("Note: Replacing existing solver in cell: " + self.get_name())
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
