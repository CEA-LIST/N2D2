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
from abc import ABC, abstractmethod
from typing import Union

import N2D2
import n2d2.activation
import n2d2.filler
import n2d2.solver
import n2d2.cells.nn
from n2d2 import Interface, Tensor

from n2d2.cells.cell import Cell
from n2d2.error_handler import deprecated
from n2d2.n2d2_interface import N2D2_Interface
from n2d2.provider import MultipleOutputsProvider, Provider
from n2d2.typed import ModelDatatyped, Modeltyped

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

_cell_frame_parameters.update(_cell_parameters)


class NeuralNetworkCell(Cell, N2D2_Interface, ABC):
    """Abstract class for layer implementation.
    """
    mappable = False

    @abstractmethod
    def __init__(self,  **config_parameters):
        """
        :param name: Cell name, default = ``CellType_id``
        :type name: str, optional
        :param activation: Activation function, default= None
        :type activation: :py:class:`n2d2.activation.ActivationFunction`, optional
        """
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

        N2D2_Interface.__init__(self, **config_parameters)

        self._deepnet = None
        self._inference = False

        self.nb_input_cells = 0

    def __getattr__(self, key: str) -> None:
        if key == "name":
            return self.get_name()
        return N2D2_Interface.__getattr__(self, key)

    def __setattr__(self, key: str, value: n2d2.activation.ActivationFunction) -> None:

        if key == 'activation':
            if not (isinstance(value, n2d2.activation.ActivationFunction) or value is None):
                raise n2d2.error_handler.WrongInputType("activation", str(type(value)), [str(n2d2.activation.ActivationFunction), "None"])
            self._config_parameters["activation"] = value
            if self._N2D2_object:
                if value:
                    self._N2D2_object.setActivation(value.N2D2())
                else:
                    self._N2D2_object.setActivation(None)
        else:
            super().__setattr__(key, value)

    @classmethod
    def _get_N2D2_complex_parameters(cls, N2D2_object):
        parameters = {}
        parameters['activation'] = \
            n2d2.converter.from_N2D2_object(N2D2_object.getActivation())
        return parameters

    @classmethod
    def create_from_N2D2_object(cls, N2D2_object, **kwargs):
        n2d2_cell = super().create_from_N2D2_object(N2D2_object)

        if isinstance(n2d2_cell, ModelDatatyped):
            ModelDatatyped.__init__(n2d2_cell,
            model=N2D2_object.getPyModel(),
            datatype=N2D2_object.getPyDataType())
        elif isinstance(n2d2_cell, Modeltyped):
            Modeltyped.__init__(n2d2_cell,
            model=N2D2_object.getPyModel())
        n2d2_cell._input_cells = []

        n2d2_cell._name = N2D2_object.getName()
        n2d2_cell._inference = True
        if "n2d2_deepnet" in kwargs:
            n2d2_cell._deepnet = kwargs["n2d2_deepnet"]
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
        """
        :return: The output tensor of the cell.
        :rtype: :py:class:`Tensor`
        """
        return Tensor.from_N2D2(self._N2D2_object.getOutputs())._set_cell(self)

    def get_diffoutputs(self, index:int=0)->Tensor:
        """
        :param index: Index of the input of the cell to consider, default=0
        :type index: int, optional
        :return: The gradient computed by the cell.
        :rtype: :py:class:`Tensor`
        """
        return Tensor.from_N2D2(self._N2D2_object.getDiffOutputs(index))._set_cell(self)

    def get_diffinputs(self):
        """
        :return: The gradient given to the cell.
        :rtype: :py:class:`Tensor`
        """
        return Tensor.from_N2D2(self._N2D2_object.getDiffInputs())._set_cell(self)

    def get_deepnet(self):
        return self._deepnet

    def set_deepnet(self, deepnet):
        self._deepnet = deepnet

    def clear_input_tensors(self):
        self._input_cells = []
        self._N2D2_object.clearInputTensors()


    def _check_tensor(self, inputs: Tensor):
        if isinstance(inputs.cell, (NeuralNetworkCell, Provider)):
            # Check x-y dimension consistency
            if not isinstance(self, n2d2.cells.nn.Fc):

                if not inputs.dims()[0:2] == self.N2D2().getInputsDims()[0:2]:
                    raise RuntimeError("Unmatching dims " + str(inputs.dims()[0:2])
                                       + " " + str(self.N2D2().getInputsDims()[0:2]))
        else:
            raise TypeError("Invalid inputs object of type " + str(type(inputs.cell)))

        # NOTE: This cannot really happen in current implementation
        if inputs.get_deepnet() is not self.get_deepnet():
            raise RuntimeError("The deepnet of the input doesn't match with the deepnet of the cell")

        return False


    def add_input(self, inputs: Union[Interface, Tensor]):

        initialized = (self.dims() != [])

        if isinstance(inputs, Interface):
            tensor_inputs = inputs.get_tensors()
        elif isinstance(inputs, Tensor):
            tensor_inputs = [inputs]
        else:
            raise TypeError("Cannot add object of type " + str(type(inputs)))

        # Check input dimension consistency before connecting new inputs
        if initialized:
            # Check for input tensor element consistency
            if isinstance(inputs, Interface):
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
        elif self.mappable and self._N2D2_object.getMapping().empty():
                self._N2D2_object.setMapping(Tensor([self.get_nb_outputs(), inputs.dimZ()], value=True,
                                                             datatype="bool", dim_format="N2D2").N2D2())

        parents = []
        # Clear old input tensors of cell to connect new inputs
        self.clear_input_tensors()
        for ipt in tensor_inputs:
            cell = ipt.cell
            # cells created by Interfaces out of any deepnet at initialization have an empty 
            # getData() method so data are manually passed to the input tensor
            if isinstance(cell, MultipleOutputsProvider) or isinstance(inputs, Interface):
                diffOutput = Tensor(ipt.dims(), value=0, dim_format="N2D2")
                self._N2D2_object.addInputBis(ipt.N2D2(), diffOutput.N2D2())
                self._N2D2_object.initialize()
            else:
                self._N2D2_object.linkInput(cell.N2D2())
            if not initialized:
                self.nb_input_cells += 1

            if not isinstance(cell, Provider):
                parents.append(cell.N2D2())
            else:
                parents.append(None)
            self._input_cells.append(cell.get_name())

        self._deepnet.N2D2().addCell(self._N2D2_object, parents)
        if (self.dims()==[]): #not initialized
            self._N2D2_object.initializeDataDependent()

    def _add_to_graph(self, inputs: Union[Tensor, Interface]):
        self.add_input(inputs)

    @deprecated(reason="You should use activation as a python attribute.")
    def set_activation(self, activation: n2d2.activation.ActivationFunction):
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
        return None

    def get_input_cells(self):
        return self._input_cells

    def clear_input(self):
        self._input_cells = []
        self._N2D2_object.clearInputs()

    def update(self):
        self._N2D2_object.update()

    def import_free_parameters(self, dir_name:str, ignore_not_exists:bool =False):
        if self._N2D2_object:
            filename = dir_name + "/" + self.get_name() + ".syntxt"
            print("Import " + filename)
            self._N2D2_object.importFreeParameters(filename, ignore_not_exists)
            self._N2D2_object.importActivationParameters(dir_name, ignore_not_exists)

    def export_free_parameters(self, dir_name:str, verbose:bool =True) -> None:
        if self._N2D2_object:
            filename = dir_name + "/" + self.get_name() + ".syntxt"
            if verbose:
                print("Export to " + filename)
            self._N2D2_object.exportFreeParameters(filename)
            filename = dir_name + "/" + self.get_name() + "_quant.syntxt"
            if verbose:
                print("Export to " + filename)
            self._N2D2_object.exportQuantFreeParameters(filename)
            self._N2D2_object.exportActivationParameters(dir_name)


    # def import_activation_parameters(self, filename, **kwargs):
    #     print("import " + filename)
    #     self._N2D2_object.importActivationParameters(filename, **kwargs)


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
        if len(self.get_input_cells()) > 0:
            output += "(["
            for idx, name in enumerate(self.get_input_cells()):
                if idx > 0:
                    output += ", "
                output += "'" + name + "'"
            output += "])"
        else:
            output += ""
        return output

    @staticmethod
    def is_exportable_to(export_name:str) -> bool:
        """
        :param export_name: Name of the export 
        :type export_name: str
        :return: ``True`` if the cell is exportable to the ``export_name`` export. 
        :rtype: bool
        """
        return False
