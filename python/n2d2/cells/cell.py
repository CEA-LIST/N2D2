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

from abc import ABC, abstractmethod
from typing import Union, Optional, List
import N2D2

from n2d2 import global_variables
from n2d2.deepnet import DeepNet
from n2d2.solver import Solver
from n2d2.filler import Filler
from n2d2.target import Target
from n2d2 import Tensor, Interface
from n2d2.provider import Provider
from n2d2 import Tensor, Interface, generate_name, methdispatch, check_types, error_handler
from n2d2.provider import DataProvider
from n2d2.target import Target
from n2d2.n2d2_interface import Options

class Cell(ABC):
    """Abstract class of the higher level of cells and cells container.
    """
    @abstractmethod
    @check_types
    def __init__(self, name:str):
        if not name:
            name = generate_name(self)
        self._name = name
        self._deepnet = None

    def __call__(self, x: Union[Tensor, Interface]):
        """
        Do the common check on the inputs and infer the deepNet from the inputs.
        """
        if not isinstance(x, (Tensor, Interface)):
            raise TypeError(self.get_name() + " received an input of type " + str(
                type(x)) + ", input should be of type n2d2.Tensor or n2d2.Interface instead.")
        self._deepnet = x.get_deepnet()

    @abstractmethod
    def test(self):
        pass
    @abstractmethod
    def learn(self):
        pass

    @abstractmethod
    def import_free_parameters(self, dir_name:str, ignore_not_exists:bool =False):
        pass

    @abstractmethod
    def export_free_parameters(self, dir_name:str, verbose:bool =True):
        pass

    def get_name(self):
        return self._name

    def get_type(self):
        return type(self).__name__


class Trainable(ABC):
    """Abstract class which regroup the method shared by trainable cells.
    """
    @abstractmethod
    def __init__(self):
        if "_config_parameters" not in self.__dict__:
            raise error_handler.ImplementationError("Trainable object is not inherited with an N2D2_Interface")
        # _config_parameters in an attribute of N2D2_Interface so we access it via the __dict__.
        config_parameters = self.__dict__["_config_parameters"]
        if "solver" in config_parameters:
            solver = config_parameters.pop('solver')
            self.set_solver(solver)
        if "filler" in config_parameters:
            filler = config_parameters.pop('filler')
            self.set_filler(filler)

    @abstractmethod
    def set_solver(self, solver:Solver):
        pass

    @abstractmethod
    def set_filler(self, filler:Filler, refill: bool=False):
        pass

    @abstractmethod
    def has_quantizer(self):
        pass


class Block(Cell):
    """
        The Block class is the most general type of cell container, from which all other containers are derived.
        It saves its cells internally with a dictionary. The Block class has no implicit structure for propagation,
        the __call__ method therefore has to be defined explicitly.
    """

    def __init__(self, cells: List[Cell], name: Optional[str]=None):
        assert isinstance(cells, list)
        self._cells = {}
        for cell in cells:
            self._cells[cell.get_name()] = cell
        Cell.__init__(self, name)

    @methdispatch
    def __getitem__(self, item):
        return NotImplemented

    @__getitem__.register(str)
    def _(self, item):
        return self.get_cell(item)

    @__getitem__.register(int)
    def _(self, item):
        k = list(self._cells.values())[item]
        return k

    @check_types
    def to_deepnet_cell(self, provider:DataProvider, target:Target=None):
        """Convert a :py:class:`n2d2.cells.Block` to a :py:class:`n2d2.cells.DeepNetCell`

        :param provider: Data provider used by the neural network
        :type provider: :py:class:`n2d2.provider.DataProvider`
        :param target: Target object
        :type target: :py:class:`n2d2.target.Target`
        :return: The corresponding :py:class:`n2d2.cells.DeepNetCell`
        :rtype: :py:class:`n2d2.cells.DeepNetCell`
        """
        dummy_input = Tensor(provider.shape())

        provider._deepnet = DeepNet()
        provider._deepnet.set_provider(provider)
        provider._deepnet.N2D2().initialize()
        dummy_input = dummy_input._set_cell(provider)

        dummy_output = self(dummy_input)
        if not isinstance(dummy_output, Tensor):
            raise RuntimeError(f"{self.__class__.__name__}.__call__() should return an n2d2.Tensor object !")

        if target:
            dummy_output = target(dummy_output)

        N2D2_deepnet = dummy_output.get_deepnet().N2D2()

        if target:
            N2D2_target = target.N2D2()
        else:
            N2D2_target =  N2D2.TargetScore("Target", dummy_output.cell.N2D2(), provider.N2D2())

        N2D2_deepnet.addTarget(N2D2_target) # TODO : Add the target twice ?
        N2D2_deepnet.setDatabase(provider.N2D2().getDatabase())
        return DeepNetCell(N2D2_deepnet)

    def is_integral(self):
        """
        Check if the parameters of every cell have an integral precision.
        """
        for cell in self._cells.values():
            # mQuantizedNbBits is initialize to 0
            if "quantizer" in cell._parameters.keys() and cell.N2D2().getQuantizedNbBits() <= 0:
                return False
        return True

    def get_cells(self):
        """
           Returns dictionary with all cells that are not Blocks (i.e. NeuralNetworkCells). This allows
           therefore to access all cells by a dictionary without having to consider the recursive block
           structure of the model
        """
        cells = {}
        self._get_cells(cells)
        return cells

    def _get_cells(self, cells: List[Cell]):
        for elem in self._cells.values():
            if isinstance(elem, Block):
                elem._get_cells(cells)
            else:
                cells[elem.get_name()] = elem
    @check_types
    def get_cell(self, item: str):
        """
           Returns the low level view of a cell.
        """
        return self.get_cells()[item]

    def test(self):
        for cell in self._cells.values():
            cell.test()
        return self

    def learn(self):
        for cell in self._cells.values():
            cell.learn()
        return self

    @check_types
    def set_solver(self, solver:Solver):
        """Set a solver for every optimizable parameters in this Block. Optimizable parameters are weights, biases and quantizers.

        :param solver: Solver to use for every optimizable parameters, default= :py:class:`n2d2.solver.SGD`
        :type solver: :py:class:`n2d2.solver.Solver`, optional
        """
        for cell in self._cells.values():
            if isinstance(cell, Block):
                cell.set_solver(solver)
            else:
                if isinstance(cell, Trainable):
                    cell.solver = solver.copy()
                    if cell.has_quantizer() and isinstance(cell.quantizer, Trainable):
                        cell.quantizer.solver = solver.copy()
                if cell.activation and cell.activation.has_quantizer() \
                        and isinstance(cell.activation.quantizer, Trainable):
                    cell.activation.quantizer.solver = solver.copy()

    def set_back_propagate(self, value):
        """Set back_propagate boolean of trainable cells.
        :param value: If True trainable cell will enable back propagation.
        :type value: bool
        """
        for cell in self.get_cells().values():
            if isinstance(cell, Trainable):
                cell.back_propagate = value

    def import_free_parameters(self, dir_name:str, ignore_not_exists:bool =False):
        for cell in self._cells.values():
            cell.import_free_parameters(dir_name, ignore_not_exists=ignore_not_exists)

    def export_free_parameters(self, dir_name:str, verbose:bool =True):
        for cell in self._cells.values():
            cell.export_free_parameters(dir_name, verbose=verbose)

    def __str__(self):
        """
        Prints the cells of the block. Note that block stored cells in a dictionary, therefore the
        order of the output depends on the order in which the cells where added to the Block
        """
        return self._generate_str(1)

    def _generate_str(self, indent_level:int):
        output = "\'" + self.get_name() + "\' " + self.get_type() + "("

        for idx, value in enumerate(self._cells.values()):
            output += "\n" + (indent_level * "\t") + "(" + str(idx) + ")"
            if isinstance(value, Block):
                output += ": " + value._generate_str(indent_level + 1)
            else:
                output += ": " + value.__str__()
        output += "\n" + ((indent_level - 1) * "\t") + ")"
        return output

    def items(self):
        return self._cells.items()


class Iterable(Block, ABC):
    """
       This abstract class describes a Block object with order, i.e. an array/list-like object.
       It implements several methods of python lists. The ``__call__`` method is implicitly defined by the order
       of the list.
    """
    @abstractmethod
    def __init__(self, cells: List[Cell], name: Optional[str]=None):
        Block.__init__(self, cells, name)
        # This is the sequential representation of the cells, since the self._cells object is a dictionary and therefore
        # does not guarantee order
        self._seq = cells

    def __getitem__(self, item):

        if isinstance(item, int):
            return self._seq.__getitem__(item)
        return super().__getitem__(item)

    def __len__(self):
        return self._seq.__len__()

    def __iter__(self):
        return self._seq.__iter__()

    def insert(self, index:int, cell:Cell):
        if not isinstance(cell, Cell):
            raise error_handler.WrongInputType("cell", type(cell), ["n2d2.cells.Cell"])
        if index < 0:
            raise ValueError("Negative index are not supported.")
        self._seq.insert(index, cell)
        self._cells[cell.get_name()] = cell

    @check_types
    def append(self, cell:Cell):
        """Append a cell at the end of the sequence."""
        self._seq.append(cell)
        self._cells[cell.get_name()] = cell

    @check_types
    def remove(self, cell:Cell):
        self._seq.remove(cell)
        del self._cells[cell.get_name()]

    def index(self, item):
        return self._seq.index(item)

    def _generate_str(self, indent_level:int):
        output = "\'" + self.get_name() + "\' " + self.get_type() + "("

        for idx, value in enumerate(self._seq):
            output += "\n" + (indent_level * "\t") + "(" + str(idx) + ")"
            if isinstance(value, Block):
                output += ": " + value._generate_str(indent_level + 1)
            else:
                output += ": " + value.__str__()
        output += "\n" + ((indent_level - 1) * "\t") + ")"
        return output


class Sequence(Iterable):
    """
         This implementation of the Iterable class describes a sequential (vertical) ordering of cells.
    """
    def __init__(self, cells: List[Cell], name: Optional[str]=None):
        Iterable.__init__(self, cells, name)

    def __call__(self, x: Union[Tensor, Interface]):
        super().__call__(x)
        for cell in self:
            x = cell(x)
        return x



class Layer(Iterable):
    """
        This implementation of the Iterable class describes a layered (horizontal) ordering of cells.
        An optional mapping can be given to define connectivity with preceding input cell
    """
    @check_types
    def __init__(self, cells:list, mapping:list =None, name:str=None):
        Iterable.__init__(self, cells, name)
        self._mapping = None
        if mapping:
            self._mapping = mapping

    def __call__(self, x: Union[Tensor, Interface]):
        super().__call__(x)
        out = []
        if isinstance(x, Interface):
            x = x.get_tensors()
        elif isinstance(x, Tensor):
            x = [x]
        else:
            raise TypeError("x should be a Tensor or an interface !")
        for out_idx, cell in enumerate(self):
            cell_inputs = []
            for in_idx, ipt in enumerate(x):
                # Default is all-to-all
                if self._mapping is None or self._mapping[in_idx][out_idx]:
                    cell_inputs.append(ipt)
            # if the cell is identity, then the output is an interface and must be converted to tensor
            for tensor in cell(Interface(cell_inputs)).get_tensors():
                out.append(tensor)
        return Interface(out)


class DeepNetCell(Block):
    """
    n2d2 wrapper for a N2D2 deepnet object. Allows chaining a N2D2 deepnet (for example loaded from a ONNX or INI file)
    into the dynamic computation graph of the n2d2 API. During each use of the  the ``__call__`` method,
    the N2D2 deepnet is converted to a n2d2 representation and the N2D2 deepnet is concatenated to the deepnet of the
    incoming tensor object.
    The object is manipulated with the bound methods of the N2D2 DeepNet object, and its computation graph is
    also exclusively defined by the DeepNet object that is passed to it during construction.
    It therefore only inherits from Block, and not from the Iterable class and its children, which are reserved for the
    python APIs implicit way of constructing graphs.
    """

    def __init__(self, N2D2_object):
        """As a user, you should **not** use this method, if you want to create a DeepNetCell object, please use :
        :py:meth:`n2d2.cells.DeepNetCell.load_from_ONNX`, :py:meth:`n2d2.cells.DeepNetCell.load_from_INI`, :py:meth:`n2d2.cells.Sequence.to_deepnet_cell`

        :param N2D2_object: The N2D2 DeepNet object
        :type N2D2_object: :py:class:`N2D2.DeepNet`
        """

        # Deepnet object that is encapsulated
        self._embedded_deepnet = DeepNet.create_from_N2D2_object(N2D2_object)

        if not N2D2_object.getName() == "":
            name = N2D2_object.getName()
        else:
            name = None

        Block.__init__(self, list(self._embedded_deepnet.get_cells().values()), name=name)

        self._deepnet = self._embedded_deepnet
        self._inference = False


    @classmethod
    @check_types
    def load_from_ONNX(cls, provider:DataProvider, model_path:str, ini_file:str=None, ignore_cells:list=None):
        """Load a deepnet from an ONNX file given a provider object.

        :param provider: Provider object to base deepnet upon
        :type provider: :py:class:`n2d2.provider.DataProvider`
        :param model_path: Path to the ``onnx`` model.
        :type model_path: str
        :param ini_file: Path to an optional ``.ini`` file with additional onnx import instructions
        :type ini_file: str
        :param ignore_cells: List of cells name to ignore, default=None
        :type ignore_cells: list, optional
        """
        if not global_variables.onnx_compiled:
            raise RuntimeError("Cannot load a model from ONNX, you did not compiled N2D2 with protobuf. " \
                "Install it with 'apt-get install protobuf-compiler' and then recompile N2D2.")

        N2D2_deepnet = N2D2.DeepNet(global_variables.default_net)
        N2D2_deepnet.setStimuliProvider(provider.N2D2())
        N2D2_deepnet.setDatabase(provider.get_database().N2D2())
        N2D2.CellGenerator.defaultModel = global_variables.default_model
        ini_parser = N2D2.IniParser()
        if ini_file is not None:
            ini_parser.load(ini_file)
        ini_parser.currentSection("onnx", True)
        if ignore_cells is not None:
            ini_parser.setProperty("Ignore", ignore_cells)
        ini_parser.setProperty("CNTK", True) # Enable Bias fusion !
        N2D2_deepnet = N2D2.DeepNetGenerator.generateFromONNX(global_variables.default_net, model_path, ini_parser,
                                            N2D2_deepnet, [None])
        return cls(N2D2_deepnet)

    @classmethod
    def load_from_INI(cls, path:str):
        """Load a deepnet from an INI file.

        :param model_path: Path to the ``ini`` file.
        :type model_path: str
        """
        n2d2_deepnet = N2D2.DeepNetGenerator.generateFromINI(global_variables.default_net, path)
        return cls(n2d2_deepnet)

    def __call__(self, x):
        super().__call__(x)
        # NOTE: This currently only supports a provider output as input
        if not isinstance(x, Tensor):
            raise ValueError("Needs tensor with provider output as input")

        # Concatenate existing deepnet graph on deepnet of input
        self._deepnet = self.concat_to_deepnet(x.get_deepnet())

        for cell in self.get_input_cells():
            cell.N2D2().clearInputTensors()
            cell.N2D2().linkInput(x.cell.N2D2())

        self._deepnet.N2D2().propagate(self._inference)

        outputs = []
        for cell in self.get_output_cells():
            outputs.append(cell.get_outputs())
        if len(outputs) == 1:
            return outputs[0]
        return outputs

    def concat_to_deepnet(self, deepnet:DeepNet):

        cells = self._embedded_deepnet.N2D2().getCells()
        layers = self._embedded_deepnet.N2D2().getLayers()
        if not layers[0][0] == "env":
            print("Is env:" + layers[0][0])
            raise RuntimeError("First layer of N2D2 deepnet is not a StimuliProvider. You may be skipping cells")

        self._cells = {}

        for layer in layers[1:]:

            for cell in layer:
                N2D2_cell = cells[cell]
                parents = self._embedded_deepnet.N2D2().getParentCells(N2D2_cell.getName())
                if len(parents) == 1 and parents[0] is None:
                    parents = []
                deepnet.N2D2().addCell(N2D2_cell, parents)
                n2d2_cell = self._embedded_deepnet.get_cells()[N2D2_cell.getName()]
                n2d2_cell.set_deepnet(deepnet)
                self._cells[n2d2_cell.get_name()] = n2d2_cell

        return deepnet

    def update(self):
        """Update learnable parameters
        """
        self.get_deepnet().update()

    def test(self):
        """Set the network to ``test`` mode.
        """
        self._inference = True
        return self

    def learn(self):
        """Set the network to ``learn`` mode.
        """
        self._inference = False
        return self

    def import_free_parameters(self, dir_name:str, ignore_not_exists:bool=False):
        """Import deepnet parameters.
        """
        print(f"Importing DeepNetCell '{self._name}' parameters from  {dir_name}")
        self._deepnet.N2D2().importNetworkFreeParameters(dir_name, ignoreNotExists=ignore_not_exists)


    def export_free_parameters(self, dir_name:str, verbose:bool=True):
        """Export deepnet parameters.
        """
        if verbose:
            print(f"Exporting DeepNetCell '{self._name}' parameters from {dir_name}")
        self._deepnet.N2D2().exportNetworkFreeParameters(dir_name)


    def remove(self, name:str, reconnect:bool =True)->None:
        """Remove a cell from the encapsulated deepnet.
        :param name: Name of cell that shall be removed.
        :type name: str
        :param reconnect: If ``True``, reconnects the parents with the child of the removed cell, default=True
        :type reconnect: bool, optional
        """
        self._embedded_deepnet.remove(name, reconnect)
        self._cells.pop(name)

    def get_deepnet(self):
        """Get the :py:class:`n2d2.deepnet.DeepNet` used for computation.
        """
        return self._deepnet

    def get_embedded_deepnet(self):
        """Get the :py:class:`n2d2.deepnet.DeepNet` used to define this cell.
        """
        return self._embedded_deepnet

    def get_input_cells(self):
        """Returns the cells located at the entry of the network.
        """
        return self._embedded_deepnet.get_input_cells()

    def get_output_cells(self):
        """Returns the cells located at the end of the network.

        :return: Return a list of cells located at the end of the network
        :rtype: list
        """
        return self._embedded_deepnet.get_output_cells()

    def fit(self, learn_epoch:int, log_epoch:int =1000, avg_window:int =10000, bench:bool =False, 
                ban_multi_device:bool =False, valid_metric:str ="Sensitivity", stop_valid:int =0, 
                log_kernels:bool =False):
        """This method is used to train the :py:class:`n2d2.cells.DeepNetCell` object.

        :param learn_epoch: The number of epochs steps
        :type learn_epoch: int
        :param log_epoch: The number of epochs between logs, default=1000
        :type log_epoch: int, optional
        :param avg_window: The average window to compute success rate during learning, default=10000
        :type avg_window: int, optional
        :param bench: If ``True``, activate the benchmarking of the learning speed , default=False
        :type bench: bool, optional
        :param valid_metric: Validation metric to use can be ``Sensitivity``, ``Specificity``, \
        ``Precision``, ``NegativePredictiveValue``, ``MissRate``, ``FallOut``, ``FalseDiscoveryRate``, \
        ``FalseOmissionRate``, ``Accuracy``, ``F1Score``, ``Informedness``, ``Markedness``, default="Sensitivity"
        :type valid_metric: str, optional
        :param stop_valid: The maximum number of successive lower score validation, default=0
        :type stop_valid: int, optional
        :param log_kernels: If ``True``, log kernels after learning, default=False
        :type log_kernels: bool, optional
        """

        # Checking inputs
        if valid_metric not in N2D2.ConfusionTableMetric.__members__.keys():
            raise error_handler.WrongValue("metric", valid_metric, N2D2.ConfusionTableMetric.__members__.keys())
        N2D2_valid_metric = N2D2.ConfusionTableMetric.__members__[valid_metric]

        # Generating the N2D2 DeepNet
        N2D2_deepnet = self._embedded_deepnet.N2D2()
        N2D2_deepnet.initialize()

        # Calling learn function
        parameters = Options(
                        avg_window=avg_window, bench=bench, learn_epoch=learn_epoch,
                        log_epoch=log_epoch, ban_multi_device=ban_multi_device,
                        valid_metric=N2D2_valid_metric, stop_valid=stop_valid,
                        log_kernels=log_kernels)
        N2D2.learn_epoch(parameters.N2D2(), N2D2_deepnet)

    def run_test(self, log:int = 1000, report:int = 100, test_index:int = -1, test_id:int = -1,
                 qat_sat:bool = False, log_kernels:bool = False, wt_round_mode:str = "NONE",
                 b_round_mode:str = "NONE", c_round_mode:str = "NONE",
                 act_scaling_mode:str = "FLOAT_MULT", log_JSON:bool = False, log_outputs:int = 0):
        """This method is used to train the :py:class:`n2d2.cells.DeepNetCell` object.

        :param log: The number of steps between logs, default=1000
        :type log: int, optional
        :param report: Number of steps between reportings, default=100
        :type report: int, optional
        :param test_index: Test a single specific stimulus index in the Test set, default=-1
        :type test_index: int, optional
        :param test_id: Test a single specific stimulus ID (takes precedence over `test_index`), default=-1
        :type test_id: int, optional
        :param qat_sat: Fuse a QAT trained model with the SAT method, default=False
        :type qat_sat: bool, optional
        :param log_kernels: Log kernels after learning, default=False
        :type log_kernels: bool, optional
        :param wt_round_mode: Weights clipping mode on export, can be ``NONE``, ``RINTF``, default="NONE"
        :type wt_round_mode: str, optional
        :param b_round_mode: Biases clipping mode on export, can be ``NONE``, ``RINTF``, default="NONE"
        :type b_round_mode: str, optional
        :param c_round_mode: Clip clipping mode on export, can be ``NONE``,``RINTF``, default="NONE"
        :type c_round_mode: str, optional
        :param act_scaling_mode: activation scaling mode on export, can be ``NONE``, ``FLOAT_MULT``, ``FIXED_MULT16``, ``SINGLE_SHIFT`` or ``DOUBLE_SHIFT``, default="FLOAT_MULT"
        :type act_scaling_mode: str, optional
        :param log_JSON: If ``True``, log JSON annotations, default=False
        :type log_JSON: bool, optional
        :param log_outputs: log layers outputs for the n-th stimulus (0 = no log), default=0
        :type log_outputs: int, optional
        """
        if wt_round_mode not in N2D2.WeightsApprox.__members__.keys():
            raise error_handler.WrongValue("wt_round_mode", wt_round_mode, N2D2.WeightsApprox.__members__.keys())
        N2D2_wt_round_mode = N2D2.WeightsApprox.__members__[wt_round_mode]
        if b_round_mode not in N2D2.WeightsApprox.__members__.keys():
            raise error_handler.WrongValue("b_round_mode", b_round_mode, N2D2.WeightsApprox.__members__.keys())
        N2D2_b_round_mode = N2D2.WeightsApprox.__members__[b_round_mode]
        if c_round_mode not in N2D2.WeightsApprox.__members__.keys():
            raise error_handler.WrongValue("b_round_mode", c_round_mode, N2D2.WeightsApprox.__members__.keys())
        N2D2_c_round_mode = N2D2.WeightsApprox.__members__[c_round_mode]

        if act_scaling_mode not in N2D2.ScalingMode.__members__.keys():
            raise error_handler.WrongValue("act_scaling_mode", act_scaling_mode, N2D2.ScalingMode.__members__.keys())
        N2D2_act_scaling_mode = N2D2.ScalingMode.__members__[act_scaling_mode]

        parameters = Options(log=log, report=report,
                        test_index=test_index, test_id=test_id, qat_SAT=qat_sat,
                        wt_round_mode=N2D2_wt_round_mode, b_round_mode=N2D2_b_round_mode,
                        c_round_mode=N2D2_c_round_mode, act_scaling_mode=N2D2_act_scaling_mode,
                        log_JSON=log_JSON, log_outputs=log_outputs, log_kernels=log_kernels)
        N2D2.test(parameters.N2D2(), self._embedded_deepnet.N2D2(), False)
