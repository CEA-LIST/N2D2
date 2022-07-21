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
from n2d2 import methdispatch, error_handler, generate_name, check_types
from n2d2.provider import MultipleOutputsProvider
from typing import Union, Any
import n2d2.global_variables as gb
from functools import reduce
from typing import Union, Any
try:
    from numpy import ndarray, array
except ImportError:
    numpy_imported=False
else:
    numpy_imported=True

cuda_available = gb.cuda_available


hard_coded_type = {
    "f": float,
    "float": float,
    "i": int,
    "int": int,
    "b": bool,
    "bool": bool,
    "d": float,
    "double": float,
}

# pylint : disable=too-many-public-methods
class Tensor:

    _tensor_generators = {
        "f": N2D2.Tensor_float,
        "float": N2D2.Tensor_float,
        "short": N2D2.Tensor_short,
        "s": N2D2.Tensor_short,
        "long": N2D2.Tensor_long,
        "l": N2D2.Tensor_long,
        "i": N2D2.Tensor_int,
        "int": N2D2.Tensor_int,
        "b": N2D2.Tensor_bool,
        "bool": N2D2.Tensor_bool,
        "d": N2D2.Tensor_double,
        "double": N2D2.Tensor_double,
        "uchar": N2D2.Tensor_unsigned_char,
        "char": N2D2.Tensor_char,

    }
    if cuda_available:
        _cuda_tensor_generators = {
            "f": N2D2.CudaTensor_float,
            "float": N2D2.CudaTensor_float,
            "short": N2D2.CudaTensor_short,
            "s": N2D2.CudaTensor_short,
            "long": N2D2.CudaTensor_long,
            "l": N2D2.CudaTensor_long,
            "i": N2D2.CudaTensor_int,
            "int": N2D2.CudaTensor_int,
            "d": N2D2.CudaTensor_double,
            "double": N2D2.CudaTensor_double,
            # bool datatype cannot be defined for CudaTensor
        }

    _dim_format = {
        "N2D2": lambda x: x,
        "Numpy": lambda x: list(reversed(x)),
    }
    @check_types
    def __init__(self, dims:Union[list, tuple], value:Any=None, cuda:bool=False, datatype:str="float",
                 cell:Any=None, dim_format:str='Numpy'):
        """
        :param dims: Dimensions of the :py:class:`n2d2.Tensor` object. (the convention used depends of the ``dim_format`` argument, by default it's the same as ``Numpy``)
        :type dims: list
        :param value: A value to fill the :py:class:`n2d2.Tensor` object.
        :type value: Must be coherent with ``datatype``
        :param datatype: Type of the data stored in the tensor, default="float"
        :type datatype: str, optional
        :param cell: A reference to the object that created this tensor, default=None
        :type cell: :py:class:`n2d2.cells.NeuralNetworkCell`, optional
        :param dim_format: Define the format used when you declare the dimensions of the tensor. \
        The ``N2D2`` convention is the reversed of the ``Numpy`` the numpy one (e.g. a [2, 3] numpy array is equivalent to a [3, 2] N2D2 Tensor), default="Numpy"
        :type dim_format: str, optional
        """
        self._leaf = False
        self.cell = cell
        self._datatype = datatype
        if not isinstance(cuda, bool):
            raise error_handler.WrongInputType("cuda", type(cuda), [str(bool)])
        self.is_cuda = cuda
        if cuda:
            if not cuda_available:
                raise RuntimeError("You did not compiled N2D2 with CUDA !")
            generators = self._cuda_tensor_generators
        else:
            generators = self._tensor_generators

        if dim_format in self._dim_format:
            dims = self._dim_format[dim_format](dims)
        else:
            raise error_handler.WrongValue('dim_format', dim_format, self._dim_format.keys())

        if value and not isinstance(value, hard_coded_type[datatype]):
            raise TypeError(f"You want to fill the tensor with '{str(type(value).__name__)}' but datatype is set to : '{str(datatype)}'.")


        if datatype in generators:
            if not value:
                self._tensor = generators[datatype](dims)
            else:
                self._tensor = generators[datatype](dims, value)
                if cuda:
                    # The "value" argument is ignored for CUDA tensor, so we need to fill the value manually.
                    # example : N2D2.CudaTensor_int([2, 2], value=int(5))
                    self[0:] = value
                    self.htod() # Need to synchronize the host to the device
        else:
            raise TypeError(f"Unrecognized Tensor datatype {str(datatype)}")



    def N2D2(self):
        """
        :return: The N2D2 tensor object
        :rtype: :py:class:`N2D2.BaseTensor`
        """
        return self._tensor

    def set_values(self, values):
        """Fill the tensor with a list of values.

        .. testcode::

            tensor = n2d2.Tensor([1, 1, 2, 2])
            input_tensor.set_values([[[[1,2],
                                       [3, 4]]]])

        :param values: A nested list that represent the tensor.
        :type values: list
        """
        if not isinstance(values, list):
            raise error_handler.WrongInputType("values", type(values), [str(list)])

        tmp = values
        nb_dims = 0
        dims = []
        while isinstance(tmp, list):
            dims.append(len(tmp))
            tmp = tmp[0]
            nb_dims += 1
        del tmp
        if nb_dims != self.nb_dims():
            raise ValueError("The number of dims should be " + str(self.nb_dims()) + " but is "+ str(nb_dims) + " instead.")
        if dims != self.shape():
            raise ValueError(f"Dimension are {str(dims)} should be {str(self.shape())} instead.")

        def flatten(list_to_flatten):
            if len(list_to_flatten) == 1:
                if isinstance(list_to_flatten[0], list):
                    result = flatten(list_to_flatten[0])
                else:
                    result = list_to_flatten
            elif isinstance(list_to_flatten[0], list):
                result = flatten(list_to_flatten[0]) + flatten(list_to_flatten[1:])
            else:
                result = [list_to_flatten[0]] + flatten(list_to_flatten[1:])
            return result

        flatten_list = flatten(values)
        for index, value in enumerate(flatten_list):
            self[index] = value


    def nb_dims(self):
        """Return the number of dimensions.
        """
        return len(self._tensor.dims())

    def dims(self):
        """Return dimensions with N2D2 convention
        """
        return self._tensor.dims()

    def dimX(self):
        return self._tensor.dimX()

    def dimY(self):
        return self._tensor.dimY()

    def dimZ(self):
        return self._tensor.dimZ()

    def dimB(self):
        return self._tensor.dimB()

    def shape(self):
        """Return dimensions with python convention
        """
        return list(reversed(self._tensor.dims()))

    def data_type(self):
        """Return the data type of the object stored by the tensor.
        """
        return self._datatype

    def _get_index(self, coord):
        """From the coordinate returns the 1D index of an element in the tensor.

        :param coord: Tuple of the coordinate
        :type coord: tuple
        """
        dims = self.dims()
        coord = list(reversed(coord))
        if len(dims) != len(coord):
            raise ValueError(f"{str(len(coord))}D array does not match {str(len(dims))}D tensor.")
        for c, d in zip(coord, dims):
            if not c < d:
                raise ValueError(f"Coordinate does not fit the dimensions of the tensor, max: {str(d)} got {str(c)}")
        idx = 0
        for i in range(len(dims)):
            if i == 0:
                idx += coord[i]
            else:
                idx += (coord[i] * reduce((lambda x,y: x*y), dims[:i]))
        return idx

    def _get_coord(self, index):
        """From the the 1D index, return the coordinate of an element in the tensor.

        :param index: index of an element
        :type index: int
        """
        coord = []
        for i in self.shape():
            coord.append(int(index%i))
            index = index/i
        return list(reversed(coord))

    def reshape(self, new_dims:list):
        """Reshape the Tensor to the specified dims (defined by the Numpy convention).

        :param new_dims: New dimensions
        :type new_dims: list
        """
        if reduce((lambda x,y: x*y), new_dims) != len(self):
            new_dims_str = ""
            for dim in new_dims:
                new_dims_str += str(dim) +" "
            old_dims_str = ""
            for dim in self.shape():
                old_dims_str += str(dim) +" "
            raise ValueError(f"new size ({new_dims_str}= {str(reduce((lambda x,y: x*y), new_dims))}) does not match current size ({old_dims_str}= {str(self.__len__())})")
        self._tensor.reshape([int(d) for d in reversed(new_dims)])

    def resize(self, new_dims:list):
        """Reshape the Tensor to the specified dims (defined by the Numpy convention).

        :param new_dims: New dimensions
        :type new_dims: list
        """
        self._tensor.resize([int(d) for d in reversed(new_dims)])

    def copy(self):
        """Copy in memory the Tensor object.
        """
        copy = Tensor(self.shape(), datatype=self.data_type(), cuda=self.is_cuda, cell=self.cell)
        copy._tensor.op_assign(self._tensor)
        return copy

    def cpu(self):
        """Convert the tensor to a cpu tensor
        """
        if self.is_cuda:
            self.is_cuda = False
            new_tensor = self._tensor_generators[self._datatype](self.dims())
            new_tensor.op_assign(self._tensor)
            self._tensor = new_tensor
        return self

    def cuda(self):
        """Convert the tensor to a cuda tensor
        """
        if not cuda_available:
            raise RuntimeError("You did not compiled N2D2 with CUDA !")
        if not self.is_cuda:
            self.is_cuda = True
            new_tensor = self._cuda_tensor_generators[self._datatype](self.dims())
            new_tensor.op_assign(self._tensor)
            self._tensor = new_tensor
        return self

    def to_numpy(self, copy:bool =False):
        """Create a numpy array equivalent to the tensor.

        :param copy: if false, memory is shared between :py:class:`n2d2.Tensor` and ``numpy.array``, else data are copied in memory, default=True
        :type copy: Boolean, optional
        """
        if not numpy_imported:
            raise ImportError("Numpy is not installed !")
        return array(self.N2D2(), copy=copy)

    @classmethod
    def from_numpy(cls, np_array):
        """Convert a numpy array into a tensor.

        :param np_array: A numpy array to convert to a tensor.
        :type np_array: :py:class:`numpy.array`
        :return: Converted tensor
        :rtype: :py:class:`n2d2.Tensor`
        """
        if not numpy_imported:
            raise ImportError("Numpy is not installed !")
        if not isinstance(np_array, ndarray):
            raise error_handler.WrongInputType("np_array", type(np_array), ["numpy.array"])

        # np_array = np_array.reshape([d for d in reversed(np_array.shape)])
        n2d2_tensor = cls([])

        # Retrieving the first element of the numpy array to get dataType.
        try:
            first_element = np_array[0]
        except IndexError as err:
            raise ValueError('Numpy array is empty, you need to have at least one element') from err
        is_first_element = False
        while not is_first_element:
            try:
                first_element = first_element[0]
            except IndexError:
                is_first_element = True
        data_type = type(first_element.item())

        # convert datatype to string
        data_type = str(data_type).split("'")[1]

        if data_type == "bool":
            # Numpy -> N2D2 doesn't work for bool because there is no buffer protocol for it.
            n2d2_tensor._datatype = data_type
            tmp_tensor = n2d2_tensor._tensor_generators["int"](np_array)
            shape = list(reversed(tmp_tensor.dims()))
            n2d2_tensor._tensor = n2d2_tensor._tensor_generators[data_type](shape)
            for i, value in enumerate(tmp_tensor):
                n2d2_tensor._tensor[i] = value
            del tmp_tensor
        else:
            n2d2_tensor._datatype = data_type
            n2d2_tensor._tensor = n2d2_tensor._tensor_generators[data_type](np_array)
        n2d2_tensor.reshape(np_array.shape)
        return n2d2_tensor

    @classmethod
    def from_N2D2(cls, N2D2_Tensor):
        """Convert an N2D2 tensor into a Tensor.

        :param N2D2_Tensor: An N2D2 Tensor to convert to a n2d2 Tensor.
        :type N2D2_Tensor: :py:class:`N2D2.BaseTensor` or :py:class:`N2D2.CudaBaseTensor`
        :return: Converted tensor
        :rtype: :py:class:`n2d2.Tensor`
        """
        if not isinstance(N2D2_Tensor, N2D2.BaseTensor):
            raise error_handler.WrongInputType("N2D2_Tensor", str(type(N2D2_Tensor)), [str(N2D2.BaseTensor)])
        n2d2_tensor = cls([])
        n2d2_tensor._tensor = N2D2_Tensor
        n2d2_tensor._datatype = N2D2_Tensor.getTypeName()
        n2d2_tensor.is_cuda = "CudaTensor" in str(type(N2D2_Tensor))
        return n2d2_tensor

    def __array__(self):
        """Magic method called by Numpy to create an array

        Example :
        ```
        t = n2d2.Tensor([2,2])
        a = numpy.array(t)
        ```
        """
        return self.to_numpy()


    def _check_value_coherency(self, value):
        if not isinstance(value, hard_coded_type[self._datatype]):
            try:
                value = hard_coded_type[self._datatype](value)
            except ValueError as err:
                raise RuntimeError(f"Autocast failed, tried to cast : {str(type(value))} to {self._datatype}") from err

    @methdispatch
    @check_types
    def __setitem__(self, index:Union[tuple, int, float, slice], value:Any):
        """
        Set an element of the tensor.
        To select the element to modify you can use :
            - the coordinate of the element;
            - the index of the flatten tensor;
            - a slice index of the flatten tensor.
        If the ``value`` type doesn't match datatype, n2d2 tries an autocast.

        :param index: Indicate the index of the item you want to set
        :type index: tuple, int, float, slice
        :param value: The value the item will take
        :type value: same type as self._datatype
        """
        return NotImplemented

    @__setitem__.register(tuple)
    @__setitem__.register(list)
    def _(self, index, value):
        self._check_value_coherency(value)
        self._tensor[self._get_index(index)] = value

    @__setitem__.register(int)
    @__setitem__.register(float)
    def _(self, index, value):
        self._check_value_coherency(value)
        self._tensor[int(index)] = value
    @__setitem__.register(slice)
    def _(self, index, value):
        self._check_value_coherency(value)
        self._tensor[index] = value

    @methdispatch
    def __getitem__(self, index)->any:
        """
        Get an element of the tensor.
        To select the element to get you can use :
            - the coordinate of the element;
            - the index of the flatten tensor.
        """
        return NotImplemented

    @__getitem__.register(tuple)
    @__getitem__.register(list)
    def _(self, index):
        return self._tensor[self._get_index(index)]

    @__getitem__.register(int)
    @__getitem__.register(float)
    def _(self, index):
        return self._tensor[int(index)]

    def __len__(self)->int:
        return len(self._tensor)

    def __iter__(self):
        return self._tensor.__iter__()

    def __contains__(self, value)->bool:
        return self._tensor.__contains__(value)

    def __eq__(self, other_tensor:bool)->bool:
        if not isinstance(other_tensor, Tensor):
            raise TypeError("You can only compare tensor with each other.")
        # Quick initialization of is_equal by checking the tensors have the same dimensions
        is_equal = (self.dims() == other_tensor.dims())
        cpt = 0
        while is_equal and cpt < len(other_tensor):
            is_equal = (self._tensor[cpt] == other_tensor[cpt])
            cpt += 1
        return is_equal

    def __str__(self)->str:
        if self.is_cuda:
            # Updating the host before printing the Tensor
            self.N2D2().synchronizeDBasedToH()
        output = "n2d2.Tensor([\n"
        output += str(self._tensor)
        output += "], device=" + ("cuda" if self.is_cuda else "cpu")
        output += ", datatype=" + self.data_type()
        if self.cell:
            output += ", cell='" + str(self.cell.get_name()) + "')"
        else:
            output += ")"
        return output

    def __repr__(self)->str:
        return str(self._tensor)

    def dtoh(self):
        """
        Synchronize Device to Host.
        CUDA tensor are stored and computed in the GPU (Device).
        You cannot read directly the GPU. A copy of the tensor exist in the CPU (Host)
        """
        if not gb.cuda_available:
            raise RuntimeError("CUDA is not enabled, you need to compile N2D2 with CUDA.")
        if self.is_cuda:
            self._tensor.synchronizeDToH()
        else:
            raise RuntimeError("Trying to synchronize a non-cuda Tensor to device")
        return self

    def htod(self):
        """
        Synchronize Host to Device.
        CUDA tensor are stored and computed in the GPU (Device).
        You cannot read directly the GPU. A copy of the tensor exist in the CPU (Host)
        """
        if not gb.cuda_available:
            raise RuntimeError("CUDA is not enabled, you need to compile N2D2 with CUDA.")
        if self.is_cuda:
            self._tensor.synchronizeHToD()
        else:
            raise RuntimeError("Trying to synchronize a non-cuda Tensor to host")
        return self

    def detach(self):
        """
        Detach the cells from the tensor, thereby removing all information about the computation graph/deepnet object.
        """
        self.cell = None
        return self

    def _set_cell(self, cell):
        self.cell = cell
        return self

    def get_tensors(self):
        return [self]

    def get_deepnet(self):
        """
        Method called by the cells, if the tensor is not part of a graph, it will be linked to an :py:class:`n2d2.provider.Provider` object.

        :return: The associated deepnet
        :rtype: :py:class:`n2d2.deepnet.DeepNet`
        """
        if self.cell is None:
            # TensorPlaceholder will set the cell attribute to it self.
            from n2d2.provider import TensorPlaceholder
            TensorPlaceholder(self)
        return self.cell.get_deepnet()

    def draw_associated_graph(self, path: str)->None:
        """Plot the graph in a figure located at `path`.

        :param path: Path were to save the plotted graph.
        :type path: str
        """
        self.get_deepnet().draw_graph(path)

    def back_propagate(self)->None:
        """
        Compute the backpropagation on the deepnet.
        """
        if not self.is_leaf():
            raise RuntimeError('This tensor is not the leaf of a graph')
        if self.cell is None:
            raise RuntimeError('This tensor is not part of a graph')
        self.cell.get_deepnet().back_propagate()

    def update(self)->None:
        """
        Update weights and biases of the cells.
        """
        if not self.is_leaf():
            raise RuntimeError('This tensor is not the leaf of a graph')
        if self.cell is None:
            raise RuntimeError('This tensor is not part of a graph')
        self.cell.get_deepnet().update()

    def is_leaf(self)->bool:
        return self._leaf

    def mean(self)->float:
        return self.N2D2().mean()

class Interface:
    """
    An :py:class:`n2d2.Interface` is used to feed multiple tensors to a cell.
    """
    def __init__(self, tensors):
        self._name = generate_name(self)
        self.tensors = []
        if not isinstance(tensors, list):
            raise ValueError("'tensors' parameter should be a list !")
        if not tensors:
            raise error_handler.IsEmptyError('Tensors')

        #if not tensors[0].cell: # Check if the first tensor is linked to a deepnet
        #    self._deepnet = None
        #else:
        #    self._deepnet = tensors[0].cell.get_deepnet()

        self._deepnet = None
        for tensor in tensors: # Check for the first tensor that is linked to a deepnet
            if tensor.cell:
                self._deepnet = tensor.cell.get_deepnet()
                break

        nb_channels = 0
        for tensor in tensors:
            if not isinstance(tensor, Tensor):
                raise ValueError(f"The elements of 'tensors' should all be of type {str(type(Tensor))}")
            if tensor.dimX() != tensors[0].dimX():
                raise ValueError("Tensors should have the same X dimension.")
            if tensor.dimY() != tensors[0].dimY():
                raise ValueError("Tensors should have the same Y dimension.")
            if tensor.dimB() != tensors[0].dimB():
                raise ValueError("Tensors should have the same batch size.")
            current_deepnet = None if not tensor.cell else tensor.cell.get_deepnet()
            if current_deepnet is None:
                current_deepnet = self._deepnet
            if current_deepnet is not self._deepnet:
                raise ValueError("The tensors used to create the Interface are not linked to the same DeepNet (maybe you want to detach the cell of the tensors ?).")
            nb_channels += tensor.dimZ()
            self.tensors.append(tensor)
        if not self._deepnet:
            self.batch_size = tensors[0].dimB()

            for tensor in self.tensors:
                size =[tensor.dimX(), tensor.dimY(), tensor.dimZ()]
                cell = MultipleOutputsProvider(size, self.batch_size)
                tensor.cell = cell
        # The dimZ of the interface correspond to the sum of the dimZ of the tensor that composed it.
        self.dim_z = nb_channels

    def get_deepnet(self):
        return self.tensors[0].get_deepnet()
    def dimB(self):
        return self.tensors[0].dimB()
    def dimY(self):
        return self.tensors[0].dimY()
    def dimX(self):
        return self.tensors[0].dimX()
    def dimZ(self):
        return self.dim_z
    def dims(self):
        #return [self.dimB(), self.dimZ(), self.dimX(), self.dimY()]
        return [self.dimX(), self.dimY(), self.dimZ(), self.dimB()]
    def __len__(self):
        return self.tensors.__len__()
    def __getitem__(self, item):
        return self.tensors.__getitem__(item)
    def get_tensors(self):
        return self.tensors
