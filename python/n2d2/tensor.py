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
import n2d2 # To remove if interface is moved to provider
from n2d2 import error_handler
from n2d2.provider import TensorPlaceholder
from functools import reduce
import random

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
        "Numpy": lambda x: [i for i in reversed(x)],
    }

    # TODO: Why is default not N2D2?
    def __init__(self, dims, value=None, cuda=False, datatype="float", cell=None, dim_format='Numpy'):
        """
        :param dims: Dimensions of the :py:class:`n2d2.Tensor` object. (the convention used depends of the ``dim_format`` argument, by default it's the same as ``Numpy``)
        :type dims: list
        :param value: A value to fill the :py:class:`n2d2.Tensor` object.
        :type value: Must be coherent with ``datatype``
        :param datatype: Type of the data stored in the tensor, default="float"
        :type datatype: str, optional
        :param cell: A reference to the object that created this tensor, default=None
        :type cell: :py:class:`n2d2.cells.NeuralNetworkCell`, optional
        :param dim_format: Define the format used when you declare the dimensions of the tensor. The ``N2D2`` convention is the reversed of the ``Numpy`` the numpy one (e.g. a [2, 3] numpy array is equivalent to a [3, 2] N2D2 Tensor), default="Numpy"
        :type dim_format: str, optional
        """
        self._leaf=False
        self.cell = cell
        self._datatype = datatype
        if not isinstance(cuda, bool):
            raise error_handler.WrongInputType("cuda", type(cuda), [str(bool)])
        self.is_cuda = cuda
        if cuda:
            generators = self._cuda_tensor_generators
        else:
            generators = self._tensor_generators

        if isinstance(dims, list):
            if not isinstance(dim_format, str):
                raise error_handler.WrongInputType("dim_format", type(dim_format), [str(str)])
            if dim_format in self._dim_format:
                dims = self._dim_format[dim_format](dims)
            else:
                raise error_handler.WrongValue('dim_format', dim_format, ", ".join(self._dim_format.keys()))
        else:
            raise error_handler.WrongInputType("dims", type(dims), [str(list)])

        if value and not isinstance(value, hard_coded_type[datatype]): # TODO : We may want to try an auto-cast ! 
            raise TypeError("You want to fill the tensor with " + str(type(value)) + " but " + str(datatype) + " is the datatype.")

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
            raise TypeError("Unrecognized Tensor datatype " + str(datatype))

        

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
            raise ValueError("Dimension are "+ str(dims) + " should be "+ str(self.shape()) + " instead.")

        def flatten(list_to_flatten):
            if len(list_to_flatten) == 1:
                if type(list_to_flatten[0]) == list:
                    result = flatten(list_to_flatten[0])
                else:
                    result = list_to_flatten
            elif type(list_to_flatten[0]) == list:
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
        return [d for d in reversed(self._tensor.dims())]
    
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
        coord = [i for i in reversed(coord)]
        if len(dims) != len(coord):
            raise ValueError(str(len(coord)) + "D array does not match " + str(len(dims)) + "D tensor.") 
        for c, d in zip(coord, dims):
            if not c < d:
                raise ValueError("Coordinate does not fit the dimensions of the tensor, max: "+str(d)+" got " + str(c)) 
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
        for i in self.shapes():
            coord.append(int(index%i))
            index = index/i
        return [i for i in reversed(coord)]

    def reshape(self, new_dims):
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
            raise ValueError("new size ("+new_dims_str+"= " + str(reduce((lambda x,y: x*y), new_dims))+") does not match current size ("+ old_dims_str+"= " +str(self.__len__())+")")
        self._tensor.reshape([int(d) for d in reversed(new_dims)])

    def copy(self):
        """Copy in memory the Tensor object.
        """
        copy = Tensor(self.shape(), datatype=self.data_type(), cuda=self.is_cuda, cell=self.cell)
        copy._tensor.op_assign(self._tensor)
        return copy

    def cpu(self):
        """Convert the tensor to a cpu tensor
        """
        # TODO : avoid to copy data
        if self.is_cuda:
            self.is_cuda = False
            new_tensor = self._tensor_generators[self._datatype](self.dims())
            new_tensor.op_assign(self._tensor)
            self._tensor = new_tensor
        return self

    def cuda(self):
        """Convert the tensor to a cuda tensor
        """
        # TODO : avoid to copy data
        if not self.is_cuda:
            self.is_cuda = True
            new_tensor = self._cuda_tensor_generators[self._datatype](self.dims())
            new_tensor.op_assign(self._tensor)
            self._tensor = new_tensor
        return self

    def to_numpy(self, copy=False):
        """Create a numpy array equivalent to the tensor.

        :param copy: if false, memory is shared between :py:class:`n2d2.Tensor` and ``numpy.array``, else data are copied in memory, default=True
        :type copy: Boolean, optional
        """
        try:
            from numpy import array 
        except ImportError:
            raise ImportError("Numpy is not installed")
        return array(self.N2D2(), copy=copy) 

    @classmethod
    def from_numpy(cls, np_array):
        """Convert a numpy array into a tensor.

        :param np_array: A numpy array to convert to a tensor.
        :type np_array: :py:class:`numpy.array`
        :return: Converted tensor
        :rtype: :py:class:`n2d2.Tensor`
        """
        try: 
            from numpy import ndarray 
        except ImportError:
            raise ImportError("Numpy is not installed !")
        if not isinstance(np_array, ndarray):
            raise error_handler.WrongInputType("np_array", type(np_array), ["numpy.array"])

        # np_array = np_array.reshape([d for d in reversed(np_array.shape)]) 
        n2d2_tensor = cls([])

        # Retrieving the first element of the numpy array to get dataType.
        try:
            first_element = np_array[0]
        except IndexError:
            raise ValueError('Numpy array is empty, you need to have at least one element')
        is_first_element = False
        while not is_first_element:
            try:
                first_element = first_element[0]
            except:
                is_first_element = True
        data_type = type(first_element.item())
        
        # convert datatype to string
        data_type = str(data_type).split("'")[1]

        if data_type == "bool":
            # Numpy -> N2D2 doesn't work for bool because there is no buffer protocol for it.
            n2d2_tensor._datatype = data_type
            tmp_tensor = n2d2_tensor._tensor_generators["int"](np_array)
            shape = [d for d in reversed(tmp_tensor.dims())]
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

    def __setitem__(self, index, value):
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
        if not isinstance(value, hard_coded_type[self._datatype]):
            try:
                value = hard_coded_type[self._datatype](value)
            except:
                raise RuntimeError("Autocast failed, tried to cast : " + str(type(value)) + " to " + self._datatype)

        if isinstance(index, tuple) or isinstance(index, list):
            self._tensor[self._get_index(index)] = value
        elif isinstance(index, int) or isinstance(index, float):
            # Force conversion to int if it's a float
            self._tensor[int(index)] = value
        elif isinstance(index, slice):
            self._tensor[index] = value
        else:
            raise error_handler.WrongInputType("index", type(index), [str(list), str(tuple), str(float), str(int), str(slice)])
        # if self.cuda:
        #     self.htod()

    def __getitem__(self, index):
        """
        Get an element of the tensor.
        To select the element to get you can use :
            - the coordinate of the element;
            - the index of the flatten tensor.
        """
        # if self.cuda:
        #     self.dtoh()
        value = None
        if isinstance(index, tuple) or isinstance(index, list):
            value = self._tensor[self._get_index(index)]
        elif isinstance(index, int) or isinstance(index, float):
            value = self._tensor[int(index)]
        else:
            raise error_handler.WrongInputType("index", type(index), [str(list), str(tuple), str(float), str(int)])
        return value
        
    def __len__(self):
        return len(self._tensor)

    def __iter__(self):
        return self._tensor.__iter__()

    def __contains__(self, value):
        return self._tensor.__contains__(value)

    def __eq__(self, other_tensor):
        if not isinstance(other_tensor, Tensor):
            raise TypeError("You can only compare tensor with each other.")
        # Quick initialization of is_equal by checking the tensors have the same dimensions
        is_equal = (self.dims() == other_tensor.dims())
        cpt = 0
        while is_equal and cpt < len(other_tensor):
            is_equal = (self._tensor[cpt] == other_tensor[cpt])
            cpt += 1
        return is_equal

    def __str__(self):
        output = "n2d2.Tensor([\n"
        output += str(self._tensor)
        output += "], device=" + ("cuda" if self.is_cuda else "cpu")
        output += ", datatype=" + self.data_type()
        if self.cell:
            output += ", cell='" + str(self.cell.get_name()) + "')"
        else:
            output += ")"
        return output

    def dtoh(self):
        """
        Synchronize Device to Host.
        CUDA tensor are stored and computed in the GPU (Device).
        You cannot read directly the GPU. A copy of the tensor exist in the CPU (Host)
        """
        if not n2d2.cuda_compiled:
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
        if not n2d2.cuda_compiled:
            raise RuntimeError("CUDA is not enabled, you need to compile N2D2 with CUDA.")
        if self.is_cuda:
            self._tensor.synchronizeHToD()
        else:
            raise RuntimeError("Trying to synchronize a non-cuda Tensor to host")
        return self

    def detach_cell(self):
        """
        Detach the cells from the tensor, thereby removing all information about the computation graph/deepnet object.
        Therefore no gradients pass this tensor after this operation has been performed.
        """
        self.cell = None
        return self

    def _set_cell(self, cell):
        self.cell = cell
        return self

    def get_deepnet(self):
        """
        Method called by the cells, if the tensor is not part of a graph, it will be linked to an :py:class:`n22d.provider.Provider` object.
        """
        if self.cell is None:
            # TensorPlaceholder will set the cell attribute to it self.
            TensorPlaceholder(self) 
        return self.cell.get_deepnet()

    def back_propagate(self):
        """
        Compute the backpropagation on the deepnet.
        """
        if not self.is_leaf():
            raise RuntimeError('This tensor is not the leaf of a graph')
        if self.cell is None:
            raise RuntimeError('This tensor is not part of a graph')
        self.cell.get_deepnet().back_propagate()
    
    def update(self):
        """
        Update weights and biases of the cells.
        """
        if not self.is_leaf():
            raise RuntimeError('This tensor is not the leaf of a graph')
        if self.cell is None:
            raise RuntimeError('This tensor is not part of a graph')
        self.cell.get_deepnet().update()

    def is_leaf(self):
        return self._leaf

    def mean(self):
        return self.N2D2().mean()

class Interface(n2d2.provider.Provider):
    """
    An :py:class:`n2d2.Interface` is used to feed multiple tensors to a cell.
    """
    def __init__(self, tensors):
        self._name = n2d2.generate_name(self)
        self.tensors = []
        if not isinstance(tensors, list):
            raise ValueError("'tensors' should be a list !")
        if not tensors:
            raise n2d2.error_handler.IsEmptyError('Tensors')

        if not tensors[0].cell: # Check if the first tensor is linked to a deepnet
            self._deepnet = None
        else:
            self._deepnet = tensors[0].cell.get_deepnet()
        
        nb_channels = 0
        for tensor in tensors:
            if not isinstance(tensor, Tensor):
                raise ValueError("The elements of 'tensors' should all be of type " + str(type(n2d2.Tensor)))
            else:
                if tensor.dimX() != tensors[0].dimX():
                    raise ValueError("Tensors should have the same X dimension.")
                if tensor.dimY() != tensors[0].dimY():
                    raise ValueError("Tensors should have the same Y dimension.")
                if tensor.dimB() != tensors[0].dimB():
                    raise ValueError("Tensors should have the same batch size.")
                current_deepnet = None if not tensor.cell else tensor.cell.get_deepnet()
                if current_deepnet is not self._deepnet:
                    raise ValueError("The tensors used to create the Interface are not linked to the same DeepNet (maybe you want to detach the cell of the tensors ?).")
                nb_channels += tensor.dimZ()
                self.tensors.append(tensor)
        if not self._deepnet:
            size =[tensors[0].dimX(), tensors[0].dimY(), nb_channels]
            self.batch_size = tensors[0].dimB()
            cell = n2d2.provider.MultipleOutputsProvider(size, self.batch_size)
            for tensor in self.tensors:
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
    def get_tensors(self):
        return self.tensors