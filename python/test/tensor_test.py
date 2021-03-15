import numpy as np

import n2d2


def test_classic_operation_tensor():
    print("Test classic operations on Tensor\n-----------------------------------")
    x,y,z=(2,3,4)
    print("Test create a tensor")
    b = n2d2.tensor.Tensor([x, y, z], defaultDataType=bool)
    print(b)
    
    # print("Test getter for dims")
    # b_dim,x_dim,y_dim,z_dim=(1,2,3,4)
    # dim_tensor = n2d2.tensor.Tensor([b_dim,x_dim,y_dim,z_dim], defaultDataType=bool)

    # print('Dims : ', dim_tensor.dims())
    # print("dimB :", dim_tensor.dimB())
    # print("dimX :", dim_tensor.dimX())
    # print("dimY :", dim_tensor.dimY())
    # print("dimZ :", dim_tensor.dimZ())

    # assert dim_tensor.dimB() == b_dim
    # assert dim_tensor.dimX() == x_dim
    # assert dim_tensor.dimY() == y_dim
    # assert dim_tensor.dimZ() == z_dim

    print("Test setting and getting values")
    print('Using coordinates')
    b[1,0,1] = 1 # Using coordinates
    assert b[1,0,1] == 1
    print("Using index")
    b[0] = 1 # using index
    assert b[0] == 1
    print("Using a index slice")
    b[1:3] = 1 # setting a slice 
    assert b[1] == 1 and b[2] == 1

    print("Test fill")
    b[0:] = False
    
    print("Test for loop")
    for i in b:
        assert i == 0

    print("Test len")
    assert len(b) == x*y*z

    print("Test dims & shape")
    x,y,z=(1,2,3)
    b = n2d2.tensor.Tensor([x, y, z], defaultDataType=bool)
    assert b.dims() == [z,y,x]
    assert b.shape() == [x,y,z]
    print("Test copy method")
    copy = b.copy()
    assert copy is not b and copy == b

    print("Test reshape")
    b.reshape([z, y, x])
    assert b.shape() == [z, y, x]

    print("Test equal")
    same_tensor = n2d2.tensor.Tensor([z, y, x], defaultDataType=bool)
    type_different = n2d2.tensor.Tensor([z, y, x], defaultDataType=float)
    different_tensor = n2d2.tensor.Tensor([z, y, x], defaultDataType=bool)
    dim_different = n2d2.tensor.Tensor([x,y], defaultDataType=bool)
    different_tensor[0:] = 1
    assert b == same_tensor
    assert b == type_different
    assert b != different_tensor
    assert b != dim_different

    print("Test contain method")
    c = n2d2.tensor.Tensor([1], defaultDataType=int)
    c[0] = 5
    assert 5 in c
    
    print("Test convert to numpy")
    b = n2d2.tensor.Tensor([3, 2])
    b[0] = 1
    b[1] = 2
    b[2] = 3
    b[3] = 4
    b[4] = 5
    b[5] = 6
    
    np_b = b.to_numpy()
    equivalent_numpy = np.array([[1, 2], [3, 4], [5, 6]])
    assert np.array_equal(np_b, equivalent_numpy)
    tensor_numpy = n2d2.tensor.Tensor([3, 2])
    tensor_numpy.from_numpy(np_b)
    assert b == tensor_numpy



def test_classic_operation_cudatensor():
    print("---------------------------------------\nTest classic operations on CUDA Tensor\n---------------------------------------")
    x,y,z=(2,3,4)
    print("Test create a CUDA tensor")
    b = n2d2.tensor.CudaTensor([x, y, z], defaultDataType=int)

    print("Test setting and getting values")
    print('Using coordinates')
    b[1,0,1] = 1 # Using coordinates
    assert b[1,0,1] == 1
    print("Using index")
    b[0] = 1 # using index
    assert b[0] == 1
    print("Using a index slice")
    b[1:3] = 1 # setting a slice 
    assert b[1] == 1 and b[2] == 1

    print("Test fill")
    b[0:] = False
    
    print("Test for loop")
    for i in b:
        assert i != 0

    print("Test len")
    assert len(b) == x*y*z

    print("Test dims & shape")
    x,y,z=(1,2,3)
    b = n2d2.tensor.CudaTensor([x, y, z], defaultDataType=int)
    assert b.dims() == [z,y,x]
    assert b.shape() == [x,y,z]
    print("Test copy method")
    copy = b.copy()
    assert copy is not b and copy == b

    print("Test reshape")
    b.reshape([z, y, x])
    assert b.shape() == [z, y, x]

    print("Test equal")
    same_tensor = n2d2.tensor.CudaTensor([z, y, x], defaultDataType=int)
    type_different = n2d2.tensor.CudaTensor([z, y, x], defaultDataType=float)
    different_tensor = n2d2.tensor.CudaTensor([z, y, x], defaultDataType=int)
    dim_different = n2d2.tensor.CudaTensor([x,y], defaultDataType=int)
    different_tensor[0:] = 1

    print("Test contain method")
    c = n2d2.tensor.CudaTensor([1], defaultDataType=int)
    c[0] = 5
    assert 5 in c

    # print("Test to list")
    # c = n2d2.tensor.CudaTensor([2, 3, 4])
    # l = c.to_list()
    # equivalent_list = [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]
    # assert equivalent_list == l
    
    print("Test convert to numpy")
    b = n2d2.tensor.CudaTensor([3, 2])
    b[0] = 1
    b[1] = 2
    b[2] = 3
    b[3] = 4
    b[4] = 5
    b[5] = 6
    
    np_b = b.to_numpy()
    equivalent_numpy = np.array([[1, 2], [3, 4], [5, 6]])
    assert np.array_equal(np_b, equivalent_numpy)
    tensor_numpy = n2d2.tensor.CudaTensor([3, 2])
    tensor_numpy.from_numpy(np_b)
    assert b == tensor_numpy


def tf_numpy_n2d2():
    tf_tensor = tensorflow.constant([[[10,20], [2, 4]], [[2, 6], [3, 5]]])
    print("TensorFlow")
    print(tf_tensor)
    # a eval() method seems to be available (look for eager execution ...)
    numpy_tensor = tf_tensor.numpy()
    print("Numpy")
    print(numpy_tensor)
    n2d2_tensor = n2d2.tensor.Tensor([3, 2], defaultDataType=int)
    n2d2_tensor.from_numpy(numpy_tensor)
    print("n2d2")
    print(n2d2_tensor)
    

def torch_numpy_n2d2():
    # TRANSFORMING TORCH TENSOR TO N2D2 WITH NUMPY
    torch_tensor = torch.tensor([[10, 20, 40], [2, 4, 5]])
    print("PyTorch")
    print(torch_tensor)
    numpy_tensor = torch_tensor.cpu().detach().numpy()
    # Doesn't detach gradient (what does grad do ?)
    # numpy_arr2 = tensor_arr.numpy()
    print("numpy")
    print(numpy_tensor)
    n2d2_tensor = n2d2.tensor.Tensor([3, 3], defaultDataType=int)
    n2d2_tensor.from_numpy(numpy_tensor)
    print("n2d2")
    print(n2d2_tensor)
    print("dimensions :", n2d2_tensor.dims())
    print("Back to numpy")
    bnumpy = np.array(n2d2_tensor)
    print(bnumpy)
    print("Back to PyTorch")
    outputs = torch.from_numpy(bnumpy)
    print(outputs.shape)
    

def tf_n2d2():
    # TODO Really messy function need a good change do NOT let it like this (atm only needs to work ...) 
    # From TensorFlow to n2d2 with no numpy.
    tf_tensor = tensorflow.constant([[[10,20], [2, 4]], [[2, 6], [3, 5]]])
    print("TensorFlow")
    print(tf_tensor)
    
    n2d2_tensor = n2d2.tensor.Tensor.fromTf(tf_tensor)
    print(n2d2_tensor)

test_classic_operation_tensor()
test_classic_operation_cudatensor()
# tf_numpy_n2d2()
# torch_numpy_n2d2()
# basic_tensor_test()
# c = n2d2.tensor.Tensor(1, defaultDataType=int)
# print(c)
# tf_n2d2()