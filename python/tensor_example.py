import n2d2
import numpy as np

print("Test create a tensor")
b = n2d2.tensor.Tensor([2,2,2], DefaultDataType=bool)
print(b)
print("Test setting values")
b[1,1,1] = 1 # Using coordinates
b[0] = 1 # using index
b[1:2] = 1 # setting a slice 
print(b) # printing tensor

print("Test fill")
b[0:] = False
print(b)
b[0:] = True
print(b)
print("Test access a value")
print(b[(0,0,0)]) # printing an element

print("Test len ", len(b))

print("Test resize :")
b.resize([2,2])
print(b)

print("Test iterrate")
for i in b:
    print(i)

print("Test convert to numpy")
numpy_tensor = np.array(b)
print(numpy_tensor)

print("Test import np array")
tensor_numpy = n2d2.tensor.Tensor([2, 2], DefaultDataType=int)
narray =np.array([[2, 1], [4, 7]])
print(narray.dtype)
tensor_numpy.fromNumpy(narray)
print(tensor_numpy)

tensor_numpy = n2d2.tensor.Tensor([2, 2], DefaultDataType=bool)
tensor_numpy.fromNumpy(np.array([[True, False], [False, False]]))
print(tensor_numpy)
print(tensor_numpy.dataType)

print("Test contain method")

c = n2d2.tensor.Tensor([1], DefaultDataType=int)
c[0] = 5
assert 5 in c
print("Works")
