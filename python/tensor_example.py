import n2d2

b = n2d2.tensor.Tensor([2,2,2], DefaultDataType=bool)
b[1,1,1] = 1 # Using coordinates
b[0] = 1 # using index
print(b) # printing tensor
print(b[(0,0,0)]) # printing an element
b[1:2] = 1 # setting a slice 
print(b)