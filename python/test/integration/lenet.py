from ini_reader import IniReader as DeepNet
import n2d2

inputs = n2d2.Tensor([128, 1, 32, 32], cuda=True) # TODO : Adding a method to setrandom tensor ?

net = DeepNet("../../models/LeNet.ini")

ini_output = net.forward(inputs)
print(type(ini_output._tensor))
inputs.detach_cell()

model = n2d2.model.lenet.LeNet(10)
py_output = model(inputs)

print(type(py_output._tensor))

assert(py_output == ini_output)

