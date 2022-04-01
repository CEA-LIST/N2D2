"""
    (C) Copyright 2021 CEA LIST. All Rights Reserved.
    Contributor(s): Johannes THIELE (johannes.thiele@cea.fr)


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

import n2d2

fc1 = n2d2.cells.Fc(28*28, 50, activation=n2d2.activation.Rectifier())
fc2 = n2d2.cells.Fc(50, 10)

n2d2.global_variables.verbosity = n2d2.global_variables.Verbosity.short
print("Short representation: only with compulsory constructor arguments")
print(fc1)
print(fc2)

print("Verbose representation: show graph and every arguments")
n2d2.global_variables.verbosity = n2d2.global_variables.Verbosity.detailed
print(fc1)
print(fc2)

"""
Similiar to Pytorch, the computational graph is defined implicitly by the control flow of the program. Therefore
there is no graph before an input has been propagated at least once through the cells. 
"""
print("Graph representation before propagation: no inputs visible")
n2d2.global_variables.verbosity = n2d2.global_variables.Verbosity.graph_only
print(fc1)
print(fc2)

x = n2d2.tensor.Tensor(dims=[1, 28, 28], value=0.5)
print(x.shape())

x = fc1(x)
x = fc2(x)

print(x)

"""
We cannot backpropagate on x, because N2D2 allows backpropagation only on specially defined leaf nodes. 
The Target class, if called, will for instance return a Tensor that is qualified as a leaf node.
On this tensor, we can call the backpropagate function. 
"""

print("Graph representation after propagation: we can now see the names of the inputs to each cell in the computation graph")
n2d2.global_variables.verbosity = n2d2.global_variables.Verbosity.graph_only
print(fc1)
print(fc2)


"""
You can plot the new graph with this method :
"""

x.draw_associated_graph("example_graph")


from n2d2.cells import Sequence, Conv, Pool2d, Dropout, Fc  
from n2d2.activation import Rectifier, Linear

"""
Creating a LeNet with two separate part.
"""

extractor = Sequence([
    Conv(1, 6, kernel_dims=[5, 5]),
    Pool2d(pool_dims=[2, 2], stride_dims=[2, 2], pooling='Max'),
    Conv(6, 16, kernel_dims=[5, 5]),
    Pool2d(pool_dims=[2, 2], stride_dims=[2, 2], pooling='Max'),
    Conv(16, 120, kernel_dims=[5, 5]),
], name="extractor")

classifier = Sequence([
    Fc(120, 84, activation=Rectifier()),
    Dropout(dropout=0.5),
    Fc(84, 10, activation=Linear(), name="last_fully"),
], name="classifier")

"""
LeNet model with two sequences !
"""
network = Sequence([extractor, classifier])

x = n2d2.Tensor([1,32,32], value=0.5)

output = network(x)

"""
Printing the network 
"""
print(network)

"""
Plotting the resulting graph
"""

output.draw_associated_graph("full_lenet_graph")

"""
Getting a cell from the encaspulated Sequence easily ! 
"""
first_fully = network["last_fully"]
print("Accessing the first fully connected layer which is encapsulated in a Sequence")
print(first_fully)

"""
Getting the output tensor of the fully !
"""
print("Getting the output of the last layer ")
print(f"Output of the second fully connected : {first_fully.get_outputs()}")

print("\nSaving only the parameters of the convnet :")
network[0].export_free_parameters("ConvNet_parameters")