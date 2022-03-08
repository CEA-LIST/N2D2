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

print("Verbose representation: only with compulsory constructor arguments")
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




