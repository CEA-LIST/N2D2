"""
    (C) Copyright 2022 CEA LIST. All Rights Reserved.
    Contributor(s): Inna KUCHER (inna.kucher@cea.fr)

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

"""
This file contains an example of the usage of the quantization.
Objective: quantize an Inception-ResNetv2 ONNX model with 8-bits weights and 8-bits activations using the LSQ quantization method.
Source to the ONNX file :  N2D2-Storage (repository to store ONNX models, incoming soon)
"""

### Import + global var ###
import n2d2
from n2d2.quantizer import LSQCell, LSQAct
import math 
nb_epochs = 1
batch_size = 32
n2d2.global_variables.cuda_device = 2
n2d2.global_variables.default_model = "Frame_CUDA"
import time
### Creating a database driver ###

print("Create database")
database = n2d2.database.ILSVRC2012(learn=1.0, random_partitioning=True)
database.load("/data1/is156025/DATABASE/ILSVRC2012", label_path="/data1/is156025/DATABASE/ILSVRC2012/synsets.txt")
print(database)
print("Create provider")
provider = n2d2.provider.DataProvider(database=database, size=[299, 299, 3], batch_size=batch_size)
print(provider)

### Applying Transformations ###
print("Adding transformations")
transformations =n2d2.transform.Composite([
    n2d2.transform.ColorSpace("RGB"),
    n2d2.transform.RangeAffine("Divides", 255.0),
    n2d2.transform.Rescale(333, 333, keep_aspect_ratio=True, resize_to_fit=False), 
    n2d2.transform.PadCrop(299, 299),
    n2d2.transform.RangeAffine("Minus", [0.5, 0.5, 0.5], second_operator="Divides", second_value=[0.5,0.5,0.5])
])

print(transformations)

flip_trans = n2d2.transform.Flip(apply_to="LearnOnly", random_horizontal_flip=True)

provider.add_transformation(transformations)
provider.add_on_the_fly_transformation(flip_trans)

print(provider)

### Loading ONNX ###

model = n2d2.cells.DeepNetCell.load_from_ONNX(provider, "./inception_resnet_v2_imagenet.onnx")

print("BEFORE MODIFICATION :")
print(model)
### Updating DeepNet parameters ###

print("Updating cells with LSQ quantizer for 8 bits ...")

lr = 0.001
mom = 0.9
decay = 0.00001
max_iter = 1281167
iter_size = 8
metric = "Precision"
q_range = 255

for cell in model:
    ### Updating Conv Cells ###
    if isinstance(cell, n2d2.cells.Conv):
        cell.quantizer = LSQCell(
                range = q_range,
                solver=n2d2.solver.SGD(
                        learning_rate_policy = "CosineDecay",
                        learning_rate=lr,
                        momentum=mom,
                        decay=decay,
                        max_iterations=max_iter,
                        iteration_size = iter_size
        ))   
        cell.set_solver_parameter("learning_rate_policy", "CosineDecay")
        cell.set_solver_parameter("learning_rate", lr)
        cell.set_solver_parameter("momentum", mom)
        cell.set_solver_parameter("decay", decay)
        cell.set_solver_parameter("max_iterations", max_iter)
        cell.set_solver_parameter("iteration_size", iter_size)

    ### Updating Fc Cells ###
    if isinstance(cell, n2d2.cells.Fc):
        cell.quantizer = LSQCell(
            range = q_range,
            solver=n2d2.solver.SGD(
                        learning_rate_policy = "CosineDecay",
                        learning_rate=lr,
                        momentum=mom,
                        decay=decay,
                        max_iterations=max_iter,
                        iteration_size = iter_size
        ))

        cell.set_solver_parameter("learning_rate_policy", "CosineDecay")
        cell.set_solver_parameter("learning_rate", lr)
        cell.set_solver_parameter("momentum", mom)
        cell.set_solver_parameter("decay", decay)
        cell.set_solver_parameter("max_iterations", max_iter)
        cell.set_solver_parameter("iteration_size", iter_size)

    ### Updating BatchNorm Cells ###
    if isinstance(cell, n2d2.cells.BatchNorm2d):
        cell.set_solver_parameter("learning_rate_policy", "CosineDecay")
        cell.set_solver_parameter("learning_rate", lr)
        cell.set_solver_parameter("momentum", mom)
        cell.set_solver_parameter("decay", decay)
        cell.set_solver_parameter("max_iterations", max_iter)
        cell.set_solver_parameter("iteration_size", iter_size)

print("Updating ReLu with LSQ quantizer for 8 bits ...")

for cell in model:
    ### Updating Rectifier ###
    if isinstance(cell.activation, n2d2.activation.Rectifier):
        cell.activation = n2d2.activation.Linear(
                quantizer=LSQAct(
                    range=q_range, 
                    solver=n2d2.solver.SGD(
                        learning_rate_policy = "CosineDecay",
                        learning_rate=lr,
                        momentum=mom,
                        decay=decay,
                        max_iterations=max_iter,
                        iteration_size = iter_size
        )))

print("AFTER MODIFICATION :")
print(model)

softmax = n2d2.cells.Softmax(with_loss=True)

loss_function =  n2d2.target.Score(provider)
max_precision = -1

deepNetCell = n2d2.cells.Sequence([model, softmax]).to_deepnet_cell(provider,loss_function)

print("\n### LEARN ###")

deepNetCell.fit(nb_epochs, valid_metric=metric)

print("\n### VALIDATION ###")

deepNetCell.run_test()

print("\n### THE END ###")