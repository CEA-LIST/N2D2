"""
    (C) Copyright 2021 CEA LIST. All Rights Reserved.
    Contributor(s): Cyril MOINEAU (cyril.moineau@cea.fr)

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
This file contain an example of the usage of the quantization.
We want to quantize a ResNet-18 ONNX model with 1-bits weights and 4-bits activations using the SAT quantization method.
Source to the ONNX file :  https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet18v1/resnet18v1.onnx
"""

### Import + global var ###
import n2d2
from n2d2_ip.quantizer import SATCell, SATAct
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, help='Path to the ILSVRC2012 dataset')
parser.add_argument("--label_path", type=str, help='Path to the ILSVRC2012 labels')
args = parser.parse_args()


nb_epochs = 100
batch_size = 128
n2d2.global_variables.cuda_device = 2
n2d2.global_variables.default_model = "Frame_CUDA"
### Creating a database driver ###

print("Create database")
database = n2d2.database.ILSVRC2012(learn=1.0, random_partitioning=True)
database.load(args.data_path, label_path=args.label_path)
print(database)
print("Create provider")
provider = n2d2.provider.DataProvider(database=database, size=[224, 224, 3], batch_size=batch_size)
print(provider)

### Applying Transformations ###
print("Adding transformations")
transformations = n2d2.transform.Composite([
    n2d2.transform.ColorSpace("RGB"),
    n2d2.transform.RangeAffine("Divides", 255.0),
    n2d2.transform.RandomResizeCrop(224, 224, scale_min=0.2, scale_max=1.0, ratio_min=0.75,
                                    ratio_max=1.33, apply_to="LearnOnly"),
    n2d2.transform.Rescale(256, 256, keep_aspect_ratio=True, resize_to_fit=False,
                          apply_to="NoLearn"),
    n2d2.transform.PadCrop(256, 256, apply_to="NoLearn"),
    n2d2.transform.SliceExtraction(224, 224, offset_x=16, offset_y=16, apply_to="NoLearn"),
])

print(transformations)

flip_trans = n2d2.transform.Flip(apply_to="LearnOnly", random_horizontal_flip=True)

provider.add_transformation(transformations)
provider.add_on_the_fly_transformation(flip_trans)

print(provider)

### Loading ONNX ###

model = n2d2.cells.DeepNetCell.load_from_ONNX(provider, "./resnet18v1.onnx")


print("BEFORE MODIFICATION :")
print(model)
### Updating DeepNet parameters ###

print("Updating cells ...")

for cell in model:
    ### Updating Conv Cells ###
    if isinstance(cell, n2d2.cells.Conv):
        # You need to replace weights filler before adding the quantizer.
        cell.set_weights_filler(
            n2d2.filler.Xavier(
                variance_norm="FanOut",
                scaling=1.0,
        ), refill=True)

        if cell.has_bias():
            cell.refill_bias()
        cell.quantizer = SATCell(
                apply_scaling=False, 
                apply_quantization=False
        )

        cell.set_solver_parameter("learning_rate_policy", "CosineDecay")
        cell.set_solver_parameter("learning_rate", 0.05)
        cell.set_solver_parameter("momentum", 0.9)
        cell.set_solver_parameter("decay", 0.00004)
        cell.set_solver_parameter("max_iterations", 192175050)
        cell.set_solver_parameter("iteration_size", 2)

    ### Updating Fc Cells ###
    if isinstance(cell, n2d2.cells.Fc):
        cell.set_weights_filler(
            n2d2.filler.Xavier(
                variance_norm="FanOut",
                scaling=1.0,
        ), refill=True)
        cell.set_bias_filler(
            n2d2.filler.Constant(
                value=0.0,
        ), refill=True)


        cell.quantizer = SATCell(
                apply_scaling=False, 
                apply_quantization=False
        )
        cell.set_solver_parameter("learning_rate_policy", "CosineDecay")
        cell.set_solver_parameter("learning_rate", 0.05)
        cell.set_solver_parameter("momentum", 0.9)
        cell.set_solver_parameter("decay", 0.00004)
        cell.set_solver_parameter("max_iterations", 192175050)
        cell.set_solver_parameter("iteration_size", 2)

    ### Updating BatchNorm Cells ###
    if isinstance(cell, n2d2.cells.BatchNorm2d):
        cell.set_solver_parameter("learning_rate_policy", "CosineDecay")
        cell.set_solver_parameter("learning_rate", 0.05)
        cell.set_solver_parameter("momentum", 0.9)
        cell.set_solver_parameter("decay", 0.00004)
        cell.set_solver_parameter("max_iterations", 192175050)
        cell.set_solver_parameter("iteration_size", 2)
print("AFTER MODIFICATION :")
print(model)

softmax = n2d2.cells.Softmax(with_loss=True)

loss_function =  n2d2.target.Score(provider)

print("\n### Training ###")
for epoch in range(nb_epochs):
    provider.set_partition("Learn")
    model.learn()

    print("\n# Train Epoch: " + str(epoch) + " #")

    for i in range(math.ceil(database.get_nb_stimuli('Learn') / batch_size)):
        x = provider.read_random_batch()
        x = model(x)
        x = softmax(x)
        x = loss_function(x)

        x.back_propagate()
        x.update()

        print("Example: " + str(i * batch_size) + ", loss: "
              + "{0:.3f}".format(x[0]), end='\r')

    print("\n### Validation ###")

    loss_function.clear_success()

    provider.set_partition('Validation')
    model.test()

    for i in range(math.ceil(database.get_nb_stimuli('Validation') / batch_size)):
        batch_idx = i * batch_size

        x = provider.read_batch(batch_idx)
        x = model(x)
        x = softmax(x)
        x = loss_function(x)

        print("Validate example: " + str(i * batch_size) + ", val success: "
              + "{0:.2f}".format(100 * loss_function.get_average_score(metric="Precision")) + "%", end='\r')

print("\nPloting the network ...")
x.get_deepnet().draw_graph("./resnet18v1_clamped")
x.get_deepnet().log_stats("./resnet18v1_clamped_stats")
print("Saving weights !")
model.get_embedded_deepnet().export_network_free_parameters("resnet_weights_clamped")

print("Updating cells")

for cell in model:
    ### Updating Rectifier ###
    if isinstance(cell.activation, n2d2.activation.Rectifier):
        cell.activation = n2d2.activation.Linear(
                quantizer=SATAct(
                    range=15,
                    solver=n2d2.solver.SGD(
                        learning_rate_policy = "CosineDecay",
                        learning_rate=0.05,
                        momentum=0.9,
                        decay=0.00004,
                        max_iterations=115305030
        )))

    if isinstance(cell, (n2d2.cells.Conv, n2d2.cells.Fc)):
        cell.quantizer.set_quantization(True)
        cell.quantizer.set_range(15)

# The first and last cell are in full precision !
model["resnetv15_conv0_fwd"].quantizer.set_range(255)
model["resnetv15_dense0_fwd"].quantizer.set_range(255)

print("\n### Training ###")
for epoch in range(nb_epochs):

    provider.set_partition("Learn")
    model.learn()

    print("\n# Train Epoch: " + str(epoch) + " #")

    for i in range(math.ceil(database.get_nb_stimuli('Learn') / batch_size)):
        x = provider.read_random_batch()
        x = model(x)
        x = softmax(x)
        x = loss_function(x)

        x.back_propagate()
        x.update()

        print("Example: " + str(i * batch_size) + ", loss: "
              + "{0:.3f}".format(x[0]), end='\r')

    print("\n### Validation ###")

    loss_function.clear_success()

    provider.set_partition('Validation')
    model.test()

    for i in range(math.ceil(database.get_nb_stimuli('Validation') / batch_size)):
        batch_idx = i * batch_size

        x = provider.read_batch(batch_idx)
        x = model(x)
        x = softmax(x)
        x = loss_function(x)

        print("Validate example: " + str(i * batch_size) + ", val success: "
              + "{0:.2f}".format(100 * loss_function.get_average_score(metric="Precision")) + "%", end='\r')

x.get_deepnet().draw_graph("./resnet18v1_quant")
x.get_deepnet().log_stats("./resnet18v1_quant_stats")
model.get_embedded_deepnet().export_network_free_parameters("resnet_weights_SAT")
