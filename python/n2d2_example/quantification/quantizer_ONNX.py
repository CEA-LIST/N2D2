"""
    (C) Copyright 2021 CEA LIST. All Rights Reserved.
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

"""
This file contain an example of the usage of the quantization.
We want to quantize a ResNet-18 ONNX model with 1-bits weights and 4-bits activations using the SAT quantization method.
Source to the ONNX file :  https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet18v1/resnet18v1.onnx
"""

### Import + global var ###
import n2d2
import math 
nb_epochs = 1
batch_size = 64
n2d2.global_variables.set_cuda_device(2)
n2d2.global_variables.default_model = "Frame_CUDA"

### Creating a database driver ###

print("Create database")
database = n2d2.database.ILSVRC2012(learn=1.0, random_partitioning=True)
database.load("/nvme0/DATABASE/ILSVRC2012", label_path="/nvme0/DATABASE/ILSVRC2012/synsets.txt")
print(database)
print("Create provider")
provider = n2d2.provider.DataProvider(database=database, size=[224, 224, 3], batch_size=batch_size)
print(provider)

### Applying Transformations ###
print("Adding transformations")
r_p_width = 256
r_p_height = 256
transformations =n2d2.transform.Composite([
    n2d2.transform.ColorSpace("RGB"),
    n2d2.transform.RangeAffine("Divides", 255.0),
    n2d2.transform.RandomResizeCrop(224, 224, scale_min=0.2, scale_max=1.0, ratio_min=0.75, 
                                    ratio_max=1.33, apply_to="LearnOnly"), 
    n2d2.transform.Rescale(256, 256, keep_aspect_ratio=True, resize_to_fit=False, 
                          apply_to="NoLearn"), 
    n2d2.transform.PadCrop(256, 256, apply_to="NoLearn"),
    n2d2.transform.SliceExtraction(224, 224, apply_to="NoLearn"),
])

print(transformations)

flip_trans = n2d2.transform.Flip(apply_to="LearnOnly", random_horizontal_flip=True)

provider.add_transformation(transformations)
provider.add_on_the_fly_transformation(flip_trans)

print(provider)

### Loading ONNX ###

model = n2d2.cells.DeepNetCell.load_from_ONNX(provider, "./resnet18v1.onnx")

sgd_config = n2d2.ConfigSection(
    learning_rate_policy = "CosineDecay",
    learning_rate=0.05,
    momentum=0.9,
    decay=0.00004,
    max_iterations=115305030
)

### Updating DeepNet parameters ###
for cell_name in model._cells:
    ### Updating Conv Cells ###
    if isinstance(model._cells[cell_name], n2d2.cells.Conv):
        # You need to replace weights filler before the quantizer.
        model._cells[cell_name].set_weights_filler(
            n2d2.filler.Xavier(
                variance_norm="FanOut",
                scaling=1.0,
        ))

        model._cells[cell_name].refill_weights()
        model._cells[cell_name].refill_bias()

        model._cells[cell_name].set_quantizer(
            n2d2.quantizer.SATCell(
                range=15, 
                apply_scaling=False, 
                apply_quantization=True
        ))
        conv_solver = model._cells[cell_name].get_weights_solver()
        conv_solver.set_parameter("learning_rate_policy", "CosineDecay")
        conv_solver.set_parameter("learning_rate", 0.05)
        conv_solver.set_parameter("momentum", 0.9)
        conv_solver.set_parameter("decay", 0.00004)
        conv_solver.set_parameter("max_iterations", 115305030)

        conv_solver = model._cells[cell_name].get_bias_solver()
        conv_solver.set_parameter("learning_rate_policy", "CosineDecay")
        conv_solver.set_parameter("learning_rate", 0.05)
        conv_solver.set_parameter("momentum", 0.9)
        conv_solver.set_parameter("decay", 0.00004)
        conv_solver.set_parameter("max_iterations", 115305030)

    ### Updating Fc Cells ###
    if isinstance(model._cells[cell_name], n2d2.cells.Fc):
        model._cells[cell_name].set_weights_filler(
            n2d2.filler.Normal(
                mean=0.0,
                std_dev=0.01,
        ))
        model._cells[cell_name].set_bias_filler(
            n2d2.filler.Constant(
                value=0.0,
        ))
        model._cells[cell_name].refill_weights()
        model._cells[cell_name].refill_bias()
        model._cells[cell_name].set_quantizer(
            n2d2.quantizer.SATCell(
                range=255, 
                apply_scaling=True, 
                apply_quantization=True,
        ))
        
        conv_solver = model._cells[cell_name].get_weights_solver()
        conv_solver.set_parameter("learning_rate_policy", "CosineDecay")
        conv_solver.set_parameter("learning_rate", 0.05)
        conv_solver.set_parameter("momentum", 0.9)
        conv_solver.set_parameter("decay", 0.00004)
        conv_solver.set_parameter("max_iterations", 115305030)

        conv_solver = model._cells[cell_name].get_bias_solver()
        conv_solver.set_parameter("learning_rate_policy", "CosineDecay")
        conv_solver.set_parameter("learning_rate", 0.05)
        conv_solver.set_parameter("momentum", 0.9)
        conv_solver.set_parameter("decay", 0.00004)
        conv_solver.set_parameter("max_iterations", 115305030)

    ### Updating Rectifier ###
    if isinstance(model._cells[cell_name].get_activation(), n2d2.activation.Rectifier):
        model._cells[cell_name].set_activation(
            n2d2.activation.Linear(
                quantizer=n2d2.quantizer.SATAct(
                    range=15, 
                    solver=n2d2.solver.SGD(**sgd_config)
        )))

print(model)

### Defining loss function ###
# This object containt a SoftMax layer !
loss_function =  n2d2.application.CrossEntropyClassifier(provider)

print("\n### Training ###")
for epoch in range(nb_epochs):

    provider.set_partition("Learn")
    model.learn()

    print("\n# Train Epoch: " + str(epoch) + " #")

    for i in range(math.ceil(database.get_nb_stimuli('Learn') / batch_size)):
        x = provider.read_random_batch()
        x = model(x)
        x = loss_function(x)

        x.back_propagate()
        x.update()
        x.get_deepnet().draw_graph("segmentation_graph")
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
        x = loss_function(x)

        print("Validate example: " + str(i * batch_size) + ", val success: "
              + "{0:.2f}".format(100 * loss_function.get_average_success()) + "%", end='\r')
