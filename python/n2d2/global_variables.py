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
from os.path import expanduser

model_cache = expanduser("~") + "/MODELS"

default_seed = 1
default_model = 'Frame'
default_datatype = 'float'
default_net = N2D2.Network(default_seed)


objects_counter = {}

# TODO : Move this function to utils ?
def generate_name(obj):
    name = obj.__class__.__name__
    if name in objects_counter:
        objects_counter[name] += 1
    else:
        objects_counter[name] = 0
    name += "_"+str(objects_counter[name])
    return name

class Verbosity:
    graph_only = 0  # Only names, cell types and inputs
    short = 1  # Constructor arguments only
    detailed = 2  # Config parameters and their parameters

verbosity = Verbosity.detailed

# TODO : Move this function to utils ?
def set_cuda_device(id):
    N2D2.CudaContext.setDevice(id)

# class ConventionConverter():
#     def __init__(self, dic):
#         self.python_to_N2D2 = dic
#         self.N2D2_to_python = {values: keys for keys, values in dic.items()}
#     def p_to_n(self, key):
#         if key not in self.python_to_N2D2:
#             # TODO : remove this later ...
#             import param_generator
#             print("This line should be added to the translator !")
#             input('"' + key + '": ' + '"' +param_generator.python_to_n2d2(key) +'",')
#             return param_generator.python_to_n2d2(key)
#             # raise ValueError("Invalid parameter : " + key + " isn't registered as a valid parameter")
#         return self.python_to_N2D2[key]
#     def n_to_p(self, key):
#         if key not in self.N2D2_to_python:
#             # TODO : remove this later ...
#             import param_generator
#             print("This line should be added to the translator !")
#             input('"' + param_generator.n2d2_to_python(key) + '": ' + '"' + key +'",')
#             return param_generator.n2d2_to_python(key)
#             # raise ValueError("Invalid parameter : " + key + " isn't registered as a valid parameter")
#         return self.N2D2_to_python[key]

# convention_converter = ConventionConverter({
#     "transformation": "Transformation",
#     "elastic_gaussian_size": "ElasticGaussianSize",
#     "stride_dims": "strideDims",
#     "dropout": "Dropout",
#     "resize_to_fit": "ResizeToFit",
#     "label_path": "labelPath",
#     "weights_solver": "WeightsSolver",
#     "nb_inputs": "NbInputs",
#     "idx": "Idx",
#     "decay": "Decay",
#     "clamping": "Clamping",
#     "scale_solver": "ScaleSolver",
#     "random_rotation": "RandomRotation",
#     "activation": "ActivationFunction",
#     "composite_stimuli": "compositeStimuli",
#     "learning_rate_policy": "LearningRatePolicy",
#     "elastic_scaling": "ElasticScaling",
#     "model": "Model",
#     "output_index": "OutputIndex",
#     "horizontal_flip": "horizontalFlip",
#     "vertical_flip": "verticalFlip",
#     "random_horizontal_flip": "RandomHorizontalFlip",
#     "random_vertical_flip": "RandomVerticalFlip",
#     "border_value": "BorderValue",
#     "single_instance_labels": "singleInstanceLabels",
#     "border_type": "BorderType",
#     "second_value": "secondValue",
#     "test": "Test",
#     "elastic_sigma": "ElasticSigma",
#     "learning_rate_decay": "LearningRateDecay",
#     "learning_rate_step_size": "LearningRateStepSize",
#     "random_offset_y": "RandomOffsetY",
#     "validation": "validation",
#     "mapping": "mapping",
#     "channel": "Channel",
#     "operation": "operation",
#     "depth": "Depth",
#     "padding_dims": "paddingDims",
#     "pool_dims": "PoolDims",
#     "momentum": "Momentum",
#     "quantizer": "Quantizer",
#     "database": "Database",
#     "weights": "weights",
#     "leak_slope": "LeakSlope",
#     "datatype": "Datatype",
#     "data_path": "DataPath",
#     "kernel_dims": "KernelDims",
#     "sub_sample_dims": "subSampleDims",
#     "random_rotation_range": "RandomRotationRange",
#     "weights_export_flip": "WeightsExportFlip",
#     "name": "Name",
#     "distribution": "distribution",
#     "random_scaling": "RandomScaling",
#     "weights_filler": "WeightsFiller",
#     "label_idx": "LabelIdx",
#     "bias_solver": "BiasSolver",
#     "group_size": "groupSize",
#     "channel_index": "ChannelIndex",
#     "no_bias": "NoBias",
#     "inputs": "Inputs",
#     "offset_y": "offsetY",
#     "offset_x": "offsetX",
#     "clipping": "Clipping",
#     "random_offset_x": "RandomOffsetX",
#     "keep_aspect_ratio": "KeepAspectRatio",
#     "shifts": "shifts",
#     "alpha": "Alpha",
#     "with_loss": "withLoss",
#     "learn": "Learn",
#     "rotation": "Rotation",
#     "use_validation_for_test": "useValidationForTest",
#     "random_read": "RandomRead",
#     "use_coarse": "useCoarse",
#     "batch_size": "batchSize",
#     "color_space": "ColorSpace",
#     "nb_outputs": "NbOutputs",
#     "bias_filler": "BiasFiller",
#     "scaling": "scaling",
#     "inc_train_extra": "incTrainExtra",
#     "mean": "mean",
#     "random_scaling_range": "RandomScalingRange",
#     "value": "value",
#     "pooling": "pooling",
#     "learning_rate": "LearningRate",
#     "first_value": "FirstValue",
#     "second_operator": "secondOperator",
#     "size": "Size",
#     "dilation_dims": "dilationDims",
#     "first_operator": "FirstOperator",
#     "width": "Width",
#     "allow_padding": "AllowPadding",
#     "label_depth": "LabelDepth",
#     "height": "Height",
#     "transformations": "Transformations",
#     "back_propagate": "BackPropagate",
#     "partition": "Partition",
#     "variance_norm": "varianceNorm",
#     "std_dev": "stdDev",
#     "extract_ROIs": "extractROIs",
#     "iteration_size" : "IterationSize",
#     "max_iterations" : "MaxIterations",
#     "polyak_momentum" : "PolyakMomentum",
#     "power" : "Power",
#     "warm_up_duration": "WarmUpDuration",
#     "warm_up_l_r_frac": "WarmUpLRFrac",
#     "random_partitioning": "RandomPartitioning",
#     "load_data_in_memory": "loadDataInMemory",
#     "composite_sitmuli": "compositeStimuli",

#     "range": "Range",
#     "apply_scaling": "ApplyScaling",
#     "apply_quantization": "ApplyQuantization",
#     "quant_mode": "QuantMode",
#     "scale_min": "scaleMin",
#     "scale_max": "scaleMax",
#     "ratio_min": "ratioMin",
#     "ratio_max": "ratioMax"
# })