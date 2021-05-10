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
import N2D2
import n2d2
import unittest
from n2d2.n2d2_interface import N2D2_Interface as N2D2_Interface




class test_params(unittest.TestCase):
    def setUp(self):
        self.parameters = {
            
        }
        self.object = None

    def test_parameters(self):
        if self.object: # We don't do test if it's the dummy class
            parameters = self.object.N2D2().getParameters()
            for param in self.parameters.keys():
                if self.object.python_to_n2d2_convention(param) in parameters:
                    param_name = self.object.python_to_n2d2_convention(param)
                    N2D2_param, N2D2_type = self.object.N2D2().getParameterAndType(param_name)
                    N2D2_param = N2D2_Interface._N2D2_type_map[N2D2_type](N2D2_param)
                    if isinstance(self.parameters[param], bool):
                        self.assertEqual(self.parameters[param], bool(int(N2D2_param)))
                    elif isinstance(self.parameters[param], list):
                        dtype = type(self.parameters[param][0])
                        N2D2_param = N2D2_param.split(" ")[:-1]
                        N2D2_param = [dtype(p) for p in N2D2_param]
                        self.assertEqual(self.parameters[param], N2D2_param)
                    else:
                        self.assertEqual(self.parameters[param], N2D2_param)



### TEST CELLS ###
class test_Fc(test_params):
    def setUp(self):
        self.parameters = {
            "name": "test",
            "activation": n2d2.activation.Tanh(),
            "weights_solver": n2d2.solver.SGD(),
            "bias_solver": n2d2.solver.SGD(),
            "weights_filler": n2d2.filler.Normal(),
            "bias_filler": n2d2.filler.Normal(),
            'no_bias': True,
            "mapping": n2d2.Tensor([5, 5],  datatype=bool),
            "quantizer": n2d2.quantizer.SATCell(),
        }
        self.object = n2d2.cells.Fc(10, 5, **self.parameters)

    def test_parameters(self):
        self.assertEqual(self.parameters["name"], self.object.N2D2().getName())
        self.assertIs(self.parameters["activation"].N2D2(), self.object.N2D2().getActivation())
        self.assertIs(self.parameters["weights_solver"].N2D2(), self.object.N2D2().getWeightsSolver())
        self.assertIs(self.parameters["bias_solver"].N2D2(), self.object.N2D2().getBiasSolver())
        self.assertIs(self.parameters["weights_filler"].N2D2(), self.object.N2D2().getWeightsFiller())
        self.assertIs(self.parameters["bias_filler"].N2D2(), self.object.N2D2().getBiasFiller())
        self.assertEqual(n2d2.Tensor.from_N2D2(self.parameters["mapping"].N2D2()), 
                         n2d2.Tensor.from_N2D2(self.object.N2D2().getMapping()))
        self.assertIs(self.parameters["quantizer"].N2D2(), self.object.N2D2().getQuantizer())
        super().test_parameters()


class test_Conv(test_params):
    def setUp(self):
        self.parameters = {
            "name": "test",
            "activation": n2d2.activation.Tanh(),
            "weights_solver": n2d2.solver.SGD(),
            "sub_sample_dims": [2, 2],
            "stride_dims": [2, 2],
            "dilation_dims": [1, 1],
            "padding_dims": [2, 2],
            "bias_solver": n2d2.solver.SGD(),
            "weights_filler": n2d2.filler.Normal(),
            "bias_filler": n2d2.filler.Normal(),
            "no_bias": True,
            "back_propagate": True,
            "weights_export_flip": True,
            "mapping": n2d2.Tensor([5, 5],  datatype=bool),
            "quantizer": n2d2.quantizer.SATCell(),
        }
        self.object = n2d2.cells.Conv(10, 5, [2, 2], **self.parameters)

    def test_parameters(self):
        self.assertEqual(self.parameters["name"], self.object.N2D2().getName())
        self.assertIs(self.parameters["activation"].N2D2(), self.object.N2D2().getActivation())
        self.assertIs(self.parameters["weights_solver"].N2D2(), self.object.N2D2().getWeightsSolver())
        self.assertIs(self.parameters["bias_solver"].N2D2(), self.object.N2D2().getBiasSolver())
        self.assertIs(self.parameters["weights_filler"].N2D2(), self.object.N2D2().getWeightsFiller())
        self.assertIs(self.parameters["bias_filler"].N2D2(), self.object.N2D2().getBiasFiller())
        self.assertEqual(self.parameters["sub_sample_dims"], [self.object.N2D2().getSubSampleX(), self.object.N2D2().getSubSampleY()])
        self.assertEqual(self.parameters["stride_dims"], [self.object.N2D2().getStrideX(), self.object.N2D2().getStrideY()])
        self.assertEqual(self.parameters["padding_dims"], [self.object.N2D2().getPaddingX(), self.object.N2D2().getPaddingY()])
        self.assertEqual(self.parameters["dilation_dims"], [self.object.N2D2().getDilationX(), self.object.N2D2().getDilationY()])
        self.assertEqual(n2d2.Tensor.from_N2D2(self.parameters["mapping"].N2D2()), 
                         n2d2.Tensor.from_N2D2(self.object.N2D2().getMapping()))
        self.assertIs(self.parameters["quantizer"].N2D2(), self.object.N2D2().getQuantizer())

        super().test_parameters()

class test_Softmax(test_params):
    def setUp(self):
        self.parameters = {
            "name": "test",
            "with_loss": True,
            "group_size": 1,
        }
        self.object = n2d2.cells.Softmax(**self.parameters)

    def test_parameters(self):
        # Need to instantiate the object (doing so by passing a dummy input)
        tensor = n2d2.Tensor([1, 5, 4, 4], cuda=True)
        self.object(tensor)

        self.assertEqual(self.parameters["name"], self.object.N2D2().getName())
        self.assertEqual(self.parameters["with_loss"], self.object.N2D2().getWithLoss())
        self.assertEqual(self.parameters["group_size"], self.object.N2D2().getGroupSize())
        super().test_parameters()

class test_Pool(test_params):
    def setUp(self):
        self.parameters = {
            "name": "test",
            "pooling": "Average",
            "stride_dims": [2, 2],
            "padding_dims": [1, 1],
            "activation": n2d2.activation.Linear(),
            "mapping": n2d2.Tensor([5, 5],  datatype=bool),
        }
        self.object = n2d2.cells.Pool([1, 1], **self.parameters)

    def test_parameters(self):
        # Need to instantiate the object (doing so by passing a dummy input)
        tensor = n2d2.Tensor([1, 5, 4, 4], cuda=True)
        self.object(tensor)

        self.assertEqual(self.parameters["name"], self.object.N2D2().getName())
        self.assertEqual(N2D2.PoolCell.Pooling.__members__[self.parameters["pooling"]], self.object.N2D2().getPooling())
        self.assertEqual(self.parameters["stride_dims"], [self.object.N2D2().getStrideX(), self.object.N2D2().getStrideY()])
        self.assertEqual(self.parameters["padding_dims"], [self.object.N2D2().getPaddingX(), self.object.N2D2().getPaddingY()])
        self.assertIs(self.parameters["activation"].N2D2(), self.object.N2D2().getActivation())
        self.assertEqual(n2d2.Tensor.from_N2D2(self.parameters["mapping"].N2D2()), 
                         n2d2.Tensor.from_N2D2(self.object.N2D2().getMapping()))
        super().test_parameters()

class test_Deconv(test_params):
    def setUp(self):
        self.parameters = {
            "name": "test",
            "activation": n2d2.activation.Linear(),
            "weights_solver": n2d2.solver.SGD(),
            "stride_dims": [2, 2],
            "dilation_dims": [1, 1],
            "padding_dims": [2, 2],
            "bias_solver": n2d2.solver.SGD(),
            "weights_filler": n2d2.filler.Normal(),
            "bias_filler": n2d2.filler.Normal(),
            "no_bias": True,
            "back_propagate": True,
            "weights_export_flip": True,
            "mapping": n2d2.Tensor([5, 5],  datatype=bool),
        }
        self.object = n2d2.cells.Deconv(10, 5, [2, 2], **self.parameters)

    def test_parameters(self):
        self.assertEqual(self.parameters["name"], self.object.N2D2().getName())
        self.assertIs(self.parameters["activation"].N2D2(), self.object.N2D2().getActivation())
        self.assertIs(self.parameters["weights_solver"].N2D2(), self.object.N2D2().getWeightsSolver())
        self.assertIs(self.parameters["bias_solver"].N2D2(), self.object.N2D2().getBiasSolver())
        self.assertIs(self.parameters["weights_filler"].N2D2(), self.object.N2D2().getWeightsFiller())
        self.assertIs(self.parameters["bias_filler"].N2D2(), self.object.N2D2().getBiasFiller())
        self.assertEqual(self.parameters["stride_dims"], [self.object.N2D2().getStrideX(), self.object.N2D2().getStrideY()])
        self.assertEqual(self.parameters["padding_dims"], [self.object.N2D2().getPaddingX(), self.object.N2D2().getPaddingY()])
        self.assertEqual(self.parameters["dilation_dims"], [self.object.N2D2().getDilationX(), self.object.N2D2().getDilationY()])
        self.assertEqual(n2d2.Tensor.from_N2D2(self.parameters["mapping"].N2D2()), 
                         n2d2.Tensor.from_N2D2(self.object.N2D2().getMapping()))
        super().test_parameters()

class test_ElemWise(test_params):
    def setUp(self):
        self.parameters = {
            "name": "test",
            "operation": "Max",
            "mode": "PerInput",
            "weights": [0.5],
            "shifts": [0.5],
            "activation": n2d2.activation.Linear(),
        }
        self.object = n2d2.cells.ElemWise(**self.parameters)

    def test_parameters(self):
        # Need to instantiate the object (doing so by passing a dummy input)
        tensor = n2d2.Tensor([1, 5, 4, 4], cuda=True)
        self.object(tensor)

        self.assertEqual(self.parameters["name"], self.object.N2D2().getName())
        self.assertEqual(N2D2.ElemWiseCell.Operation.__members__[self.parameters["operation"]], 
                        self.object.N2D2().getOperation())
        self.assertEqual(self.parameters["weights"], self.object.N2D2().getWeights())
        self.assertEqual(self.parameters["shifts"], self.object.N2D2().getShifts())
        self.assertIs(self.parameters["activation"].N2D2(), self.object.N2D2().getActivation())

        super().test_parameters()

class test_Dropout(test_params):
    def setUp(self):
        self.parameters = {
            "name": "test",
            "dropout": 0.3,
        }
        self.object = n2d2.cells.Dropout(**self.parameters)

    def test_parameters(self):
        # Need to instantiate the object (doing so by passing a dummy input)
        tensor = n2d2.Tensor([1, 5, 4, 4], cuda=True)
        self.object(tensor)
        self.assertEqual(self.parameters["name"], self.object.N2D2().getName())
        self.assertEqual(self.parameters["dropout"], self.object.N2D2().getDropout())
        super().test_parameters()

class test_Padding(test_params):
    def setUp(self):
        self.parameters = {
            "name": "test",
            "top_pad": 1,
            "bot_pad":0,
            "left_pad": 0,
            "right_pad": 1,
        }
        self.object = n2d2.cells.Padding(**self.parameters)

    def test_parameters(self):
        # Need to instantiate the object (doing so by passing a dummy input)
        tensor = n2d2.Tensor([1, 5, 4, 4], cuda=True)
        self.object(tensor)
        self.assertEqual(self.parameters["top_pad"], self.object.N2D2().getTopPad())
        self.assertEqual(self.parameters["bot_pad"], self.object.N2D2().getBotPad())
        self.assertEqual(self.parameters["left_pad"], self.object.N2D2().getLeftPad())
        self.assertEqual(self.parameters["right_pad"], self.object.N2D2().getRightPad())
        self.assertEqual(self.parameters["name"], self.object.N2D2().getName())
        super().test_parameters()

class test_BatchNorm2d(test_params):
    def setUp(self):
        self.parameters = {
            "name": "test",
            "nb_inputs": 5, 
            "scale_solver": n2d2.solver.SGD(),
            "bias_solver": n2d2.solver.SGD(),
            "moving_average_momentum":0,
            "epsilon": 1,
        }
        self.object = n2d2.cells.BatchNorm2d(**self.parameters)

    def test_parameters(self):
        # Need to instantiate the object (doing so by passing a dummy input)
        tensor = n2d2.Tensor([1, 5, 4, 4], cuda=True)
        self.object(tensor)
        self.assertEqual(self.parameters["name"], self.object.N2D2().getName())
        self.assertEqual(self.parameters["scale_solver"].N2D2(), self.object.N2D2().getScaleSolver())
        self.assertEqual(self.parameters["bias_solver"].N2D2(), self.object.N2D2().getBiasSolver())
        self.assertEqual(self.parameters["moving_average_momentum"], self.object.N2D2().getMovingAverageMomentum())
        self.assertEqual(self.parameters["epsilon"], self.object.N2D2().getEpsilon())

        super().test_parameters()

class test_Activation(test_params):
    def setUp(self):
        self.parameters = {
            "name": "test",
            "activation": n2d2.activation.Tanh(),
        }
        self.object = n2d2.cells.Activation(**self.parameters)

    def test_parameters(self):
        # Need to instantiate the object (doing so by passing a dummy input)
        tensor = n2d2.Tensor([1, 5, 4, 4], cuda=True)
        self.object(tensor)
        self.assertIs(self.parameters["activation"].N2D2(), self.object.N2D2().getActivation())
        self.assertEqual(self.parameters["name"], self.object.N2D2().getName())
        super().test_parameters()

class test_Reshape(test_params):
    def setUp(self):
        self.parameters = {
            "name": "test",
            "dims": [4, 4, 5],
        }
        self.object = n2d2.cells.Reshape(**self.parameters)

    def test_parameters(self):
        # Need to instantiate the object (doing so by passing a dummy input)
        tensor = n2d2.Tensor([1, 5, 4, 4], cuda=True)
        self.object(tensor)
        self.assertEqual(self.parameters["name"], self.object.N2D2().getName())
        self.assertEqual(self.parameters["dims"], self.object.N2D2().getDims())
        super().test_parameters()

### TEST DATABASE ###

class test_DIR(test_params):
    def setUp(self):
        self.parameters = {
            "load_data_in_memory": True,
        }
        self.object = n2d2.database.DIR(**self.parameters)

    def test_parameters(self):
        # Need to instantiate the object (doing so by passing a dummy input)
        self.assertEqual(self.parameters["load_data_in_memory"], self.object.N2D2().getLoadDataInMemory())
        super().test_parameters()

class test_MNIST(test_params):
    def setUp(self):
        self.parameters = {
            "label_path": "",
            "extract_roi": True,
            "validation": 0.2
        }
        self.object = n2d2.database.MNIST("/nvme0/DATABASE/MNIST/raw/", **self.parameters)

    def test_parameters(self):
        super().test_parameters()


class test_CIFAR100(test_params):
    def setUp(self):
        self.parameters = {
            "use_coarse": True,
            "validation": 0.2,
            "use_test_for_validation": False,
        }
        self.object = n2d2.database.CIFAR100(**self.parameters)

    def test_parameters(self):
        self.assertEqual(self.parameters["use_coarse"], self.object.N2D2().getUseCoarse())
        self.assertEqual(self.parameters["use_test_for_validation"], self.object.N2D2().getUseTestForValidation())
        super().test_parameters()

class test_ILSVRC2012(test_params):
    def setUp(self):
        self.parameters = {
            "learn": 0.2,
            "use_validation_for_test": True,
            "background_class": False,
        }
        self.object = n2d2.database.ILSVRC2012(**self.parameters)

    def test_parameters(self):
        self.assertEqual(self.parameters["learn"], self.object.N2D2().getLearn())
        self.assertEqual(self.parameters["background_class"], self.object.N2D2().getBackgroundClass())
        self.assertEqual(self.parameters["use_validation_for_test"], self.object.N2D2().getUseValidationForTest())
        super().test_parameters()

class test_Cityscapes(test_params):
    def setUp(self):
        self.parameters = {
            "inc_train_extra": False,
            "use_coarse": True,
            "single_instance_labels": False,
        }
        self.object = n2d2.database.Cityscapes(**self.parameters)

    def test_parameters(self):
        self.assertEqual(self.parameters["inc_train_extra"], self.object.N2D2().getIncTrainExtra())
        self.assertEqual(self.parameters["use_coarse"], self.object.N2D2().getUseCoarse())
        self.assertEqual(self.parameters["single_instance_labels"], self.object.N2D2().getSingleInstanceLabels())
        super().test_parameters()

class test_GTSRB(test_params):
    def setUp(self):
        self.parameters = {
            "validation": 0.2,
        }
        self.object = n2d2.database.GTSRB(**self.parameters)

    def test_parameters(self):
        self.assertEqual(self.parameters["validation"], self.object.N2D2().getValidation())
        super().test_parameters()
    
### Transformation ###

class test_PadCrop(test_params):
    def setUp(self):
        self.parameters = {
            "width": 10,
            "height":10,
            "additive_wh": False,
            "border_type": "WrapBorder",
            "border_value": [0.0, 0.0, 0.0],
        }
        self.object = n2d2.transform.PadCrop(**self.parameters)

    def test_parameters(self):
        self.assertEqual(self.parameters["width"], self.object.N2D2().getWidth())
        self.assertEqual(self.parameters["height"], self.object.N2D2().getHeight())
        self.assertEqual(self.parameters["additive_wh"], self.object.N2D2().getAdditiveWH())
        self.assertEqual(self.object.N2D2().BorderType.__members__[self.parameters["border_type"]],
                        self.object.N2D2().getBorderType())
        self.assertEqual(self.parameters["border_value"], self.object.N2D2().getBorderValue())
        super().test_parameters()

class test_Distortion(test_params):
    def setUp(self):
        self.parameters = {
            "elastic_gaussian_size": 10,
            "elastic_sigma": 5.0,
            "elastic_scaling": 0.0,
            "scaling": 0.0,
            "rotation": 0.0,
            "ignore_missing_data": False,
        }
        self.object = n2d2.transform.Distortion(**self.parameters)

    def test_parameters(self):
        self.assertEqual(self.parameters["elastic_gaussian_size"], self.object.N2D2().getElasticGaussianSize())
        self.assertEqual(self.parameters["elastic_sigma"], self.object.N2D2().getElasticSigma())
        self.assertEqual(self.parameters["elastic_scaling"], self.object.N2D2().getElasticScaling())
        self.assertEqual(self.parameters["scaling"], self.object.N2D2().getScaling())
        self.assertEqual(self.parameters["rotation"], self.object.N2D2().getRotation())
        self.assertEqual(self.parameters["ignore_missing_data"], self.object.N2D2().getIgnoreMissingData())
        super().test_parameters()

class test_Rescale(test_params):
    def setUp(self):
        self.parameters = {
            "height": 10,
            "width": 10,
            "keep_aspect_ratio": True,
            "resize_to_fit": False,
        }
        self.object = n2d2.transform.Rescale(**self.parameters)

    def test_parameters(self):
        self.assertEqual(self.parameters["height"], self.object.N2D2().getHeight())
        self.assertEqual(self.parameters["width"], self.object.N2D2().getWidth())
        self.assertEqual(self.parameters["keep_aspect_ratio"], self.object.N2D2().getKeepAspectRatio())
        self.assertEqual(self.parameters["resize_to_fit"], self.object.N2D2().getResizeToFit())
        super().test_parameters()

class test_ColorSpace(test_params):
    def setUp(self):
        self.parameters = {
            "color_space": "CIELab",
        }
        self.object = n2d2.transform.ColorSpace(**self.parameters)

    def test_parameters(self):
        self.assertEqual(self.object.N2D2().ColorSpace.__members__[self.parameters["color_space"]],
                        self.object.N2D2().getColorSpace())
        super().test_parameters()

class test_RangeAffine(test_params):
    def setUp(self):
        self.parameters = {
            "first_operator": "Plus",
            "first_value": [1.0],
            "second_operator": "Plus",
            "second_value": [1.0],
            "truncate": False,
        }
        self.object = n2d2.transform.RangeAffine(**self.parameters)

    def test_parameters(self):
        self.assertEqual(self.object.N2D2().Operator.__members__[self.parameters["first_operator"]],
                        self.object.N2D2().getFirstOperator())
        self.assertEqual(self.parameters["first_value"], self.object.N2D2().getFirstValue())
        self.assertEqual(self.object.N2D2().Operator.__members__[self.parameters["second_operator"]],  
                        self.object.N2D2().getSecondOperator())
        self.assertEqual(self.parameters["second_value"], self.object.N2D2().getSecondValue())
        self.assertEqual(self.parameters["truncate"], self.object.N2D2().getTruncate())
        super().test_parameters()
class test_SliceExtraction(test_params):
    def setUp(self):
        self.parameters = {
            "width": 4,
            "height": 2,
            "offset_x": 0,
            "offset_y": 0,
            "random_offset_x":False,
            "random_offset_y":False,
            "random_rotation":True,
            "random_scaling": True,
            "random_rotation_range": [0.0, 1.0],
            "random_scaling_range": [0.0, 1.0],
            "allow_padding": True,
            "border_type": "WrapBorder",
            "border_value": [1.0],
        }
        self.object = n2d2.transform.SliceExtraction(**self.parameters)

    def test_parameters(self):
        self.assertEqual(self.parameters["width"], self.object.N2D2().getWidth())
        self.assertEqual(self.parameters["height"], self.object.N2D2().getHeight())
        self.assertEqual(self.parameters["offset_x"], self.object.N2D2().getOffsetX())
        self.assertEqual(self.parameters["offset_y"], self.object.N2D2().getOffsetY())
        self.assertEqual(self.parameters["random_offset_x"], self.object.N2D2().getRandomOffsetX())
        self.assertEqual(self.parameters["random_offset_y"], self.object.N2D2().getRandomOffsetY())
        self.assertEqual(self.parameters["random_rotation"], self.object.N2D2().getRandomRotation())
        self.assertEqual(self.parameters["random_rotation_range"], self.object.N2D2().getRandomRotationRange())
        self.assertEqual(self.parameters["random_scaling"], self.object.N2D2().getRandomScaling())
        self.assertEqual(self.parameters["random_scaling_range"], self.object.N2D2().getRandomScalingRange())
        self.assertEqual(self.parameters["allow_padding"], self.object.N2D2().getAllowPadding())
        self.assertEqual(self.object.N2D2().BorderType.__members__[self.parameters["border_type"]],  
                        self.object.N2D2().getBorderType())
        self.assertEqual(self.parameters["border_value"], self.object.N2D2().getBorderValue())
        super().test_parameters()

class test_Flip(test_params):
    def setUp(self):
        self.parameters = {
            "horizontal_flip": True,
            "vertical_flip": True,
            "random_horizontal_flip": True,
            "random_vertical_flip": True,
        }
        self.object = n2d2.transform.Flip(**self.parameters)

    def test_parameters(self):
        self.assertEqual(self.parameters["horizontal_flip"], self.object.N2D2().getHorizontalFlip())
        self.assertEqual(self.parameters["vertical_flip"], self.object.N2D2().getVerticalFlip())
        self.assertEqual(self.parameters["random_horizontal_flip"], self.object.N2D2().getRandomHorizontalFlip())
        self.assertEqual(self.parameters["random_vertical_flip"], self.object.N2D2().getRandomVerticalFlip())

        super().test_parameters()

class test_RandomResizeCrop(test_params):
    def setUp(self):
        self.parameters = {
            "width": 10,
            "height": 10,
            "offset_x": 0,
            "offset_y": 0,
            "scale_min": 0.0,
            "scale_max": 0.0,
            "ratio_min": 0.0,
            "ratio_max": 0.0,
        }
        self.object = n2d2.transform.RandomResizeCrop(**self.parameters)

    def test_parameters(self):
        self.assertEqual(self.parameters["width"], self.object.N2D2().getWidth())
        self.assertEqual(self.parameters["height"], self.object.N2D2().getHeight())
        self.assertEqual(self.parameters["offset_x"], self.object.N2D2().getOffsetX())
        self.assertEqual(self.parameters["offset_y"], self.object.N2D2().getOffsetY())
        self.assertEqual(self.parameters["scale_min"], self.object.N2D2().getScaleMin())
        self.assertEqual(self.parameters["scale_max"], self.object.N2D2().getScaleMax())
        self.assertEqual(self.parameters["ratio_min"], self.object.N2D2().getRatioMin())
        self.assertEqual(self.parameters["ratio_max"], self.object.N2D2().getRatioMax())
        super().test_parameters()

class test_ChannelExtraction(test_params):
    def setUp(self):
        self.parameters = {
            "channel": "Green",
        }
        self.object = n2d2.transform.ChannelExtraction(**self.parameters)

    def test_parameters(self):
        self.assertEqual(self.object.N2D2().Channel.__members__[self.parameters["channel"]],  
                        self.object.N2D2().getChannel())
        super().test_parameters()

### Filler ###
class test_He(test_params):
    def setUp(self):
        self.parameters = {
            "datatype": "float",
            "variance_norm": 'Average',
            "scaling": 1.0,
            "mean_norm": 1.0,
        }
        self.object = n2d2.filler.He(**self.parameters)

    def test_parameters(self):
        self.assertEqual(self.parameters["mean_norm"], self.object.N2D2().getMeanNorm())
        self.assertEqual(self.parameters["scaling"], self.object.N2D2().getScaling())
        self.assertEqual(self.object.N2D2().VarianceNorm.__members__[self.parameters["variance_norm"]],  
                        self.object.N2D2().getVarianceNorm())
        super().test_parameters()

class test_Normal(test_params):
    def setUp(self):
        self.parameters = {
            "datatype": "float",
            "mean": 0.0,
            "std_dev": 1.0,
        }
        self.object = n2d2.filler.Normal(**self.parameters)

    def test_parameters(self):
        self.assertEqual(self.parameters["mean"], self.object.N2D2().getMean())
        self.assertEqual(self.parameters["std_dev"], self.object.N2D2().getStdDev())
        super().test_parameters()

class test_Xavier(test_params):
    def setUp(self):
        self.parameters = {
            "datatype": "float",
            "variance_norm": "FanOut",
            "distribution": "Normal",
            "scaling": 1.0, 
        }
        self.object = n2d2.filler.Xavier(**self.parameters)

    def test_parameters(self):
        self.assertEqual(self.object.N2D2().VarianceNorm.__members__[self.parameters["variance_norm"]],
                        self.object.N2D2().getVarianceNorm())
        self.assertEqual(self.object.N2D2().Distribution.__members__[self.parameters["distribution"]],  
                        self.object.N2D2().getDistribution())
        self.assertEqual(self.parameters["scaling"], self.object.N2D2().getScaling())
        super().test_parameters()

class test_Constant(test_params):
    def setUp(self):
        self.parameters = {
            "datatype": "float",
            "value": 1.0,
        }
        self.object = n2d2.filler.Constant(**self.parameters)

    def test_parameters(self):
        self.assertEqual(self.parameters["value"], self.object.N2D2().getValue())
        super().test_parameters()


### Solver ###

class test_SGD(test_params):
    def setUp(self):
        self.parameters = {
            "datatype": "float",
            "model": "Frame",
            "learning_rate": 0.001,
            "momentum": 0.0,
            "decay": 0.0,
            "power": 1.0,
            "iteration_size": 1,
            "max_iterations": 1,
            "warm_up_duration": 0,
            "warm_up_lr_frac": 1.0,
            "learning_rate_policy": "None",
            "learning_rate_step_size": 1,
            "learning_rate_decay": 0.1,
            "clamping": False,
            "polyak_momentum": False,
        }
        self.object = n2d2.solver.SGD(**self.parameters)

    def test_parameters(self):
        self.assertEqual(self.parameters["learning_rate"], self.object.N2D2().getmLearningRate())
        self.assertEqual(self.parameters["momentum"], self.object.N2D2().getMomentum())
        self.assertEqual(self.parameters["decay"], self.object.N2D2().getDecay())
        self.assertEqual(self.parameters["power"], self.object.N2D2().getPower())
        self.assertEqual(self.parameters["iteration_size"], self.object.N2D2().getIterationSize())
        self.assertEqual(self.parameters["max_iterations"], self.object.N2D2().getMaxIterations())
        self.assertEqual(self.parameters["warm_up_duration"], self.object.N2D2().getWarmUpDuration())
        self.assertEqual(self.parameters["warm_up_lr_frac"], self.object.N2D2().getWarmUpLRFrac())
        self.assertEqual(self.object.N2D2().LearningRatePolicy.__members__[self.parameters["learning_rate_policy"]],  
                        self.object.N2D2().getLearningRatePolicy())
        self.assertEqual(self.parameters["learning_rate_step_size"], self.object.N2D2().getLearningRateStepSize())
        self.assertEqual(self.parameters["learning_rate_decay"], self.object.N2D2().getLearningRateDecay())
        self.assertEqual(self.parameters["clamping"], bool(int(self.object.N2D2().getmClamping())))
        self.assertEqual(self.parameters["polyak_momentum"], bool(int(self.object.N2D2().getPolyakMomentum())))

        super().test_parameters()

class test_Adam(test_params):
    def setUp(self):
        self.parameters = {
            "datatype": "float",
            "model": "Frame",
            "learning_rate": 0.001,
            "clamping": False,
            "beta1": 0.1,
            "beta2": 0.1,
            "epsilon": 0.1,
        }
        self.object = n2d2.solver.Adam(**self.parameters)

    def test_parameters(self):
        self.assertEqual(self.parameters["learning_rate"], self.object.N2D2().getmLearningRate())
        self.assertEqual(self.parameters["clamping"], bool(int(self.object.N2D2().getmClamping())))
        self.assertEqual(self.parameters["beta1"], self.object.N2D2().getBeta1())
        self.assertEqual(self.parameters["beta2"], self.object.N2D2().getBeta2())
        self.assertEqual(self.parameters["epsilon"], self.object.N2D2().getEpsilon())
        super().test_parameters()

### Activation ###

class Linear(test_params):
    def setUp(self):
        self.parameters = {
            "quantizer": n2d2.quantizer.SATAct(),
        }
        self.object = n2d2.activation.Linear(**self.parameters)

    def test_parameters(self):
        self.assertIs(self.parameters["quantizer"].N2D2(), self.object.N2D2().getQuantizer())
        super().test_parameters()

class Rectifier(test_params):
    def setUp(self):
        self.parameters = {
            "leak_slope": 0.0,
            "clipping": 0.0,
            "quantizer": n2d2.quantizer.SATAct(),
        }
        self.object = n2d2.activation.Rectifier(**self.parameters)

    def test_parameters(self):
        self.assertEqual(self.parameters["leak_slope"], self.object.N2D2().getLeakSlope())
        self.assertEqual(self.parameters["clipping"], self.object.N2D2().getClipping())
        self.assertIs(self.parameters["quantizer"].N2D2(), self.object.N2D2().getQuantizer())
        super().test_parameters()

class Tanh(test_params):
    def setUp(self):
        self.parameters = {
            "alpha": 0.0,
            "quantizer": n2d2.quantizer.SATAct(),
        }
        self.object = n2d2.activation.Tanh(**self.parameters)

    def test_parameters(self):
        self.assertEqual(self.parameters["alpha"], self.object.N2D2().getAlpha())
        self.assertIs(self.parameters["quantizer"].N2D2(), self.object.N2D2().getQuantizer())
        super().test_parameters()

### Provider ###

class DataProvider(test_params):
    def setUp(self):
        self.parameters = {
            "database": n2d2.database.Database(),
            "batch_size": 1,
            "size": [10, 10],
            "composite_stimuli": False,
            "random_read":True,
        }
        self.object = n2d2.provider.DataProvider(**self.parameters)

    def test_parameters(self):
        self.assertIs(self.parameters["database"].N2D2(), self.object.N2D2().getDatabase())
        self.assertEqual(self.parameters["batch_size"], self.object.N2D2().getBatchSize())
        self.assertEqual(self.parameters["size"], self.object.N2D2().getSize())
        self.assertEqual(self.parameters["composite_stimuli"], self.object.N2D2().isCompositeStimuli())
        super().test_parameters()


# print(self.object.N2D2().getParameters()) 
if __name__ == '__main__':
    unittest.main()