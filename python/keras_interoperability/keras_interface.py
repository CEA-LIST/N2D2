"""
(C) Copyright 2021 CEA LIST. All Rights Reserved.
Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                Cyril MOINEAU (cyril.moineau@cea.fr)


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
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.eager import backprop
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from warnings import warn

import tf2onnx

import N2D2
import n2d2

# pylint: disable=protected-access

class CustomSequential(keras.Sequential):
    """A customSequential model which embedded an N2D2 Network.
    """
    def __init__(self, deepnet_cell: n2d2.cells.DeepNetCell,
                 batch_size: int, outputs_shape: tf.Tensor, name: str=None, **kwargs) -> None:
        """
        :param deepnet_cell: The network used for propagation and backpropagation.
        :type deepnet_cell: :py:class:`n2d2.cells.DeepNetCell`
        :param batch_size: Batch size used
        :type batch_size: int
        :param output_shape: Shape of the output.
        :type output_shape: tf.Tensor
        :param name: Name of the model, default=None
        :type name: str, optional
        """
        super(CustomSequential, self).__init__([], name, **kwargs)
        self.deepNet= deepnet_cell._embedded_deepnet.N2D2()
        self._deepnet_cell = deepnet_cell
        self.batch_size = batch_size
        self.quant_model = None
        self.outputs_shape = outputs_shape
        self._transpose_input = n2d2.cells.Transpose([1, 2, 0, 3])
        # transpose_grad is the inverse operation of transpose_input
        self._transpose_grad = n2d2.cells.Transpose([2, 0, 1, 3])

    def get_deepnet_cell(self) -> n2d2.cells.DeepNetCell:
        """
        :return: The DeepNet Cell used by the custom sequential
        :rtype: :py:class:`n2d2.cells.DeepNetCell`
        """
        return self._deepnet_cell

    def compile(self, optimizer:n2d2.solver.Solver=None ,*args, **kwargs) -> None:
        """Overwritten Tensorflow compile method.

        :param optimizer: The N2D2 solver used to optimize weights and biases, default=:py:class:`n2d2.solver.SGD`
        :type optimizer: :py:class:`n2d2.solver.Solver`, optional
        """
        if optimizer is not None:
            if isinstance(optimizer, n2d2.solver.Solver):
                self._deepnet_cell.set_solver(optimizer)
            elif isinstance(optimizer, str) or isinstance(optimizer, keras.optimizers.Optimizer):
                raise TypeError(f"The model '{self.name}' embed an N2D2 Network, the N2D2 parameters are not visible " + \
                "by Keras Solver and thus cannot be optimized by a Keras optimizer. You should use an 'n2d2.solver.Solver'.")
            else:
                raise n2d2.error_handler.WrongInputType("optimizer", optimizer, ["n2d2.solver.Solver"])
        # By default, eager mode is disabled with compile(), preventing the use
        # of .numpy() in call()
        # FIXME: If this Sequential is inside another block, this is not enough!
        super(CustomSequential, self).compile(*args, **kwargs, run_eagerly=True)


    @tf.custom_gradient
    def custom_op(self, x: tf.Tensor):
        """Method to handle propagation
        """
        x_var = tf.Variable(x)

        x_numpy = x.numpy()
        inputs_batch_size = x_numpy.shape[0] # TODO : Check size is the same as input shape of the network ?
        inputs_shape = np.array(x_numpy.shape)
        # Make sure we have a full batch
        if inputs_batch_size < self.batch_size:
            inputs_shape[0] = self.batch_size
            x_numpy.resize(inputs_shape)

        if len(inputs_shape) == 2:
            # Adding two unit dimensions
            x_numpy = x_numpy.reshape(self.batch_size, 1, 1, -1)

        firstCellName = self.deepNet.getLayers()[1][0] # 0 = env
        lastCellName = self.deepNet.getLayers()[-1][-1]

        # Transposing input to respect nchw N2D2 input
        x_n2d2 = n2d2.Tensor.from_numpy(x_numpy)
        x_n2d2 = self._transpose_input(x_n2d2)
        x_tensor = x_n2d2.N2D2()

        firstCell = self.deepNet.getCell_Frame_Top(firstCellName)
        self.diffOutputs = N2D2.Tensor_float(x_n2d2.dims())
        firstCell.clearInputs()
        # Need to add Input like this to initialize diffOutputs,
        # else we get a segFault when backPropagating because diffOutput would not be initialized !
        firstCell.addInputBis(x_tensor, self.diffOutputs)

        self.deepNet.propagate(not self.training)
        y_tensor = self.deepNet.getCell_Frame_Top(lastCellName).getOutputs()
        y_tensor.synchronizeDToH()
        y_numpy = np.array(y_tensor)

        # Set the correct output shape
        # N2D2 output shape is always 4 dimensions, but expected dimension
        # can be lower (if output of a Fc cell for example).
        outputs_shape = np.array(self.outputs_shape)
        outputs_shape[0] = self.batch_size   # set batch size

        y_numpy = y_numpy.reshape(outputs_shape)

        if inputs_batch_size < self.batch_size:
            outputs_shape[0] = inputs_batch_size

            y_numpy = np.copy(y_numpy)
            y_numpy.resize(outputs_shape)

        y = tf.convert_to_tensor(y_numpy, dtype=tf.float32)


        def custom_grad(dy:tf.Tensor):
            """Method to handle backpropagation
            """
            dy_numpy = dy.numpy()
            # Make sure we have a full batch
            if inputs_batch_size < self.batch_size:
                diffInputs_shape = np.array(dy_numpy.shape)
                diffInputs_shape[0] = self.batch_size
                dy_numpy.resize(diffInputs_shape)

            # perform operation on tensor #
            dy_tensor = N2D2.Tensor_float(-dy_numpy * self.batch_size)

            diffInputs = self.deepNet.getCell_Frame_Top(lastCellName).getDiffInputs()
            dy_tensor.reshape(diffInputs.dims())
            diffInputs.op_assign(dy_tensor)

            if not diffInputs.isValid():
                diffInputs.setValid()
            diffInputs.synchronizeHToD()

            self.deepNet.backPropagate()
            self.deepNet.update()

            dx_tensor = self.deepNet.getCell_Frame_Top(firstCellName).getDiffOutputs()

            dx_tensor.synchronizeDToH()

            # transposing back the gradient
            n2d2_dx_tensor = n2d2.Tensor.from_N2D2(dx_tensor)
            n2d2_dx_tensor = self._transpose_grad(n2d2_dx_tensor)
            dx_tensor = n2d2_dx_tensor.N2D2()

            dx_numpy = np.array(dx_tensor)

            dy_tensor = N2D2.Tensor_float(-dy_numpy * self.batch_size)
            dx = tf.convert_to_tensor(-dx_numpy / self.batch_size, dtype=tf.float32)
            # print("----- GRAD -----")
            # for i in self.get_deepnet_cell():
            #     print(f"CELL  :{i.get_name()}")
            #     print("----- INPUT -----")
            #     print(f"{i.get_diffinputs()}")
            #     print("----- OUTPUT -----")
            #     print(f"{i.get_diffoutputs()}")

            # exit()
            return dx

        return y, custom_grad

    def call(self, inputs: tf.Tensor, training: bool=False)->tf.Tensor:
        """Method called to run the forward pass.

        :param  inputs: Tensor to propagate
        :type inputs: ``tf.Tensor``
        :param training: If True set the model to training mode, default=False
        :type training: bool, optional
        """
        self.training=training
        if self.quant_model is not None:
            return self.quant_model(inputs=inputs)
        else:
            with backprop.GradientTape() as tape:
                inputs_var = tf.Variable(inputs)
                outputs = self.custom_op(inputs_var)
            return outputs
    def summary(self):
        """Print model information.
        """
        print(self._deepnet_cell)

class ContextNoBatchNormFuse:
    """
    Patch: Force tf2onnx not to fuse BatchNorm into Conv.
    This is a workaround and may not work in future version of tf2onnx.
    Related merge request : https://github.com/onnx/tensorflow-onnx/pull/1907
    """
    def __enter__(self):
        self.func_map_copy = tf2onnx.optimizer.back_to_back_optimizer._func_map.copy()
        if "remove_back_to_back" in tf2onnx.optimizer._get_optimizers() and \
            ('Conv', 'BatchNormalization') in tf2onnx.optimizer.back_to_back_optimizer._func_map:
            tf2onnx.optimizer.back_to_back_optimizer._func_map.pop(('Conv', 'BatchNormalization'))
            self.fuse_removed=True
        else:
            raise RuntimeError("N2D2 could not find tf2onnx attributes this error may be due to an update" \
            " of the tf2onnx library, lowering your version of tf2onnx to 1.9.2 will solve this error. Please make sure to leave an issue at " \
            f"https://github.com/CEA-LIST/N2D2/issues stating this error message and your version of tf2onnx ({tf2onnx.__version__}) so " \
            "that we can update N2D2 to the latest version of tf2onnx.")
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        if self.fuse_removed:
            tf2onnx.optimizer.back_to_back_optimizer._func_map = self.func_map_copy

def wrap(tf_model: keras.Sequential, batch_size: int, name: str=None, for_export: bool=False) -> CustomSequential:
    """Generate a custom model which run with N2D2 on backend.
    The conversion between TensorFlow/Keras and N2D2 is done with ONNX.

    :param tf_model: The TensorFlow/Keras model to transfert to N2D2.
    :type tf_model: ``keras.Sequential``
    :param batch_size: Batch size used.
    :type batch_size: int
    :param name: Name of the model, default=tf_model.name
    :type name: str, optional
    :param for_export: If True, remove some layers to make the model exportable, default=False
    :type for_export: bool, optional
    :return: Custom sequential
    :rtype: ``keras.Sequential``
    """
    inputs_shape = np.array(tf_model.inputs[0].shape)
    inputs_shape[0] = batch_size
    outputs_shape = tf_model.outputs[0].shape
    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda inputs: tf_model(inputs))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(inputs_shape, tf_model.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)

    # Using tf2onnx
    # Preferred method, as keras2onnx is much less popular than tf2onnx
    ####################################################################
    print("Exporting the model to ONNX ...")
    input_names = [t.name for t in frozen_func.inputs]
    # output_names = [t.name for t in frozen_func.outputs]
    model_name = name if name is not None \
                else tf_model.name if tf_model.name is not None\
                else "model"

    spec = [tf.TensorSpec(inputs_shape, tf.float32, name=input_name) for input_name in input_names]

    with ContextNoBatchNormFuse() as ctx:
        tf2onnx.convert.from_keras(
            tf_model,
            input_signature=spec,
            opset=10,
            inputs_as_nchw=input_names,
            output_path=model_name + ".onnx")
            # output_path= "raw_" + model_name + ".onnx")

    # print("Simplifying the ONNX model ...")
    # onnx_model = onnx.load(model_name + ".onnx")
    # model_simp, check = simplify(onnx_model)
    # assert check, "Simplified ONNX model could not be validated"
    # onnx.save(model_simp, model_name + ".onnx")

    if n2d2.global_variables.cuda_compiled:
        # Making sure Keras did not changed the device !
        n2d2.global_variables.cuda_device = n2d2.global_variables.cuda_device

    database = n2d2.database.Database()

    if len(inputs_shape) == 4:
        input_dims = [inputs_shape[2], inputs_shape[1], inputs_shape[3]]
    elif len(inputs_shape) == 2: # Input is a Fc.
        input_dims = [inputs_shape[1], 1, 1]
    else:
        raise RuntimeError(f"Input shape {inputs_shape} is not supported.")
    provider = n2d2.provider.DataProvider(database,
                                          input_dims,
                                          batch_size=inputs_shape[0])

    deepnet_cell = n2d2.cells.DeepNetCell.load_from_ONNX(provider, model_name + ".onnx")

    previous_cell = None

    # Make a copy of the list returned by values, otherwise it would change  in size
    # when iterating.
    cells = [cell for cell in deepnet_cell._cells.values()]
    # /!\ We iterate on a copy of cells, do not remove the next cell /!\ 
    for cell in cells:
        # Layers modification after the import !
        if isinstance(cell, n2d2.cells.Softmax):
            # ONNX import Softmax with_loss = True supposing we are using a CrossEntropy loss.
            cell.with_loss = False
        if for_export:
            # Keras add Reshape before FullyConnected layers, which are not exportable.
            if isinstance(cell, n2d2.cells.Fc) and isinstance(previous_cell, n2d2.cells.Reshape):
                try:
                    deepnet_cell.remove(previous_cell.get_name(), reconnect=True)
                except RuntimeError as err:
                    raise RuntimeError(f'N2D2 could not remove the layer' \
                    f"\"{previous_cell.get_name()}\".\n" \
                    "If you do not need to export this network, try to set" \
                    "\"for_export=False\"") from err
        previous_cell = cell

    deepnet_cell._embedded_deepnet.N2D2().initialize()

    N2D2.DrawNet.drawGraph(deepnet_cell._embedded_deepnet.N2D2(), "model")
    return CustomSequential(deepnet_cell, batch_size, outputs_shape, name=model_name)
