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
import tf2onnx
# from onnxsim import simplify
# import onnx
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

import N2D2
import n2d2

class CustomSequential(keras.Sequential):
    def __init__(self, deepNetCell, batch_size, outputs_shape, name=None, **kwargs):
        super(CustomSequential, self).__init__([], name, **kwargs)
        self.deepNet= deepNetCell._embedded_deepnet.N2D2()
        self._deepnet_cell = deepNetCell
        self.batch_size=batch_size
        self.quant_model = None
        self.outputs_shape = outputs_shape
        self._transpose_input = n2d2.cells.Transpose([1,2,0,3])
        # transpose_grad is the inverse operation of transpose_input
        self._transpose_grad = n2d2.cells.Transpose([2,0,1,3])

    def compile(self, *args, **kwargs):
        # TODO : We can update N2D2 Solver here.

        # By default, eager mode is disabled with compile(), preventing the use
        # of .numpy() in call()
        # FIXME: If this Sequential is inside another block, this is not enough!
        super(CustomSequential, self).compile(*args, **kwargs, run_eagerly=True)


    @tf.custom_gradient
    def custom_op(self, x):
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


        def custom_grad(dy):
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
            return dx

        return y, custom_grad

    def call(self, inputs, training=None, mask=None):
        self.training=training if training is not None else False
        if self.quant_model is not None:
            return self.quant_model(inputs=inputs)
        else:
            with backprop.GradientTape() as tape:
                inputs_var = tf.Variable(inputs)
                outputs = self.custom_op(inputs_var)
            return outputs


def wrap(tf_model, batch_size, name=None, for_export=False):
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
    :return:
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
    output_names = [t.name for t in frozen_func.outputs]
    model_name = name if name is not None \
                else tf_model.name if tf_model.name is not None\
                else "model"
    
    spec = [tf.TensorSpec(inputs_shape, tf.float32, name=input_name) for input_name in input_names]

    tf2onnx.convert.from_keras(
                tf_model,
                input_signature=spec, 
                opset=10,
                inputs_as_nchw=input_names,
                output_path=model_name + ".onnx")
                # output_path= "raw_" + model_name + ".onnx")

    # print("Simplifying the ONNX model ...")
    # onnx_model = onnx.load("raw_" + model_name + ".onnx")
    # model_simp, check = simplify(onnx_model)
    # assert check, "Simplified ONNX model could not be validated" # TODO : if check fail try to use the raw model !
    # onnx.save(model_simp, model_name + ".onnx")

    # Import ONNX in N2D2
    if n2d2.global_variables.cuda_compiled:
        n2d2.global_variables.default_model = "Frame_CUDA"
        n2d2.global_variables.cuda_device = 0

    db = n2d2.database.Database()
    provider = n2d2.provider.DataProvider(db, [inputs_shape[2], inputs_shape[1], inputs_shape[3]], batch_size=inputs_shape[0])

    deepNetCell = n2d2.cells.DeepNetCell.load_from_ONNX(provider, model_name + ".onnx")

    previous_cell = None
    for cell in deepNetCell:
        # Layers modification after the import !
        if isinstance(cell, n2d2.cells.Softmax):
            # ONNX import Softmax with_loss = True supposing we are using a CrossEntropy loss.
            cell.with_loss = False
        if for_export:
            # Keras add Reshape before FullyConnected layers, which are not exportable.
            if isinstance(cell, n2d2.cells.Fc) and isinstance(previous_cell, n2d2.cells.Reshape):
                try:
                    deepNetCell.remove(previous_cell.get_name(), reconnect=True)
                except RuntimeError as err:
                    raise RuntimeError(f'N2D2 could not remove the layer "{previous_cell.get_name()}".\n \
                        If you do not need to export this network, try to set "for_export=False"') from err
        previous_cell = cell
    
    deepNetCell._embedded_deepnet.N2D2().initialize()

    N2D2.DrawNet.drawGraph(deepNetCell._embedded_deepnet.N2D2(), "model")
    return CustomSequential(deepNetCell, batch_size, outputs_shape, name=model_name)

