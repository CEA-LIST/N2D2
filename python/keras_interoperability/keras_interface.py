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
from onnxsim import simplify
import onnx
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

import N2D2
import n2d2

# Work instead of using run_eagerly=True in compile()
# tf.config.experimental_functions_run_eagerly = True

class CustomSequential(keras.Sequential):
    def __init__(self, deepNet, batch_size, outputs_shape, name=None, **kwargs):
        super(CustomSequential, self).__init__([], name, **kwargs)
        self.deepNet= deepNet
        self.batch_size=batch_size
        self.quant_model = None
        self.outputs_shape = outputs_shape

    def compile(self, *args, **kwargs):
        # TODO : We can update N2D2 Solver here.

        # By default, eager mode is disabled with compile(), preventing the use
        # of .numpy() in call()
        # FIXME: If this Sequential is inside another block, this is not enough!
        super(CustomSequential, self).compile(*args, **kwargs, run_eagerly=True)


    @tf.custom_gradient
    def custom_op(self, x):
        x_var = tf.Variable(x)

        x_numpy = x.numpy()
        inputs_batch_size = x_numpy.shape[0] # TODO : Check size is the same as input shape of the network ?
        inputs_shape = np.array(x_numpy.shape)
        # Make sure we have a full batch
        if inputs_batch_size < self.batch_size:
            inputs_shape[0] = self.batch_size
            x_numpy.resize(inputs_shape)
        # print("Numpy shape : ", x_numpy.shape)
        # perform operation on tensor #
        fistCellName = self.deepNet.getLayers()[1][0] # 0 = env
        lastCellName = self.deepNet.getLayers()[-1][-1]

        x_tensor = N2D2.Tensor_float(x_numpy) # Need to change convention NHWC -> HWCN 
        
        x_tensor.reshape([inputs_shape[3], inputs_shape[1], inputs_shape[2],inputs_shape[0]])

        firstCell = self.deepNet.getCell_Frame_Top(fistCellName)
        self.diffOutputs = N2D2.Tensor_float(x_numpy.shape)
        firstCell.clearInputs()
        # Need to add Input like this to initialize diffOutputs, 
        # else we get a segFault when backPropagating because diffOutput would not be initialized !
        firstCell.addInputBis(x_tensor, self.diffOutputs) 

        # TODO: propagate() provides targets and process targets which is 
        # useless here
        self.deepNet.propagate(N2D2.Database.Learn, False, [])
        y_tensor = self.deepNet.getCell_Frame_Top(lastCellName).getOutputs()
        y_tensor.synchronizeDToH()
        y_numpy = np.array(y_tensor)
        ###############################

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

        #print("OP: ", y_numpy[0,:])

        y = tf.convert_to_tensor(y_numpy, dtype=tf.float32)


        def custom_grad(dy):
            dy_numpy = dy.numpy()

            #print("GRAD: ", dy_numpy[0,:])

            # Make sure with have a full batch
            if inputs_batch_size < self.batch_size:
                diffInputs_shape = np.array(dy_numpy.shape)
                diffInputs_shape[0] = self.batch_size

                dy_numpy.resize(diffInputs_shape)

            # perform operation on tensor #
            dy_tensor = N2D2.Tensor_float(-dy_numpy * self.batch_size)
            diffInputs = self.deepNet.getCell_Frame_Top(lastCellName).getDiffInputs()
            #FIXME: incoherency in dims
            dy_tensor.reshape(diffInputs.dims())
            diffInputs.op_assign(dy_tensor)
            diffInputs.synchronizeHToD()

            self.deepNet.backPropagate([])
            self.deepNet.update([])

            dx_tensor = self.deepNet.getCell_Frame_Top(fistCellName).getDiffOutputs()
            dx_tensor.synchronizeDToH()
            dx_numpy = np.array(dx_tensor)
            dy_tensor = N2D2.Tensor_float(-dy_numpy * self.batch_size)
            dx = tf.convert_to_tensor(-dx_numpy / self.batch_size, dtype=tf.float32)
            return dx
            # return None
        return y, custom_grad

    def call(self, inputs, training=None, mask=None):
        if self.quant_model is not None:
            # if inputs.shape[0] < self.batch_size:
            #     inputs_shape = np.array(inputs.shape)
            #     inputs_shape[0] = self.batch_size
            #     inputs.reshape(inputs_shape)

            return self.quant_model(inputs=inputs)
        else:
            with backprop.GradientTape() as tape:
                inputs_var = tf.Variable(inputs)
                outputs = self.custom_op(inputs_var)
            
            # Explicitly compute the gradient
            # !!! don't just call tape.gradient() without taking the result !!!
            #dummy = tape.gradient(outputs, inputs)

            ######## DEBUG #########
            # tf_outputs = self.tf_model.call(inputs, training, mask)
            # tf_n = tf_outputs.numpy()
            # n2d2_n =  outputs.numpy()
            # for i, j in zip(tf_n.flatten(), n2d2_n.flatten()):
            #     if(abs(float(i) - float(j)) > (0.01 * (abs(j)+ 0.0001))):
            #         print("TF tensor : ")
            #         print(tf_outputs)
            #         print("N2D2 tensor : ")
            #         print(outputs)
            #         print(f"Diff values : {i} != {j}\nShape TF : {tf_outputs.shape};\nShape N2D2 : {outputs.shape}")
            #         raise ValueError("TF and N2D2 are different !")
            ######## DEBUG #########
            return outputs



def wrap(tf_model, batch_size, name=None):
    # Don't let TF optimize these layers
    for layer in tf_model.layers:
        layer.trainable = False

    

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
    inputNames = [t.name for t in frozen_func.inputs]
    outputNames = [t.name for t in frozen_func.outputs]

    with frozen_func.graph.as_default():
        with tf.compat.v1.Session() as sess:
            onnx_graph = tf2onnx.tfonnx.process_tf_graph(sess.graph,
                opset=10,
                input_names=inputNames,
                output_names=outputNames)

            model_proto = onnx_graph.make_model("test")
            with open("raw_model.onnx", "wb") as f:
                f.write(model_proto.SerializeToString())

    print("Simplifying the ONNX model ...")
    onnx_model = onnx.load("raw_model.onnx")
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated" # TODO : if check fail try to use the raw model !
    onnx.save(model_simp, "model.onnx")

    # Import ONNX in N2D2
    n2d2.global_variables.default_model = "Frame_CUDA"
    
    db = n2d2.database.Database()
    provider = n2d2.provider.DataProvider(db,[inputs_shape[3], inputs_shape[2], inputs_shape[1]], batch_size=inputs_shape[0])
    deepNetCell = n2d2.cells.DeepNetCell.load_from_ONNX(provider, "model.onnx")

    for cell in deepNetCell:
        # Layers modification after the import !
        if isinstance(cell, n2d2.cells.Softmax):
            # ONNX import Softmax with_loss = True supposing we are using a CrossEntropy loss.
            cell.with_loss = False


    print("N2D2 model : \n", deepNetCell)

    deepNet = deepNetCell._embedded_deepnet.N2D2()
    deepNet.initialize()
    return CustomSequential(deepNet, batch_size, outputs_shape, name=name)

    # N2D2.DrawNet.drawGraph(self.deepNet, "model")

    # Now this CustomSequential model run on N2D2