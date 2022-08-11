'''
@author: https://github.com/mlcommons/tiny
@license: https://apache.org/licenses/LICENSE-2.0
'''
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Note on channel location
# channels_first corresponds to inputs with shape batch_shape + (channels, spatial_dim1, spatial_dim2) or NCHW
# Channel_last (the default) corresponds to batch_shape + (spatial_dim1, spatial_dim2, channels) or NHWC

# Input is a single dimension array of length = (spectrogram length x nb of bins)
# Can be given as it is for FC. But shall be reshaped to 2D for convolution


class kws_dscnn (tf.keras.Sequential):
    def __init__(self, INPUT_H, INPUT_W, nb_of_words,
                 BNorm=True, DO_rate=0.0,
                 kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.01),
                 model_name='dscnn', for_tflite =False):
        super(kws_dscnn, self).__init__(name=model_name) #Mandatory

        # (None, InputH, InputW, 1 channel)
        self.dim_h=INPUT_H
        self.dim_w=INPUT_W
        self.BNorm=BNorm
        self.words=nb_of_words
        self.DO_rate=DO_rate
        self.kernel_initializer=kernel_initializer
        self.ReLU_max=None

        # TFLITE does no support GlobalAveragePooling layer
        # Modification of network shall be done in this case
        # Refer to bug (b/144955155) in tensorflow github
        self.tflite=for_tflite

        # Model description
        self.add(keras.Input(shape=(self.dim_h, self.dim_w,1), name="input_49_10"))

        self._conv_block(64, kernel=(10,4), strides=(2, 2))

        for i in range(1,5):
            self._dw_pw_conv_blocks(64, block_id=i)

        # Take the stride 2,2 into account
        self.dim_h=int(np.ceil(self.dim_h/float(2)))
        self.dim_w=int(np.ceil(self.dim_w/float(2)))

        if self.tflite:
            self.add(
                keras.layers.AveragePooling2D(pool_size=(self.dim_h,self.dim_w))
                )
            self.add(
                keras.layers.Flatten()
                )
        else:
            self.add(
                keras.layers.GlobalAveragePooling2D(name='global_avg_pooling')
                )

        self.add(
            keras.layers.Dense(nb_of_words,
                         activation='softmax',
                         use_bias=True,
                         kernel_initializer=kernel_initializer,
                         name='final_fc')
            )
        self.summary()

    def _conv_block(self, filters, kernel, strides):
        """
        Full convolution
        Dropout at input and output if rate>0

        """
        #channel_axis = 1 if keras.backend.image_data_format() == 'channels_first' else -1
        self.add (
        keras.layers.Conv2D(
          filters,
          kernel,
          padding='same',
          use_bias=True,
          strides=strides,
          kernel_initializer=self.kernel_initializer,
          name='conv0'
        )
        )
        self.add(keras.layers.ReLU(self.ReLU_max, name='conv0_relu'))
        self.add(keras.layers.Dropout(self.DO_rate))

    def _dw_pw_conv_blocks(self, pointwise_conv_filters, block_id=1):
        """
        Sequence of Depthwise + Pointwise convolutions
        Number of DW filters inherits from previous layer
        Pointwise kernel is 1x1
        BatchNormalization and dropout included

        """

        # Default is channel_last, HWC
        channel_axis = 1 if keras.backend.image_data_format() == 'channels_first' else -1

        # DW
        self.add(
            keras.layers.DepthwiseConv2D((3, 3),
                                     use_bias=True,
                                     padding='same',
                                     kernel_initializer=self.kernel_initializer,
                                     name='conv_dw_%d' % block_id)
            )
        if self.BNorm:
            self.add(
                keras.layers.BatchNormalization(axis=channel_axis, name='conv_dw_%d_bn' % block_id)
                )
        self.add(
            keras.layers.ReLU(self.ReLU_max, name='conv_dw_%d_relu' % block_id)
            )
        self.add(
            keras.layers.Dropout(self.DO_rate)
                )

        # PW
        self.add(
            keras.layers.Conv2D(
              pointwise_conv_filters, (1, 1),
              padding='same',
              use_bias=True,
              strides=(1, 1),
              kernel_initializer=self.kernel_initializer,
              name='conv_pw_%d' % block_id)
            )
        if self.BNorm:
            self.add(
                keras.layers.BatchNormalization(axis=channel_axis, name='conv_pw_%d_bn' % block_id)
                )
        self.add(
            keras.layers.ReLU(self.ReLU_max, name='conv_pw_%d_relu' % block_id)
            )
        self.add(
            keras.layers.Dropout(self.DO_rate)
            )
