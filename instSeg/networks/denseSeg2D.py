from .blocks import *
# from blocks import *
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Add, Conv2D, MaxPooling2D, ZeroPadding2D, Cropping2D
from tensorflow.keras.regularizers import l2
import numpy as np

'''
Author: Long Chen, LfB, RWTH Aachen University
'''


class DenseSeg2D(tf.keras.Model):

    def __init__(self,
                 nfilters=64,
                 nstage=4,
                 stage_conv=4,
                 padding='same', # 'same' 'valid', 'full'
                 dropout_type='default',  # 'spatial', 'default'
                 dropout_rate=None,
                 batch_norm=False,
                 up_type='deConv', # 'bilinear', 'upConv' or 'deConv' 
                 merge_type='cat', # 'add', 'cat'
                 kernel_initializer='he_normal',
                 use_bias=False,
                 weight_decay=1e-4,
                 name='DenseSeg2D',
                 **kwargs):

        super().__init__(name=name, **kwargs)
        assert padding in ['same', 'valid', 'full']
        assert up_type in ['bilinear', 'upConv', 'deConv' ]
        assert merge_type in ['add', 'cat']
        assert dropout_type in ['spatial', 'default']
        assert nfilters % 2 == 0
        assert nfilters % stage_conv == 0
        assert nstage >= 2

        self.padding = padding
        self.nfilters = nfilters
        self.nstage = nstage
        self.stage_conv = stage_conv
        self.merge_type = merge_type
        self.batch_norm = batch_norm


        self.conv_init = Conv2D(nfilters, 3, padding='same', kernel_initializer=kernel_initializer, use_bias=use_bias, kernel_regularizer=l2(weight_decay))

        self.layers_c = {}
        for idx in range(nstage):
            self.layers_c['dense{:d}'.format(idx)] = DenseBlock2D(nlayers=stage_conv,
                                                                padding=padding, 
                                                                growth_rate=nfilters // stage_conv, 
                                                                dropout_type=dropout_type, 
                                                                dropout_rate=dropout_rate,
                                                                batch_norm=batch_norm, 
                                                                kernel_initializer=kernel_initializer,
                                                                use_bias=use_bias,
                                                                weight_decay=weight_decay,
                                                                name='denseBlock{:d}'.format(idx))

            self.layers_c['trans{:d}'.format(idx)] = TransitionLayer2D(compression=0.5,
                                                                        dropout_type=dropout_type, 
                                                                        dropout_rate=dropout_rate, 
                                                                        batch_norm=batch_norm,
                                                                        kernel_initializer=kernel_initializer, 
                                                                        use_bias=use_bias, 
                                                                        weight_decay=weight_decay,
                                                                        name='transitionLayer{:d}'.format(idx))
            if idx != 0 :
                self.layers_c['pool{:d}'.format(idx)] = MaxPooling2D(pool_size=(2, 2))

                self.layers_c['up{:d}'.format(idx)] = UpLayer2D(nfilters,
                                                                up_type=up_type,
                                                                up_scale=2**idx,
                                                                batch_norm=batch_norm,
                                                                kernel_initializer=kernel_initializer, 
                                                                use_bias=use_bias, 
                                                                weight_decay=weight_decay)
            P = (2 ** nstage -  2 ** (idx + 1)) * stage_conv
            
            if padding == 'valid':
                self.layers_c['feature_wrap{:d}'.format(idx)] = Cropping2D(cropping=((P, P), (P, P)))
            elif padding == 'full':
                self.layers_c['feature_wrap{:d}'.format(idx)] = ZeroPadding2D(padding=(P, P))
            else:
                self.layers_c['feature_wrap{:d}'.format(idx)] = ZeroPadding2D(padding=(0, 0))

        # merge feature map
        if self.merge_type == 'add':
            self.merge = Add()
        else:
            self.merge = Concatenate(axis=-1)

        # output conv
        if self.batch_norm:
            self.batchnorm_output = BatchNormalization()
        self.relu_output = ReLU()
        self.conv_output = Conv2D(nfilters, 1, padding='valid', kernel_initializer=kernel_initializer, use_bias=use_bias, kernel_regularizer=l2(weight_decay))
    
    def valid_input_size(self, input_size):
        is_valid = True
        input_size = np.array(input_size).astype(np.uint16)
        if self.padding  == 'same' or self.padding  == 'full':
            D = 2 ** (self.nstage)
            if np.any(input_size % D == 0):
                is_valid = False
                input_size = np.round(input_size/D)*D
        if self.padding  == 'valid':
            input_size = input_size - 2 * self.stage_conv
            for _ in range(self.nstage-1):
                remainder = input_size % 2
                input_size = input_size / 2
                if np.any(remainder != 0):
                    is_valid = False
                input_size = input_size - 2 * self.stage_conv
                if np.any(input_size < 1):
                    is_valid = False
            input_size = np.round(input_size)
            input_size[input_size<1] = 1
            for _ in range(self.nstage-1):
                input_size = input_size + 2 * self.stage_conv
                input_size = input_size * 2
            input_size = input_size + 2 * self.stage_conv
        
        if self.padding == 'same':
            output_size = input_size
        elif self.padding == 'full':
            output_size = input_size + 2 * (2 ** self.nstage - 1) * self.stage_conv
        elif self.padding == 'valid':
            output_size = input_size - 2 * (2 ** self.nstage - 1) * self.stage_conv
        return is_valid, tuple(input_size.astype(np.uint16)), tuple(output_size.astype(np.uint16))


    def call(self, inputs, training=False):

        feature_list = []

        outputs = self.conv_init(inputs)

        for idx in range(self.nstage):
            if idx != 0:
                outputs = self.layers_c['pool{:d}'.format(idx)](outputs)
            outputs = self.layers_c['dense{:d}'.format(idx)](outputs, training)
            outputs = self.layers_c['trans{:d}'.format(idx)](outputs, training)
            if idx == 0:
                outputs_wrap = outputs
            else:
                outputs_wrap = self.layers_c['up{:d}'.format(idx)](outputs, training)
            feature_list.append(self.layers_c['feature_wrap{:d}'.format(idx)](outputs_wrap))
                
        outputs = self.merge(feature_list)
        if self.batch_norm:
            outputs = self.batchnorm_output(outputs, training)
        outputs = self.relu_output(outputs)
        outputs = self.conv_output(outputs)
        
        return outputs


if __name__ == "__main__":
    import numpy as np
    from tensorflow import keras
    import os


    ############################

    model = DenseSeg2D(nfilters=16,
                       nstage=3,
                       stage_conv=4,
                       padding='same',
                       dropout_type='default',  # 'spatial', 'normal'
                       dropout_rate=0.25,
                       up_type='upConv', # 'bilinear', 'upConv' or 'deConv' 
                       merge_type='cat', # 'add', 'cat'
                       kernel_initializer='he_normal',
                       use_bias=False,
                       weight_decay=1e-4)

    is_valid, input_sz, output_sz = model.valid_input_size((128,128))
    print(is_valid, input_sz, output_sz)

    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy')
    model.build(input_shape=(1,512,512,1))
    model.summary()

    logdir="./logs_check"
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    train_images = np.zeros((4,512,512,1)).astype(np.float32)
    train_labels = np.zeros((4,512,512,1)).astype(np.int32)
    # output = model(train_images)
    # output = np.array(output)
    # print(output.shape)

    # Train the model.
    model.fit(train_images, train_labels, batch_size=1, epochs=1, 
              callbacks=[tensorboard_callback])