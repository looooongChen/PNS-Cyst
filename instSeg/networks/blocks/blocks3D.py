from sys import modules
import tensorflow as tf
from tensorflow.keras.layers import Conv3D, ReLU, BatchNormalization, Dropout, Concatenate, SpatialDropout3D, UpSampling3D, Conv3DTranspose, ZeroPadding3D, Cropping3D
from tensorflow.keras.regularizers import l2


class DenseBlock3D(tf.keras.layers.Layer):

    def __init__(self, 
                 nlayers=3, 
                 growth_rate=8, 
                 padding='same', # 'same' 'valid', 'full'
                 dropout_type='spatial', 
                 dropout_rate=None,
                 batch_norm=False, 
                 kernel_initializer='he_normal',
                 use_bias=False,
                 weight_decay=1e-4,
                 name='denseBlock3D',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        assert dropout_type in ['spatial', 'default']

        self.nlayers = nlayers
        self.growth_rate = growth_rate
        self.padding = padding
        self.dropout_type = dropout_type
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.kernel_initializer = kernel_initializer
        self.use_bias = use_bias
        self.weight_decay = weight_decay
        self.layers_c = {}

        for idx in range(self.nlayers):
            if self.dropout_rate:
                if self.dropout_type == 'spatial':
                    self.layers_c['dropout_{:d}'.format(idx)] = SpatialDropout3D(self.dropout_rate)
                else:
                    self.layers_c['dropout_{:d}'.format(idx)] = Dropout(self.dropout_rate)
            if self.batch_norm:
                self.layers_c['batchnorm_{:d}'.format(idx)] = BatchNormalization()
            self.layers_c['relu_{:d}'.format(idx)] = ReLU()
            self.layers_c['conv_{:d}'.format(idx)] = Conv3D(self.growth_rate, 3, padding='valid', kernel_initializer=self.kernel_initializer, use_bias=self.use_bias, kernel_regularizer=l2(self.weight_decay))
            if self.padding == 'valid':
                self.layers_c['input_wrap{:d}'.format(idx)] = ZeroPadding3D(padding=(0, 0, 0))
                self.layers_c['feature_wrap{:d}'.format(idx)] = Cropping3D(cropping=((1, 1), (1, 1), (1, 1)))
            elif self.padding == 'full':
                self.layers_c['input_wrap{:d}'.format(idx)] = ZeroPadding3D(padding=(2, 2, 2))
                self.layers_c['feature_wrap{:d}'.format(idx)] = ZeroPadding3D(padding=(1, 1, 1))
            else:
                self.layers_c['input_wrap{:d}'.format(idx)] = ZeroPadding3D(padding=(1, 1, 1))
                self.layers_c['feature_wrap{:d}'.format(idx)] = ZeroPadding3D(padding=(0, 0, 0))
            self.layers_c['cat_{:d}'.format(idx)] = Concatenate(axis=-1)

        self.layers_c['cat'] = Concatenate(axis=-1)

    def call(self, inputs, training=False):
        for idx in range(self.nlayers):
            outputs = self.layers_c['input_wrap{:d}'.format(idx)](inputs)
            if self.batch_norm:
                outputs = self.layers_c['batchnorm_{:d}'.format(idx)](outputs, training)
            outputs = self.layers_c['relu_{:d}'.format(idx)](outputs)
            if self.dropout_rate:
                outputs = self.layers_c['dropout_{:d}'.format(idx)](outputs, training)
            outputs = self.layers_c['conv_{:d}'.format(idx)](outputs)
            inputs = self.layers_c['feature_wrap{:d}'.format(idx)](inputs)
            inputs = self.layers_c['cat_{:d}'.format(idx)]([outputs, inputs])
        return inputs

class TransitionLayer3D(tf.keras.layers.Layer):

    def __init__(self,
                 compression=1.0,
                 dropout_type='spatial', 
                 dropout_rate=None, 
                 batch_norm = False,
                 kernel_initializer='he_normal', 
                 use_bias=False, 
                 weight_decay=1e-4,
                 name='transitionLayer3D',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        assert dropout_type in ['spatial', 'default']

        self.compression = compression
        self.dropout_type = dropout_type
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.kernel_initializer = kernel_initializer
        self.use_bias = use_bias
        self.weight_decay = weight_decay

        if self.batch_norm:
            self.batchnorm = BatchNormalization()
        self.relu = ReLU()
        if self.dropout_rate:
            if self.dropout_type == 'spatial':
                self.drop = SpatialDropout3D(self.dropout_rate)
            else:
                self.drop = Dropout(self.dropout_rate)

    def build(self, input_shape):
        filters = int(input_shape[-1] * self.compression)
        self.conv = Conv3D(filters, 1, padding='valid', kernel_initializer=self.kernel_initializer, use_bias=self.use_bias, kernel_regularizer=l2(self.weight_decay))

    def call(self, inputs, training=False):
        if self.batch_norm:
            outputs = self.batchnorm(inputs, training)
        else:
            outputs = inputs
        outputs = self.relu(outputs)
        if self.dropout_rate:
            outputs = self.drop(outputs, training)
        outputs = self.conv(outputs)
        return outputs 


class UpLayer3D(tf.keras.layers.Layer):

    def __init__(self,
                 nfilters,
                 up_type='upConv', # 'bilinear', 'upConv' or 'deConv' 
                 up_scale=2,
                 batch_norm=False,
                 kernel_initializer='he_normal', 
                 use_bias=False, 
                 weight_decay=1e-4,
                 name='upLayer3D',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        assert up_type in ['bilinear', 'upConv', 'deConv']

        self.nfilters = nfilters
        self.up_type = up_type
        self.up_scale = up_scale
        self.kernel_initializer = kernel_initializer
        self.use_bias = use_bias
        self.weight_decay = weight_decay

        if self.up_type == 'bilinear':
            self.up = UpSampling3D(size=(self.up_scale, self.up_scale, self.up_scale))
        else:
            if self.batch_norm:
                self.batchnorm = BatchNormalization()
            self.relu = ReLU()
            if self.up_type == 'upConv':
                self.up = UpSampling3D(size=(self.up_scale, self.up_scale, self.up_scale))
                self.conv = Conv3D(self.nfilters, 2*self.up_scale-1, padding='same', kernel_initializer=self.kernel_initializer, use_bias=self.use_bias, kernel_regularizer=l2(self.weight_decay))
            else:
                self.up = Conv3DTranspose(self.nfilters, 2*self.up_scale, strides=self.up_scale, padding='same', kernel_initializer=self.kernel_initializer, use_bias=self.use_bias, kernel_regularizer=l2(self.weight_decay))

    def call(self, inputs, training=False):
        if self.up_type == 'bilinear':
            outputs = self.up(inputs)
        else:
            if self.batch_norm:
                outputs = self.batchnorm(inputs, training)
            else:
                outpus = inputs
            outputs = self.relu(outputs)
            outputs = self.up(outputs)
            if self.up_type == 'upConv':
                outputs = self.conv(outputs)

        return outputs 


if __name__ == "__main__":
    import numpy as np
    from tensorflow import keras
    import os

    # layer = DenseBlock3D(nlayers=3, 
    #                     growth_rate=8, 
    #                     padding='same', # 'same' 'valid', 'full'
    #                     dropout_type='spatial', 
    #                     dropout_rate=None, 
    #                     kernel_initializer='he_normal',
    #                     use_bias=False,
    #                     weight_decay=1e-4,
    #                     name='denseBlock3D')


    # layer = TransitionLayer3D(compression=0.5,
    #                         dropout_type='spatial', 
    #                         dropout_rate=0.1, 
    #                         kernel_initializer='he_normal', 
    #                         use_bias=False, 
    #                         weight_decay=1e-4)

    layer = UpLayer3D(nfilters=24,
                    up_type='upConv', # 'bilinear', 'upConv' or 'deConv' 
                    up_scale=2,
                    kernel_initializer='he_normal', 
                    use_bias=False, 
                    weight_decay=1e-4,
                    name='upLayer3D')


    inputs = tf.keras.Input(shape=(128,128,64,24))
    outputs = layer(inputs)
    outputs = Conv3D(2, 1)(outputs)
    model = keras.Model(inputs, outputs)

    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy')
    model.build(input_shape=(128,128,64,24))
    model.summary()

    logdir="./logs_check"
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    train_images = np.zeros((2,128,128,64,24)).astype(np.float32)
    # train_labels = np.zeros((2,128,128,64,1)).astype(np.int32)
    train_labels = np.zeros((2,256,256,128,1)).astype(np.int32)

    out = model(train_images)
    out = np.array(out)
    print(out.shape)

    # Train the model.
    model.fit(train_images, train_labels, batch_size=1, epochs=1, 
              callbacks=[tensorboard_callback])