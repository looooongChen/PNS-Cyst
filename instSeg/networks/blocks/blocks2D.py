import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ReLU, BatchNormalization, Dropout, Concatenate, SpatialDropout2D, UpSampling2D, Conv2DTranspose, ZeroPadding2D, Cropping2D
from tensorflow.keras.regularizers import l2


class DenseBlock2D(tf.keras.layers.Layer):

    def __init__(self, 
                 nlayers=4, 
                 growth_rate=16, 
                 padding='same', # 'same' 'valid', 'full'
                 batch_norm=False,
                 dropout_type='default', 
                 dropout_rate=None, 
                 kernel_initializer='he_normal',
                 use_bias=False,
                 weight_decay=1e-4,
                 name='denseBlock2D',
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
                    self.layers_c['dropout_{:d}'.format(idx)] = SpatialDropout2D(self.dropout_rate)
                else:
                    self.layers_c['dropout_{:d}'.format(idx)] = Dropout(self.dropout_rate)
            if self.batch_norm:
                self.layers_c['batchnorm_{:d}'.format(idx)] = BatchNormalization()
            self.layers_c['relu_{:d}'.format(idx)] = ReLU()
            self.layers_c['conv_{:d}'.format(idx)] = Conv2D(self.growth_rate, 3, padding='valid', kernel_initializer=self.kernel_initializer, use_bias=self.use_bias, kernel_regularizer=l2(self.weight_decay))
            if self.padding == 'valid':
                self.layers_c['input_wrap{:d}'.format(idx)] = ZeroPadding2D(padding=(0, 0))
                self.layers_c['feature_wrap{:d}'.format(idx)] = Cropping2D(cropping=((1, 1), (1, 1)))
            elif self.padding == 'full':
                self.layers_c['input_wrap{:d}'.format(idx)] = ZeroPadding2D(padding=(2, 2))
                self.layers_c['feature_wrap{:d}'.format(idx)] = ZeroPadding2D(padding=(1, 1))
            else:
                self.layers_c['input_wrap{:d}'.format(idx)] = ZeroPadding2D(padding=(1, 1))
                self.layers_c['feature_wrap{:d}'.format(idx)] = ZeroPadding2D(padding=(0, 0))
            self.layers_c['cat_{:d}'.format(idx)] = Concatenate(axis=-1)


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


class TransitionLayer2D(tf.keras.layers.Layer):

    def __init__(self,
                 compression=0.5,
                 dropout_type='default', 
                 dropout_rate=None, 
                 batch_norm=False,
                 kernel_initializer='he_normal', 
                 use_bias=False, 
                 weight_decay=1e-4,
                 name='transitionLayer2D',
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
                self.drop = SpatialDropout2D(self.dropout_rate)
            else:
                self.drop = Dropout(self.dropout_rate)

    def build(self, input_shape):
        filters = int(input_shape[-1] * self.compression)
        self.conv = Conv2D(filters, 1, padding='valid', kernel_initializer=self.kernel_initializer, use_bias=self.use_bias, kernel_regularizer=l2(self.weight_decay))

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


class UpLayer2D(tf.keras.layers.Layer):

    def __init__(self,
                 nfilters,
                 up_type='upConv', # 'bilinear', 'upConv' or 'deConv' 
                 up_scale=2,
                 batch_norm=False,
                 kernel_initializer='he_normal', 
                 use_bias=False, 
                 weight_decay=1e-4,
                 name='upLayer2D',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        assert up_type in ['bilinear', 'upConv', 'deConv']

        self.nfilters = nfilters
        self.up_type = up_type
        self.up_scale = up_scale
        self.batch_norm = batch_norm
        self.kernel_initializer = kernel_initializer
        self.use_bias = use_bias
        self.weight_decay = weight_decay

        if self.up_type == 'bilinear':
            self.up = UpSampling2D(size=(self.up_scale, self.up_scale), interpolation='bilinear')
        else:
            if self.batch_norm:
                self.batchnorm = BatchNormalization()
            self.relu = ReLU()
            if self.up_type == 'upConv':
                self.up = UpSampling2D(size=(self.up_scale, self.up_scale), interpolation='bilinear')
                self.conv = Conv2D(self.nfilters, 2*self.up_scale-1, padding='same', kernel_initializer=self.kernel_initializer, use_bias=self.use_bias, kernel_regularizer=l2(self.weight_decay))
            else:
                self.up = Conv2DTranspose(self.nfilters, 2*self.up_scale, strides=self.up_scale, padding='same', kernel_initializer=self.kernel_initializer, use_bias=self.use_bias, kernel_regularizer=l2(self.weight_decay))

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

    # layer = DenseBlock2D(nlayers=4, 
    #                     padding='full',
    #                     growth_rate=16, 
    #                     dropout_rate=0.2, 
    #                     weight_decay=1e-4,
    #                     kernel_initializer='he_normal',
    #                     use_bias=False)


    # layer = TransitionLayer2D(compression=0.5,
    #                         dropout_type='spatial', 
    #                         dropout_rate=0.1, 
    #                         kernel_initializer='he_normal', 
    #                         use_bias=False, 
    #                         weight_decay=1e-4)

    layer = UpLayer2D(nfilters=8,
                        up_scale=4,
                        up_type='deConv')


    inputs = tf.keras.Input(shape=(512,512,16))
    outputs = layer(inputs)
    model = keras.Model(inputs, outputs)

    # model.compile(optimizer='adam',loss='sparse_categorical_crossentropy')
    # model.build(input_shape=(1,512,512,1))
    # model.summary()

    # logdir="./logs_check"
    # tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    train_images = np.zeros((4,512,512,16)).astype(np.float32)
    # train_labels = np.zeros((4,512,512,1)).astype(np.int32)
    
    # # train_images = np.zeros((4,256,256,1)).astype(np.float32)
    # # train_labels = np.zeros((4,512,512,1)).astype(np.int32)

    out = model(train_images)
    out = np.array(out)
    print(out.shape)

    # # Train the model.
    # model.fit(train_images, train_labels, batch_size=1, epochs=1, 
    #           callbacks=[tensorboard_callback])