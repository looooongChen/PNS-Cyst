import tensorflow as tf
from tensorflow.keras.layers import UpSampling2D, Conv2D, Conv2DTranspose, Dense
from tensorflow.keras.regularizers import l2
import tempfile
import os

class EfficientNetSeg(tf.keras.Model):

    def __init__(self,
                 input_shape=(512,512),
                 version='B0',
                 filters=32,
                 up_type='deConv', # 'upConv', 'deConv'
                 merge_type='cat', # 'add', 'cat'
                 use_bias=False,
                 weight_decay=1e-5,
                 name='EfficientNetSeg',
                 **kwargs):

        super().__init__(name=name, **kwargs)

        self.up_type = up_type
        self.merge_type = merge_type

        self.conv_init = Conv2D(3, 1)

        if version == 'EfficientNetB0':
            layer_name = ['block1a_se_excite', 'block2b_add', 'block3b_add', 'block5c_add', 'block7a_se_excite']
            filters = [filters, 32, 24, 40, 112]
            net = tf.keras.applications.efficientnet.EfficientNetB0(include_top=False, weights='imagenet', input_shape=(input_shape[0], input_shape[1], 3))
        if version == 'EfficientNetB1':
            layer_name = ['block1b_add', 'block2c_add', 'block3c_add', 'block5d_add', 'block7b_se_excite']
            filters = [filters, 16, 24, 40, 112]
            net = tf.keras.applications.efficientnet.EfficientNetB1(include_top=False, weights='imagenet', input_shape=(input_shape[0], input_shape[1], 3))
        if version == 'EfficientNetB2':
            layer_name = ['block1b_add', 'block2c_add', 'block3c_add', 'block5d_add', 'block7b_add']
            filters = [filters, 16, 24, 48, 120]
            net = tf.keras.applications.efficientnet.EfficientNetB2(include_top=False, weights='imagenet', input_shape=(input_shape[0], input_shape[1], 3))
        if version == 'EfficientNetB3':
            layer_name = ['block1b_add', 'block2c_add', 'block3c_add', 'block5e_add', 'block7b_add']
            filters = [filters, 24, 32, 48, 136]
            net = tf.keras.applications.efficientnet.EfficientNetB3(include_top=False, weights='imagenet', input_shape=(input_shape[0], input_shape[1], 3))
        if version == 'EfficientNetB4':
            layer_name = ['block1b_add', 'block2d_add', 'block3d_add', 'block5f_add', 'block7b_add']
            filters = [filters, 24, 32, 56, 160]
            net = tf.keras.applications.efficientnet.EfficientNetB4(include_top=False, weights='imagenet', input_shape=(input_shape[0], input_shape[1], 3))
        if version == 'EfficientNetB5':
            layer_name = ['block1c_add', 'block2e_add', 'block3e_add', 'block5g_add', 'block7c_add']
            filters = [filters, 24, 40, 64, 176]
            net = tf.keras.applications.efficientnet.EfficientNetB5(include_top=False, weights='imagenet', input_shape=(input_shape[0], input_shape[1], 3))
        if version == 'EfficientNetB6':
            layer_name = ['block1c_add', 'block2f_add', 'block3f_add', 'block5h_add', 'block7c_add']
            filters = [filters, 32, 40, 72, 200]
            net = tf.keras.applications.efficientnet.EfficientNetB6(include_top=False, weights='imagenet', input_shape=(input_shape[0], input_shape[1], 3))
        if version == 'EfficientNetB7':
            layer_name = ['block1d_add', 'block2g_add', 'block3g_add', 'block5j_add', 'block7c_add']
            filters = [filters, 32, 48, 80, 224]
            net = tf.keras.applications.efficientnet.EfficientNetB7(include_top=False, weights='imagenet', input_shape=(input_shape[0], input_shape[1], 3))

        if weight_decay > 0:
            for layer in net.layers:
                if isinstance(layer, Conv2D):
                    layer.kernel_regularizer = l2(weight_decay)

            tmp_weights_path = os.path.join(tempfile.gettempdir(), 'efficientnet_tmp_weights.h5')
            net.save_weights(tmp_weights_path)
            net = tf.keras.models.model_from_json(net.to_json())        
            net.load_weights(tmp_weights_path, by_name=True)
        
        ft2 = net.get_layer(layer_name[0]).output # stride 1/2
        ft4 = net.get_layer(layer_name[1]).output # stride 1/4
        ft8 = net.get_layer(layer_name[2]).output # stride 1/8
        ft16 = net.get_layer(layer_name[3]).output # stride 1/16
        ft32 = net.get_layer(layer_name[4]).output # stride 1/32
        
        self.backbone = tf.keras.Model(inputs=net.inputs, outputs=[ft2, ft4, ft8, ft16, ft32])

        if weight_decay == 0:
            Conv = lambda filters: Conv2D(filters, 3, padding='same') if self.up_type == 'upConv' else Conv2DTranspose(filters, 2, 2, padding='same', use_bias=use_bias)
        else:
            Conv = lambda filters: Conv2D(filters, 3, padding='same', kernel_regularizer=l2(weight_decay)) if self.up_type == 'upConv' else Conv2DTranspose(filters, 2, 2, padding='same', use_bias=use_bias, kernel_regularizer=l2(weight_decay))
        if self.up_type == 'upConv':
            self.ft2_up = UpSampling2D(size=(2, 2), interpolation='bilinear')
            self.ft4_up = UpSampling2D(size=(2, 2), interpolation='bilinear')
            self.ft8_up = UpSampling2D(size=(2, 2), interpolation='bilinear')
            self.ft16_up = UpSampling2D(size=(2, 2), interpolation='bilinear')
            self.ft32_up = UpSampling2D(size=(2, 2), interpolation='bilinear')

            self.ft2_conv = Conv(filters[0])
            self.ft4_conv = Conv(filters[1])
            self.ft8_conv = Conv(filters[2])
            self.ft16_conv = Conv(filters[3])
            self.ft32_conv = Conv(filters[4])
        else: 
            self.ft2_up = Conv(filters[0])
            self.ft4_up = Conv(filters[1])
            self.ft8_up = Conv(filters[2])
            self.ft16_up = Conv(filters[3])
            self.ft32_up = Conv(filters[4])

    def call(self, inputs, training=False):

        inputs = self.conv_init(inputs)
        
        ft2, ft4, ft8, ft16, ft32 = self.backbone(inputs, training)
        
        ft16c = self.ft32_conv(self.ft32_up(ft32)) if self.up_type == 'upConv' else self.ft32_up(ft32)
        ft16 = ft16c + ft16 if self.merge_type == 'add' else tf.concat([ft16c, ft16], axis=-1)
        ft8c = self.ft16_conv(self.ft16_up(ft16)) if self.up_type == 'upConv' else self.ft16_up(ft16)
        ft8 = ft8c + ft8 if self.merge_type == 'add' else tf.concat([ft8c, ft8], axis=-1)
        ft4c = self.ft8_conv(self.ft8_up(ft8)) if self.up_type == 'upConv' else self.ft8_up(ft8)
        ft4 = ft4c + ft4 if self.merge_type == 'add' else tf.concat([ft4c, ft4], axis=-1)
        ft2c = self.ft4_conv(self.ft4_up(ft4)) if self.up_type == 'upConv' else self.ft4_up(ft4)
        ft2 = ft2c + ft2 if self.merge_type == 'add' else tf.concat([ft2c, ft2], axis=-1)
        ft = self.ft2_conv(self.ft2_up(ft2)) if self.up_type == 'upConv' else self.ft2_up(ft2)

        return ft


if __name__ == "__main__":
    import numpy as np

    # model = tf.keras.applications.efficientnet.EfficientNetB7(include_top=False, weights='imagenet', input_shape=(256,256,3))
    # tf.keras.utils.plot_model(model, to_file='efficientNetB7.png', show_shapes=True, show_layer_names=True, dpi=96)
    # model.summary()

    model = EfficientNetSeg(input_shape=(512,512),
                            filters=32,
                            version='EfficientNetB0',
                            up_type='deConv', # 'upConv', 'deConv'
                            merge_type='cat', # 'add', 'cat'
                            weight_decay=1e-5,
                            use_bias=False,)
    
    inputs = np.ones((1,512,512,1))
    outputs = model(inputs)
    print(outputs.shape)

