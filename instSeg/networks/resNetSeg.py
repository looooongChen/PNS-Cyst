import tensorflow as tf
from tensorflow.keras.layers import UpSampling2D, Conv2D, Conv2DTranspose
from tensorflow.keras.regularizers import l2
import tempfile
import os

class ResNetSeg(tf.keras.Model):

    def __init__(self,
                 input_shape=(512,512),
                 version='resnet50',
                 filters=32,
                 up_type='deConv', # 'upConv', 'deConv'
                 merge_type='cat', # 'add', 'cat'
                 use_bias=False,
                 weight_decay=1e-5,
                 name='ResNetSeg',
                 **kwargs):

        super().__init__(name=name, **kwargs)

        self.up_type = up_type
        self.merge_type = merge_type

        self.conv_init = Conv2D(3, 1)

        if version == 'ResNet50':
            net = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet', input_shape=(input_shape[0], input_shape[1], 3))
        if version == 'ResNet101':
            net = tf.keras.applications.ResNet101V2(include_top=False, weights='imagenet', input_shape=(input_shape[0], input_shape[1], 3))
        if version == 'ResNet152':
            net = tf.keras.applications.ResNet152V2(include_top=False, weights='imagenet', input_shape=(input_shape[0], input_shape[1], 3))

        if weight_decay > 0:
            for layer in net.layers:
                if isinstance(layer, Conv2D):
                    layer.kernel_regularizer = l2(weight_decay)

            tmp_weights_path = os.path.join(tempfile.gettempdir(), 'resnet_tmp_weights.h5')
            net.save_weights(tmp_weights_path)
            net = tf.keras.models.model_from_json(net.to_json())        
            net.load_weights(tmp_weights_path, by_name=True)

        ft2 = net.get_layer('conv1_conv').output
        ft4 = net.get_layer('conv2_block2_out').output # stride 1/4
        ft8 = net.get_layer('conv3_block3_out').output # stride 1/8
        ft16 = net.get_layer('conv4_block5_out').output
        ft32 = net.get_layer('conv5_block3_out').output
        
        self.backbone = tf.keras.Model(inputs=net.inputs,
                                       outputs=[ft2, ft4, ft8, ft16, ft32])

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

            self.ft2_conv = Conv(32)
            self.ft4_conv = Conv(64)
            self.ft8_conv = Conv(256)
            self.ft16_conv = Conv(512)
            self.ft32_conv = Conv(1024)
        else: 
            self.ft2_up = Conv(32)
            self.ft4_up = Conv(64)
            self.ft8_up = Conv(256)
            self.ft16_up = Conv(512)
            self.ft32_up = Conv(1024)

        
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

    # model = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet', input_shape=(256,256,3))
    # tf.keras.utils.plot_model(model, to_file='resnet50.png', show_shapes=True, show_layer_names=True, dpi=96)
    # model.summary()

    model = ResNetSeg(input_shape=(512,512),
                        filters=32,
                        up_type='deConv', # 'upConv', 'deConv'
                        merge_type='cat', # 'add', 'cat'
                        weight_decay=1e-5,
                        use_bias=False,
                        version='ResNet50')
    
    inputs = np.ones((1,512,512,1))
    outputs = model(inputs)
    print(outputs.shape)
    print(np.sum(model.losses))

