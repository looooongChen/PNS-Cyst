# import tensorflow as tf 
from tensorflow import keras
import tensorflow.keras.backend as K 
from instSeg.model_base import InstSegMul 
from instSeg.enumDef import *
from instSeg.layers import  *
from instSeg.backbone import backboneFactory
from instSeg.utils import *
import os

class InstSegCascade(InstSegMul):

    def __init__(self, config, model_dir='./'):
        super().__init__(config, model_dir)
        self.config.model_type = MODEL_CASCADE

    def build_model(self):
        self.input_img = keras.layers.Input((self.config.H, self.config.W, self.config.image_channel), name='input_img')
        
        if self.config.input_normalization == 'per-image':
            self.normalized_img = tf.image.per_image_standardization(self.input_img)
        elif self.config.input_normalization == 'constant':
            self.normalized_img = (self.input_img - self.config.input_normalization_bias)/self.config.input_normalization_scale
        else:
            self.normalized_img = self.input_img

        if self.config.positional_embedding is True:
            self.normalized_img = PosConv(filters=self.config.filters)(self.normalized_img)

        backbone_func = backboneFactory(self.config)
        
        output_list = []
        for i, m in enumerate(self.config.modules):
            if i != 0:   
                feature_suppression = tf.keras.layers.Conv2D(self.config.feature_forward_dimension, 1, padding='same', 
                                                             activation='linear', kernel_initializer='he_normal', 
                                                             name='feature_'+m)
                if self.config.stop_gradient:
                    features = tf.stop_gradient(tf.identity(features))
                features = feature_suppression(features)
                input_list = [self.normalized_img, tf.nn.l2_normalize(features, axis=-1)]
            else:
                input_list = [self.normalized_img]

            backbone = backbone_func(name='net_'+m)
            features = backbone(K.concatenate(input_list, axis=-1))

            if m == 'semantic':
                outlayer = keras.layers.Conv2D(filters=self.config.classes, kernel_size=1, activation='softmax')
                output_list.append(outlayer(features))
            if m == 'contour':
                outlayer = keras.layers.Conv2D(filters=1, kernel_size=1, activation='sigmoid')
                output_list.append(outlayer(features))
            if m == 'edt':
                activation = 'sigmoid' if self.config.edt_loss == 'binary_crossentropy' else 'linear'
                outlayer = keras.layers.Conv2D(filters=1, kernel_size=1, activation=activation)
                output_list.append(outlayer(features))
            if m == 'flow':
                outlayer = keras.layers.Conv2D(filters=2, kernel_size=1, activation='linear')
                output_list.append(outlayer(features))
            if m == 'embedding':
                outlayer = keras.layers.Conv2D(filters=self.config.embedding_dim, kernel_size=1, activation='linear')
                output_list.append(outlayer(features))     
                    
        self.model = keras.Model(inputs=self.input_img, outputs=output_list)
        
        if self.config.verbose:
            self.model.summary()
    


