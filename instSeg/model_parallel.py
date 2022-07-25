# import tensorflow as tf 
from tensorflow import keras
import tensorflow.keras.backend as K
from instSeg.model_base import InstSegMul 
from instSeg.enumDef import *
from instSeg.utils import *
from instSeg.layers import *
from instSeg.backbone import backboneFactory
import tensorflow as tf
import os


class InstSegParallel(InstSegMul):

    def __init__(self, config, model_dir='./'):
        super().__init__(config, model_dir)
        self.config.model_type = MODEL_PARALLEL
        # assert len(config.modules) == 2


    def build_model(self):

        if self.config.backbone.startswith('resnet'):
            assert self.config.image_channel == 3

        self.input_img = keras.layers.Input((self.config.H, self.config.W, self.config.image_channel), name='input_img')

        if self.config.input_normalization == 'per-image':
            self.normalized_img = tf.image.per_image_standardization(self.input_img)
        elif self.config.input_normalization == 'constant':
            self.normalized_img = (self.input_img - self.config.input_normalization_bias)/self.config.input_normalization_scale
        else:
            self.normalized_img = self.input_img

        if self.config.positional_embedding is True:
            self.normalized_img = PosConv(filters=self.config.filters)(self.normalized_img)

        output_list = []

        self.net = backboneFactory(self.config)(self.config.backbone)

        features = self.net(self.normalized_img)

        output_list = []
        for idx, m in enumerate(self.config.modules):
            if m == 'semantic':
                outlayer = keras.layers.Conv2D(filters=self.config.classes, kernel_size=1, activation='softmax')
            if m == 'contour':
                outlayer = keras.layers.Conv2D(filters=1, kernel_size=1, activation='sigmoid')
            if m == 'edt':
                activation = 'sigmoid' if self.config.edt_loss == 'binary_crossentropy' else 'linear'
                outlayer = keras.layers.Conv2D(filters=1, kernel_size=1, activation=activation)
            if m == 'flow':
                outlayer = keras.layers.Conv2D(filters=2, kernel_size=1, activation='linear')
            if m == 'embedding':
                outlayer = keras.layers.Conv2D(filters=self.config.embedding_dim, kernel_size=1, activation='linear')
            output_list.append(outlayer(features))     
                    
        self.model = keras.Model(inputs=self.input_img, outputs=output_list)
        
        if self.config.verbose:
            self.model.summary()