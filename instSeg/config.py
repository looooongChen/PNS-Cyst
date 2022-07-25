# construct the adjacent matrix using MAX_OBJ, 
# if the object number in your cases is large, increase it correspondingly 
import pickle
from instSeg.enumDef import *

MAX_OBJ = 500

class Config(object):

    def __init__(self, image_channel=3):

        self.verbose = False

        self.model_type = MODEL_BASE
        self.save_best_metric = 'AJI' # 'AJI'/'AP'/
        # input size
        self.H = 512
        self.W = 512
        self.image_channel = image_channel
        
        # backbone config
        self.backbone = 'uNet' # 'uNet', 'ResNet50', 'ResNet101', 'ResNet152', 'EfficientNetB0' - 'EfficientNetB7'
        self.filters = 64
        ## config for specific to unet
        self.weight_decay = 1e-5
        self.nstage = 5
        self.stage_conv = 2
        self.padding = 'same'
        self.residual = False
        self.dropout_rate = 0
        self.batch_norm = False
        self.net_upsample = 'deConv' # 'upConv', 'deConv'
        self.net_merge = 'cat' # 'add', 'cat'

        # losses
        self.focal_loss_gamma = 2.0
        self.sensitivity_specificity_loss_beta = 1.0
        ## semantic module
        self.classes = 2
        self.semantic_loss = 'dice'
        self.semantic_weight = 1
        self.semantic_in_ram = False
        ## contour loss
        self.contour_loss = 'focal_loss'
        self.contour_weight = 1
        self.contour_radius = 2 
        self.contour_in_ram = False
        ## euclidean dist transform regression
        self.edt_loss = 'mse'
        self.edt_weight = 1
        self.edt_normalize = True
        self.edt_in_ram = False 
        ## flow regression
        self.flow_loss = 'masked_mse'
        self.flow_weight = 1
        self.flow_in_ram = False
        self.flow_mode = 'offset'
        ## embedding loss
        self.embedding_dim = 8
        self.positional_embedding = None # 'global', 'harmonic'
        self.octave = 4
        self.embedding_loss = 'cos'
        self.embedding_l1_weight = 0.01 
        self.embedding_weight = 1
        self.embedding_include_bg = True
        self.neighbor_distance = 10
        self.max_obj = MAX_OBJ

        # data augmentation
        self.flip = False
        self.elastic_deform = False
        self.elastic_strength = 200
        self.elastic_scale = 10
        self.random_rotation = False
        self.random_crop = False
        self.random_crop_range = (0.6, 0.8)
        self.random_gamma = False
        self.random_gamma_range = (0.5, 2)
        self.blur = False
        self.blur_gamma = 2

        # training config:
        self.train_epochs = 100
        self.train_batch_size = 3
        self.train_learning_rate = 1e-4
        self.lr_decay_period = 10000
        self.lr_decay_rate = 0.9

        # validation 
        self.validation_start_epoch = 1

        # post-process
        # object filtering
        self.obj_min_edt = 2
        self.obj_min_size=0
        self.obj_max_size=float('inf')

        # dcan
        self.dcan_thres_contour=0.5
        # embedding
        self.emb_cluster_thres=0.7
        self.emb_cluster_max_step=float('inf')
        # distance regression map
        self.edt_instance_thres = 5
        self.edt_fg_thres = 3
        # flow tracking
        self.flow_tracking_iters = 100
        # self.flow_stop = 0.7 if self.flow_mode == 'inwards' else 1
        self.flow_stop = 1
        # semantic
        self.semantic_bg = 0
        
    
    def save(self, path):
        if path.endswith('.pkl'):
            with open(path, 'wb') as output:
                pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)


# class ConfigContour(Config):

#     def __init__(self, image_channel=3):
#         super().__init__(image_channel=image_channel)

class ConfigCascade(Config):

    def __init__(self, image_channel=3):
        super().__init__(image_channel=image_channel)
        self.input_normalization = None # 'per-image', 'constant', None
        self.input_normalization_bias = 0
        self.input_normalization_scale = 1
        self.model_type = MODEL_CASCADE
        self.modules = ['semantic', 'edt', 'embedding']
        # config feature forward
        self.feature_forward_dimension = 32
        self.stop_gradient = True

class ConfigParallel(Config):

    def __init__(self, image_channel=3):
        super().__init__(image_channel=image_channel)
        self.input_normalization = None # 'per-image', 'constant', None
        self.input_normalization_bias = 0
        self.input_normalization_scale = 1
        self.model_type = MODEL_PARALLEL
        self.modules = ['semantic', 'edt']

