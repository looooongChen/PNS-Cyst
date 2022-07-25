from instSeg.networks import UNet, EfficientNetSeg, ResNetSeg

def backboneFactory(config):

    arch = config.backbone
    
    assert arch in ['uNet', 'ResNet50', 'ResNet101', 'ResNet152', 'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7']

    if arch.lower() == 'unet':
        arch_func = lambda name: UNet(nfilters=config.filters,
                                      nstage=config.nstage,
                                      stage_conv=config.stage_conv,
                                      residual=config.residual,
                                      dropout_rate=config.dropout_rate,
                                      batch_norm=config.batch_norm,
                                      up_type=config.net_upsample, 
                                      merge_type=config.net_merge, 
                                      weight_decay=config.weight_decay,
                                      name=name)
    elif arch.startswith('ResNet'):
        arch_func = lambda name: ResNetSeg(input_shape=(config.H, config.W),
                                           version=arch,
                                           filters=config.filters,
                                           up_type=config.net_upsample,
                                           merge_type=config.net_merge,
                                           weight_decay=config.weight_decay,
                                           name=name)
    elif arch.startswith('EfficientNet'):
        arch_func = lambda name: EfficientNetSeg(input_shape=(config.H, config.W),
                                                 version=arch,
                                                 filters=config.filters,
                                                 up_type=config.net_upsample,
                                                 merge_type=config.net_merge,
                                                 weight_decay=config.weight_decay,
                                                 name=name)
    
    return arch_func