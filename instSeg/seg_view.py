import numpy as np
from skimage.measure import regionprops, label
from instSeg.stitcher import *
from skimage.morphology import square
import copy
import os
import cv2


def seg_in_tessellation(model, img, patch_sz=[512,512], margin=[64,64], overlap=[64,64], mode='lst', verbose=False):

    '''
    Args:
        model: model for prediction, easiest way to load model: instSeg.load_model(model_dir)
        img: input image
        patch_sz: size of patches to split
        margin: parts to ignore, since predictions at the image surrounding area tend to not be as good as the centeral part
            2*margin should less than image size
        overlap: size of overlapping area
            2*margin + overlap should also less than image size
        mode: 'lst' or 'bi'
            'lst' (label stitch) model: instances are obtained in patch level, then stitched. An overlap is necessray to match intances from neighbouring patches
            'bi' (binary instance) model: we can stitch binary semantic segmentation and boundary map and then get instances
    '''

    meta = split(img, patch_sz, (overlap[0]+2*margin[0], overlap[1]+2*margin[1]), patch_in_ram=True, save_dir=None)

    if mode == 'lst':
        if overlap[0] == 0 or overlap[1]==0:
            print("WARNING: overlap is necessary for label map stitching!!!")
        for idx, patch in enumerate(meta['patches']):
            if verbose:
                print('processing: patch ', idx)
            p = model.predict(patch['data'], keep_size=True)[0]
            patch['data'] = p[margin[0]:p.shape[0]-margin[0], margin[1]:p.shape[1]-margin[1]]
            patch['position'] = [patch['position'][0]+margin[0], patch['position'][1]+margin[1]]
            patch['size'] = [patch['size'][0]-2*margin[0], patch['size'][1]-2*margin[1]]
        instances = stitch(meta, channel=1, mode='label')
        return instances
    
    if mode == 'bi':

        assert 'semantic' in model.config.modules and 'contour' in model.config.modules
        # only for binary semantic segmentation
        meta_contour = copy.deepcopy(meta)
        for idx, patch in enumerate(meta['patches']):
            if verbose:
                print('processing: patch ', idx)
            p = model.predict_raw(patch['data'], keep_size=True)
            # semantic map
            semantic = p['semantic'][...,1].copy()
            patch['data'] = semantic[margin[0]:semantic.shape[0]-margin[0], margin[1]:semantic.shape[1]-margin[1]]
            patch['position'] = [patch['position'][0]+margin[0], patch['position'][1]+margin[1]]
            patch['size'] = [patch['size'][0]-2*margin[0], patch['size'][1]-2*margin[1]]
            # contour map
            contour = p['contour'].copy()
            meta_contour['patches'][idx]['data'] = contour[margin[0]:semantic.shape[0]-margin[0], margin[1]:semantic.shape[1]-margin[1]]
            meta_contour['patches'][idx]['position'] = patch['position']
            meta_contour['patches'][idx]['size'] = patch['size']
            
        semantic = stitch(meta, channel=1, mode='average') > 0.5
        contour = stitch(meta_contour, channel=1, mode='max') > 0.5

        contour = cv2.dilate(contour.astype(np.uint8), square(3), iterations = 1)

        instances = semantic * (contour == 0)
        instances = label(instances).astype(np.uint16)
        while True:
            pixel_add = cv2.dilate(instances, square(3), iterations = 1) * (instances == 0) * semantic
            if np.sum(pixel_add) != 0:
                instances += pixel_add
            else:
                break

        return instances










