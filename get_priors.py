import os 
import torch
import numpy as np 
from itertools import product
from numpy import sqrt
import matplotlib.pyplot as plt
import cv2


def get_priors():
    """
    1. Paper uses the formula to calculate the scale of different boxes for each feature map.
       sk = s(min) + ((s(max) - s(min)) / (m-1)) * (k-1), where s(min) = 0.2 and s(max) = 0.9
    2. Also for 1 aspect ratio, he uses the scale, sk(new) = sqrt(sk * s(k+1))
    3. For the calculation of width and height of default boxes, width = sk * sqrt(aspect_ratio) 
       and height = sk / sqrt(aspect_ratio) 
    """
    feature_maps = {'x_1':19, 'x_2':10, 'x_3':5, 'x_4':3, 'x_5':2, 'x_6':1}
    scales = {'x_1':0.2, 'x_2':0.34, 'x_3':0.48, 'x_4':0.62, 'x_5':0.76, 'x_6':0.9}
    aspect_ratios = {'x_1':[1., 2., 0.5], 'x_2':[1., 2., 3., 0.5, .333], 'x_3':[1., 2., 3., 0.5, .333], 
                    'x_4':[1., 2., 3., 0.5, .333], 'x_5':[1., 2., 0.5], 'x_6':[1., 2., 0.5]}
    keys = list(feature_maps.keys())
    priors = []
    for idx, key in enumerate(feature_maps):
        for i, j in product(range(feature_maps[key]), repeat = 2):
            center_x = (j + 0.5) / feature_maps[key] ## Adding 0.5 to get the center of pixel from (i,j)
            center_y = (i + 0.5) / feature_maps[key]
            for ratio in aspect_ratios[key]:
                width = scales[key] * sqrt(ratio)
                height = scales[key] / sqrt(ratio)
                priors.append([center_x, center_y, width, height ])
                ## If ratio = 1, then we have to calculate another prior box
                if ratio == 1:
                    try:
                        new_scale = sqrt(scales[keys[idx]] * scales[keys[idx+1]])
                    except:
                        new_scale = 1
                    priors.append([center_x, center_y, new_scale, new_scale]) ## as ratio is 1
    priors = torch.FloatTensor(priors)
    priors.clamp(0,1)
    return priors

if __name__ == "__main__":
    priors = get_priors()