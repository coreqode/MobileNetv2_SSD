import os
import torch
import torch.nn as nn
from get_priors import get_priors

def detect(pred_localization, pred_confidence, min_score, max_overlap, top_k):
    batch_size = pred_localization.size(0)
    priors = get_priors()
    assert priors.size(0) == pred_localization.size(1) == pred_confidence.size(1)
    ## since predictions are in (batch x priors x 4) for localization and (batch x priors x class) for confidence
    for i in range(batch_size):
        ## finding offsets of prediction with actual priors
        #TODO: can multiply with empirical number 10 and 5 after seeing the results as mentioned in the caffe repo.
        offset_x, offset_y = (pred_localization[i,:,:2] - priors[:,:2]) / priors[:,2:]
        offset_w, offset_h = torch.log(pred_localization[i,:,2:] / priors[:,2:]) 
        offsets = torch.cat([offset_x, offset_y, offset_w, offset_h], dim = 1)

        ## converting the cx,cy,w,h format to xmin,ymin,xmax,ymax format
        xmin, ymin = offset_x[]


if __name__ == "__main__":
    priors = get_priors()
    print(priors[:,:2].size())
