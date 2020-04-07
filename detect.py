import os
import torch
import torch.nn as nn
from get_priors import get_priors

def detect(pred_localization, pred_confidence, priors,  min_score, max_overlap, top_k):
    """
    pred_localization: bbox predictions in shape of (batch x priors x 4)
    pred_confidence: confidence score predictions in shape of (batch x priors x classes)
    priors: Actual priors boxes in shape of (priors x 4)
    """
    batch_size = pred_localization.size(0)
    priors = priors.unsqueeze(0)  ## (1 x priors x 4)
    priors = priors.repeat(batch_size, 1, 1) ##To repeat the same value along dimension of batch size
    assert priors.size(0) == pred_localization.size(1) == pred_confidence.size(1) == priors.size(0)
    ## since predictions are in (batch x priors x 4) for localization and (batch x priors x class) for confidence
    for i in range(batch_size):
        ## finding offsets of prediction with actual priors
        #TODO: can multiply with empirical number 10 and 5 after seeing the results as mentioned in the caffe repo.
        offset_x, offset_y = (pred_localization[i,:,:2] - priors[:,:2]) / priors[:,2:]
        offset_w, offset_h = torch.log(pred_localization[i,:,2:] / priors[:,2:]) 
        offsets = torch.cat([offset_x, offset_y, offset_w, offset_h], dim = 1)

        ## converting the cx,cy,w,h format to xmin,ymin,xmax,ymax format
        xmin, ymin = offsets[:,:2] - offsets[:,2:] / 2
        xmax, ymax = offsets[:,:2] - offsets[:,2:] / 2



if __name__ == "__main__":
    priors = get_priors()
    print(priors.size())
