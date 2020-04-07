import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from MobileNetv2 import Mobilenetv2_base
from itertools import product
from get_priors import get_priors

class Auxillary_Convolutions(nn.Module):
    def __init__( self):
        super(Auxillary_Convolutions,self).__init__()  ## input will be 10 x 10
        self.conv_1 = nn.Conv2d(160, 256, kernel_size=1, padding= 0)
        self.conv_2 = nn.Conv2d(256, 512, kernel_size=3, stride = 2, padding= 1) ## 5 x 5

        self.conv_3 = nn.Conv2d(512, 128, kernel_size=1, padding= 0)
        self.conv_4 = nn.Conv2d(128, 256, kernel_size=3, stride = 2, padding= 1) ## 3 x 3

        self.conv_5 = nn.Conv2d(256, 128, kernel_size=1, padding= 0)
        self.conv_6 = nn.Conv2d(128, 256, kernel_size=3, stride = 2, padding= 1) ## 2 x 2

        self.conv_7 = nn.Conv2d(256, 128, kernel_size=1, padding= 0)
        self.conv_8 = nn.Conv2d(128, 256, kernel_size=3, stride = 2, padding= 1) ## 1 x 1
    
    def forward(self, x_2):
        x = F.relu6(self.conv_1(x_2))
        x_3 = F.relu6(self.conv_2(x))
        x = F.relu6(self.conv_3(x_3))
        x_4 = F.relu6(self.conv_4(x))
        x = F.relu6(self.conv_5(x_4))
        x_5 = F.relu6(self.conv_6(x))
        x = F.relu6(self.conv_7(x_5))
        x_6 = F.relu6(self.conv_8(x))
        return x_3, x_4, x_5 , x_6 


class Prediction_Convolutions(nn.Module):
    def __init__(self, n_classes):
        super(Prediction_Convolutions, self).__init__()
        self.n_classes = n_classes

        ## Numbers of anchors every pixel have in feature map
        n_anchors = {'x_1':4, 'x_2':6, 'x_3':6, 'x_4':6, 'x_5':4, 'x_6':4}

        ## Localization Convolutions. Multiply by 4 because of 4 offsets to predict
        self.loc_x_1 = nn.Conv2d(96, n_anchors['x_1'] * 4, kernel_size = 3, padding = 1)
        self.loc_x_2 = nn.Conv2d(160, n_anchors['x_2'] * 4, kernel_size = 3, padding = 1)
        self.loc_x_3 = nn.Conv2d(512, n_anchors['x_3'] * 4, kernel_size = 3, padding = 1)
        self.loc_x_4 = nn.Conv2d(256, n_anchors['x_4'] * 4, kernel_size = 3, padding = 1)
        self.loc_x_5 = nn.Conv2d(256, n_anchors['x_5'] * 4, kernel_size = 3, padding = 1)
        self.loc_x_6 = nn.Conv2d(256, n_anchors['x_6'] * 4, kernel_size = 3, padding = 1)

        ##Class Prediction convolutions
        self.cls_x_1 = nn.Conv2d(96, n_anchors['x_1'] * self.n_classes, kernel_size = 3, padding=1)
        self.cls_x_2 = nn.Conv2d(160, n_anchors['x_2'] * self.n_classes, kernel_size = 3, padding=1)
        self.cls_x_3 = nn.Conv2d(512, n_anchors['x_3'] * self.n_classes, kernel_size = 3, padding=1)
        self.cls_x_4 = nn.Conv2d(256, n_anchors['x_4'] * self.n_classes, kernel_size = 3, padding=1)
        self.cls_x_5 = nn.Conv2d(256, n_anchors['x_5'] * self.n_classes, kernel_size = 3, padding=1)
        self.cls_x_6 = nn.Conv2d(256, n_anchors['x_6'] * self.n_classes, kernel_size = 3, padding=1)
    
    def forward(self, x_1, x_2, x_3, x_4, x_5, x_6):
        batch_size = x_1.size(0)
        ## For Localization
        ## Permuting because flattening will be done in feture map dimension but we want the all anchor boxes 
        ## for one every feature map
        l_x_1 = self.loc_x_1(x_1).permute(0,2,3,1).contiguous().view(batch_size, -1, 4)
        l_x_2 = self.loc_x_2(x_2).permute(0,2,3,1).contiguous().view(batch_size, -1, 4)
        l_x_3 = self.loc_x_3(x_3).permute(0,2,3,1).contiguous().view(batch_size, -1, 4)
        l_x_4 = self.loc_x_4(x_4).permute(0,2,3,1).contiguous().view(batch_size, -1, 4)
        l_x_5 = self.loc_x_5(x_5).permute(0,2,3,1).contiguous().view(batch_size, -1, 4)
        l_x_6 = self.loc_x_6(x_6).permute(0,2,3,1).contiguous().view(batch_size, -1, 4)

        ## For Classification
        c_x_1 = self.cls_x_1(x_1).permute(0,2,3,1).contiguous().view(batch_size, -1, self.n_classes)
        c_x_2 = self.cls_x_2(x_2).permute(0,2,3,1).contiguous().view(batch_size, -1, self.n_classes)
        c_x_3 = self.cls_x_3(x_3).permute(0,2,3,1).contiguous().view(batch_size, -1, self.n_classes)
        c_x_4 = self.cls_x_4(x_4).permute(0,2,3,1).contiguous().view(batch_size, -1, self.n_classes)
        c_x_5 = self.cls_x_5(x_5).permute(0,2,3,1).contiguous().view(batch_size, -1, self.n_classes)
        c_x_6 = self.cls_x_6(x_6).permute(0,2,3,1).contiguous().view(batch_size, -1, self.n_classes)

        localization = torch.cat([l_x_1, l_x_2, l_x_3, l_x_4, l_x_5, l_x_6], dim = 1)
        confidence = torch.cat([c_x_1, c_x_2, c_x_3, c_x_4, c_x_5, c_x_6], dim = 1)
        return localization, confidence

class SSD300(nn.Module):
    def __init__(self, n_classes):
        super(SSD300, self).__init__()
        self.n_classes = n_classes
        self.backbone = Mobilenetv2_base()
        self.aux_conv = Auxillary_Convolutions()
        self.predict_conv = Prediction_Convolutions(self.n_classes)
        self.softmax = nn.Softmax(dim=2)   ## batch x priors x classes
    
    def forward(self, image):
        x_1, x_2 = self.backbone(image)
        x_3, x_4, x_5, x_6 = self.aux_conv(x_2)
        localization, confidence = self.predict_conv(x_1, x_2, x_3, x_4, x_5, x_6)
        confidence = self.softmax(confidence)
        return localization, confidence


if __name__ == "__main__":
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    model = SSD300(21)
    model = model.to(device)
    image = torch.randn(2, 3, 300,300)
    image = image.to(device)
    output = model(image)
    # print(summary(model, (3, 300 , 300)))
