import torch
import torch.nn as nn
from network import Conv2d

class MCNN(nn.Module):
    '''
    Multi-column CNN 
        -Implementation of Single Image Crowd Counting via Multi-column CNN (Zhang et al.)
    '''
    
    def __init__(self, bn=False):
        super(MCNN, self).__init__()

        # branch1 (large kernel size) for large objects. (grayscale)
        self.branch1 = nn.Sequential(Conv2d( 1, 16, 9, same_padding=True, bn=bn), # batch normalization
                                     nn.MaxPool2d(2), # 512 x 384
                                     Conv2d(16, 32, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2), # 256 x 192
                                     Conv2d(32, 16, 7, same_padding=True, bn=bn),
                                     Conv2d(16,  8, 7, same_padding=True, bn=bn))

        # branch2 (medium kernel size) for medium objects. (grayscale)
        self.branch2 = nn.Sequential(Conv2d( 1, 20, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(20, 40, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(40, 20, 5, same_padding=True, bn=bn),
                                     Conv2d(20, 10, 5, same_padding=True, bn=bn))

        # branch3 (small kernel size) for small objects. (grayscale)
        self.branch3 = nn.Sequential(Conv2d( 1, 24, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(24, 48, 3, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(48, 24, 3, same_padding=True, bn=bn),
                                     Conv2d(24, 12, 3, same_padding=True, bn=bn))
        # FUSION
        # branch1 → 8 channels
        # branch2 → 10 channels
        # branch3 → 12 channels
        # input: 8 + 10 + 12 = 30 channels, kernel_size = 1 x 1
        # idea: takes all 30 channels and combines them into 1 at each pixel.
        # output: 1 channel
        self.fuse = nn.Sequential(Conv2d( 30, 1, 1, same_padding=True, bn=bn))
        
    def forward(self, im_data):
        x1 = self.branch1(im_data) # 1, 8, 192, 256
        x2 = self.branch2(im_data) # 1, 10, 192, 256
        x3 = self.branch3(im_data) # 1, 12, 192, 256
        x = torch.cat((x1,x2,x3),1) # concat = 1, 30, 192, 256
        x = self.fuse(x) # fusion = 1, 1, 192, 256
        
        return x