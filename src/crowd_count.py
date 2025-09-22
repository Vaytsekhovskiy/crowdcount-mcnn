import torch.nn as nn
import network
from models import MCNN


class CrowdCounter(nn.Module):
    def __init__(self):
        super(CrowdCounter, self).__init__() # extends torch.nn.Module
        self.DME = MCNN() # init MCNN
        self.loss_fn = nn.MSELoss() # Mean Squared Error loss function
        
    @property
    def loss(self):
        return self.loss_mse
    
    def forward(self,  im_data, gt_data=None):
        # numpy array -> torch tensor
        im_data = network.np_to_variable(im_data, is_cuda=False, is_training=self.training)
        # pass input image through MCNN -> return density_map
        density_map = self.DME(im_data)
        
        if self.training:
            # numpy array -> torch tensor
            gt_data = network.np_to_variable(gt_data, is_cuda=False, is_training=self.training)
            # update loss_mse
            self.loss_mse = self.build_loss(density_map, gt_data)
            
        return density_map
    
    def build_loss(self, density_map, gt_data):
        # calculating MSE loss
        loss = self.loss_fn(density_map, gt_data)
        return loss
