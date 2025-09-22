import os
import torch
import numpy as np

from src.crowd_count import CrowdCounter
from src import network
from src.data_loader import ImageDataLoader
from src import utils


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
vis = True
save_output = True

data_path = '../data/original/shanghaitech/part_B_final/test_data/images/' # each image = 1024 x 768
gt_path = '../data/original/shanghaitech/part_B_final/test_data/ground_truth_csv/'
model_path = '../final_models/mcnn_shtechB_110.h5'

output_dir = './output/'
model_name = os.path.basename(model_path).split('.')[0]
file_results = os.path.join(output_dir,'results_' + model_name + '_.txt')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
output_dir = os.path.join(output_dir, 'density_maps_' + model_name)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


net = CrowdCounter()
      
trained_model = os.path.join(model_path)
network.load_net(trained_model, net) # The trained weight of the model is loaded.
#net.cuda()
net.eval()
mae = 0.0
mse = 0.0

#load test data
# data_path - test images
# gt_path - csv file, containing ground_truth density maps
# gt_downsample - downsampling density map to save memory
# pre_load - load all data to memory
data_loader = ImageDataLoader(data_path, gt_path, shuffle=False, gt_downsample=True, pre_load=True)

for blob in data_loader:
    # input image as a tensor
    im_data = blob['data'] # each image = 1024 x 768
    # density map for the image
    gt_data = blob['gt_density']
    # predict density map for im_data
    density_map = net(im_data, gt_data) # net.forward()
    # torch tensor to numpy array
    density_map = density_map.data.cpu().numpy() # if (gt_downsample) 192 x 256
    # number of people in ground truth
    gt_count = np.sum(gt_data)
    # number of people in prediction
    et_count = np.sum(density_map)

    mae += abs(gt_count-et_count)
    mse += ((gt_count-et_count)*(gt_count-et_count))
    if vis:
        utils.display_results(im_data, gt_data, density_map)
    if save_output:
        utils.save_density_map(density_map, output_dir, 'output_' + blob['fname'].split('.')[0] + '.png')

# data_loader.get_num_samples() - number of test images
mae = mae/data_loader.get_num_samples()
mse = np.sqrt(mse/data_loader.get_num_samples()) # -> RMSE
print('\nMAE: %0.2f, MSE: %0.2f' % (mae,mse))

f = open(file_results, 'w') 
f.write('MAE: %0.2f, MSE: %0.2f' % (mae,mse))
f.close()