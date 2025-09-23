import os
import torch
import numpy as np
import sys

from src.crowd_count import CrowdCounter
from src import network
from src.data_loader import ImageDataLoader
from src.timer import Timer
from src import utils
from src.evaluate_model import evaluate_model

try:
    from termcolor import cprint
except ImportError:
    cprint = None

try:
    from pycrayon import CrayonClient
except ImportError:
    CrayonClient = None


def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)



method = 'mcnn'
dataset_name = 'shtechA'
output_dir = './saved_models/'

train_path = '../data/formatted_trainval/shanghaitech_part_A_patches_9/train'
train_gt_path = '../data/formatted_trainval/shanghaitech_part_A_patches_9/train_den'
val_path = '../data/formatted_trainval/shanghaitech_part_A_patches_9/val'
val_gt_path = '../data/formatted_trainval/shanghaitech_part_A_patches_9/val_den'

#training configuration
start_step = 0
end_step = 2000 # number of epochs
lr = 0.00001 # learning speed
momentum = 0.9
disp_interval = 500 # every disp_interval display statistics
log_interval = 250


#Tensorboard  config
use_tensorboard = False # log metrics through tensorboard
save_exp_name = method + '_' + dataset_name + '_' + 'v1'
remove_all_log = False   # remove all historical experiments in TensorBoard
exp_name = None # the previous experiment name in TensorBoard

# ------------
rand_seed = 64678  
if rand_seed is not None:
    np.random.seed(rand_seed) # fix random numbers generator (same operations are repeatable now)
    torch.manual_seed(rand_seed) # fix generator for cpu
    torch.cuda.manual_seed(rand_seed) # fix generator for gpu


# load net
net = CrowdCounter()
network.weights_normal_init(net, dev=0.01) # normal distribution random weights init
#net.cuda()
net.train() # switch model to train mode

params = list(net.parameters()) # all net parameters
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr) # Adam optimisation

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# tensorboard
use_tensorboard = use_tensorboard and CrayonClient is not None
if use_tensorboard:
    cc = CrayonClient(hostname='127.0.0.1')
    if remove_all_log:
        cc.remove_all_experiments()
    if exp_name is None:    
        exp_name = save_exp_name 
        exp = cc.create_experiment(exp_name)
    else:
        exp = cc.open_experiment(exp_name)

# training
train_loss = 0
step_cnt = 0 # global batch counter
re_cnt = False
t = Timer()
t.tic()

#load test data
# train_path - train images
# train_gt_path - csv files, containing train density maps
# shuffle - the data is shuffled randomly before each training epoch
# gt_downsample - downsampling density map to save memory
# pre_load - load all data to memory
data_loader = ImageDataLoader(train_path, train_gt_path, shuffle=True, gt_downsample=True, pre_load=True)
data_loader_val = ImageDataLoader(val_path, val_gt_path, shuffle=False, gt_downsample=True, pre_load=True)
best_mae = float("inf")

for epoch in range(start_step, end_step+1):    
    step = -1
    train_loss = 0 # total loss per epoch
    for blob in data_loader:      # blob = {image, density_map, image_file_name}
        step = step + 1        
        im_data = blob['data']
        gt_data = blob['gt_density']
        density_map = net(im_data, gt_data)
        loss = net.loss # MSE loss function
        train_loss += loss.item()
        step_cnt += 1
        optimizer.zero_grad() # old gradient reset
        loss.backward() # backpropagation (get new gradients)
        optimizer.step() # update models weights

        # logging
        if step % disp_interval == 0:
            duration = t.toc(average=False)
            # learning speed (batch/sec)
            fps = step_cnt / duration
            # ground truth number of people
            gt_count = np.sum(gt_data)
            density_map = density_map.data.cpu().numpy()
            # prediction number of people
            et_count = np.sum(density_map)
            utils.save_results(im_data,gt_data,density_map, output_dir)
            log_text = 'epoch: %4d, step %4d, Time: %.4fs, gt_cnt: %4.1f, et_cnt: %4.1f' % (epoch,
                step, 1./fps, gt_count,et_count)
            log_print(log_text, color='green', attrs=['bold'])
            re_cnt = True    
    
       
        if re_cnt:                                
            t.tic()
            re_cnt = False

    # save weights every 2 epochs
    if (epoch % 2 == 0):
        # save weights to h5 format
        save_name = os.path.join(output_dir, '{}_{}_{}.h5'.format(method,dataset_name,epoch))
        # saving network...
        network.save_net(save_name, net)
        #calculate error on the validation dataset 
        mae,mse = evaluate_model(save_name, data_loader_val)
        if mae < best_mae:
            best_mae = mae
            best_mse = mse
            best_model = '{}_{}_{}.h5'.format(method,dataset_name,epoch)
        log_text = 'EPOCH: %d, MAE: %.1f, MSE: %0.1f' % (epoch,mae,mse)
        log_print(log_text, color='green', attrs=['bold'])
        log_text = 'BEST MAE: %0.1f, BEST MSE: %0.1f, BEST MODEL: %s' % (best_mae,best_mse, best_model)
        log_print(log_text, color='green', attrs=['bold'])
        if use_tensorboard:
            exp.add_scalar_value('MAE', mae, step=epoch)
            exp.add_scalar_value('MSE', mse, step=epoch)
            exp.add_scalar_value('train_loss', train_loss/data_loader.get_num_samples(), step=epoch)
        
    

