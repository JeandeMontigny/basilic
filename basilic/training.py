import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import h5py

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from collections import namedtuple
from collections import defaultdict

# ---------------------------------------------------------------- #
# tuple containing information for one spike
# TODO: adapt this tuple to my data, or adapt my data to this tuple?
TrainSpike = namedtuple('DataPoint',
    ['amps', 'waveforms', 'ch_locs', 'center_loc', 'spike_id', 'exp_id', 'min_waveform', 'min_amp'])

# ---------------------------------------------------------------- #
class TrainingDataSet(torch.utils.data.Dataset):
    # list of TrainSpike
    def __init__(self, TrainSpike_data_set):
        'Initialization'
        self.TrainSpike_data_set = TrainSpike_data_set

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.TrainSpike_data_set)

    def __getspike__(self, index):
        'Return one sample of data'
        train_spike = self.TrainSpike_data_set[index]

        # TODO: my data. Which are mandatory? I don't have center_loc/ch_locs, exp_id, min_amp, min_waveform
        amps = train_spike.amps
        waveforms = train_spike.waveforms
        ch_locs = train_spike.ch_locs
        center_loc = train_spike.center_loc
        spike_id = train_spike.spike_id
        exp_id = train_spike.exp_id
        min_amp = train_spike.min_amp
        min_waveform = train_spike.min_waveform

        #TODO: sort this out
        # my datas
        spike_time_list, spike_id_list, recorded_channels_ids, channel_locations_list, waveforms_list, waveforms_ids_list
        # decat model data
        amps, waveforms, ch_locs, center_loc, spike_id, exp_id, min_amp, min_waveform

        return

# ---------------------------------------------------------------- #
class EXPVAEWAVE(nn.Module):
    def __init__(self, training_set, args, abs_, optimize_both_exp=True, batchnorm=True, prior_var=80, device='cuda'):
        super(EXPVAEWAVE, self).__init__()
        self.capacity1 = 500
        self.capacity2 = 250
        self.dropout1 = 0
        self.dropout2 = 0
        self.optimize_both_exp = optimize_both_exp
        self.num_spikes = len(training_set)
        self.prior_varx = prior_var
        self.prior_vary = prior_var
        self.prior_varz = prior_var
        num_amps = training_set[0][0].shape[0]
        waveforms_flattened_dim = training_set[0][1].shape[0]*training_set[0][1].shape[1]

        self.fc1 = nn.Linear(waveforms_flattened_dim, self.capacity1)
        self.fc2 = nn.Linear(self.capacity1, self.capacity2)
        self.fc_mean = nn.Linear(self.capacity2, 3)
        self.fc_var = nn.Linear(self.capacity2, 3)

        self.batchnorm = batchnorm
        if(batchnorm):
            self.batchnorm1 = nn.BatchNorm1d(self.capacity1)
            self.batchnorm2 = nn.BatchNorm1d(self.capacity2)

        a_s = abs_[:,0]
        b_s = abs_[:,1]
        abs_ = torch.from_numpy(abs_).float().to(device)
        a_s = torch.from_numpy(a_s).float().to(device)
        self.b_s = torch.from_numpy(b_s).float().to(device)

        self.exps_0 = abs_.clone().detach()
        if(optimize_both_exp):
            self.exps = abs_.clone().detach().requires_grad_(True)
        else:
            self.exps = a_s.clone().detach().requires_grad_(True)

    def encode(self, amps, waveforms):
        EPSILON = 1e-6
        if(self.batchnorm):
            x = self.batchnorm1(self.fc1(torch.flatten(waveforms, start_dim=1)))
        else:
            x = self.fc1(torch.flatten(waveforms, start_dim=1))
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout1)
        if(self.batchnorm):
            x = self.batchnorm2(self.fc2(x))
        else:
            x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout2)
        xyz_mu = self.fc_mean(x)
        xyz_var = F.softplus(self.fc_var(x)) + EPSILON

        x_mu = xyz_mu[:,0]
        x_mu = x_mu.view(x_mu.shape[0], 1)
        x_var = xyz_var[:,0]
        x_var = x_var.view(x_var.shape[0], 1)

        y_mu = xyz_mu[:,1]
        y_mu = y_mu.view(y_mu.shape[0], 1)
        y_var = xyz_var[:,1]
        y_var = y_var.view(y_var.shape[0], 1)

        z_mu = xyz_mu[:,2]
        z_mu = z_mu.view(z_mu.shape[0], 1)
        z_var = xyz_var[:,2]
        z_var = z_var.view(z_var.shape[0], 1)
        z_var = z_var

        return x_mu, x_var, y_mu, y_var, z_mu, z_var

    def reparameterize_normal(self, mu, var):
        std = torch.sqrt(var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def getTensorDistances(self, n_loc, ch_locs):
        n_loc = n_loc.view(n_loc.shape[0], 1, n_loc.shape[1])
        subtract = (n_loc - ch_locs)**2
        summed = torch.sum(subtract, dim=2)
        return torch.sqrt(summed)

    def decode(self, sampled_n_loc, ch_locs, exp_ids):
        distances = self.getTensorDistances(sampled_n_loc, ch_locs)#
        #Exponential observation model with model parameters a, b
        if(self.optimize_both_exp):
            a_exps = torch.index_select(self.exps, 0, exp_ids)[:,0]
            b_exps = torch.index_select(self.exps, 0, exp_ids)[:,1]
        else:
            a_exps = torch.index_select(self.exps, 0, exp_ids)
            b_exps = torch.index_select(self.b_s, 0, exp_ids)
        a_exps = a_exps.view(a_exps.shape[0], 1)
        b_exps = b_exps.view(b_exps.shape[0], 1)
        recon_amps = -torch.exp(distances*b_exps + a_exps)
        return recon_amps

    def forward(self, amps, waveforms, ch_locs, exp_ids):
        x_mu, x_var, y_mu, y_var, z_mu, z_var = self.encode(amps, waveforms)
        if(self.training):
            x_sample = self.reparameterize_normal(x_mu, x_var)
            y_sample = self.reparameterize_normal(y_mu, y_var)
            z_sample = self.reparameterize_normal(z_mu, z_var)
        else:
            x_sample = x_mu
            y_sample = y_mu
            z_sample = z_mu
        sampled_n_loc = torch.cat((x_sample, y_sample, z_sample), 1)
        recon_amps = self.decode(sampled_n_loc, ch_locs, exp_ids)
        return recon_amps, x_mu, x_var, y_mu, y_var, z_mu, z_var

# ---------------------------------------------------------------- #
def load_training_data(file_name, data_directory = "../data/"):
    # load data from the file
    data_file = h5py.File(data_directory + file_name, 'r')

    spike_time_list = data_file['spike_time_list']
    spike_id_list = data_file['spike_id_list']
    recorded_channels_ids = data_file['recorded_channels_ids']
    channel_locations_list = data_file['channel_locations_list']
    waveforms_list = data_file['waveforms_list']
    waveforms_ids_list = data_file['waveforms_ids_list']

    data_file.close()

    # convert data to torch format
    torch_spike_time_list = torch.from_numpy(spike_time_list).float()
    torch_spike_id_list = torch.from_numpy(spike_id_list).float()
    torch_recorded_channels_ids = torch.from_numpy(recorded_channels_ids).float()
    torch_channel_locations_list = torch.from_numpy(channel_locations_list).float()
    torch_waveforms_list = torch.from_numpy(waveforms_list).float()
    torch_waveforms_ids_list = torch.from_numpy(waveforms_ids_list).float()

    TrainSpike_data_set = []
    # for each spike, create a TrainSpike containing all informations from this spike
    for i in enumerate(torch_spike_time_list):
        torch_spike_time_list[i]
        torch_spike_id_list[i]
        torch_recorded_channels_ids[i]
        torch_channel_locations_list[i]
        torch_waveforms_list[i]
        torch_waveforms_ids_list[i]
        # create TrainSpike
        train_spike = TrainSpike(each_data=each_data)
        # add it to the data set of TrainSpike
        TrainSpike_data_set.append(train_spike)

    training_set = TrainingDataSet(TrainSpike_data_set)

    # return spike_time_list, spike_id_list, recorded_channels_ids, channel_locations_list, waveforms_list, waveforms_ids_list
    return training_set

# ---------------------------------------------------------------- #
def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

# ---------------------------------------------------------------- #
# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_amps, t_amps, x_mu, x_var, y_mu, y_var, z_mu, z_var, epochs, prior_varx, prior_vary, prior_varz, device):
    batch_size = recon_amps.shape[0]
    MSE = torch.sum(F.mse_loss(recon_amps, t_amps[:,:,0], reduction='none')*t_amps[:,:,1])/batch_size

    m_qx = x_mu
    m_px = torch.zeros(m_qx.shape).to(device)
    var_qx = x_var
    var_px = (torch.zeros(var_qx.shape) + prior_varx**2).to(device)
    KLD_x = kl_divergence_normal(m_qx, var_qx, m_px, var_px)/batch_size

    m_qy = y_mu
    m_py = torch.zeros(m_qx.shape).to(device)
    var_qy = y_var
    var_py = (torch.zeros(var_qx.shape) + prior_vary**2).to(device)
    KLD_y = kl_divergence_normal(m_qy, var_qy, m_py, var_py)/batch_size

    m_qz = z_mu
    m_pz = torch.zeros(m_qx.shape).to(device)
    var_qz = z_var
    var_pz = (torch.zeros(var_qx.shape) + prior_varz**2).to(device)
    KLD_z = kl_divergence_normal(m_qz, var_qz, m_pz, var_pz)/batch_size

    return MSE + KLD_x + KLD_y + KLD_z

# ---------------------------------------------------------------- #
def kl_divergence_normal(mu_q, var_q, mu_p, var_p):
    kld = torch.sum(0.5*(torch.log(var_p) - torch.log(var_q)) + torch.div(var_q + (mu_q - mu_p)**2, 2*var_p) - 0.5)
    return kld

# ---------------------------------------------------------------- #
def train(model, device, args, optimizer, train_loader, epoch, train_losses):
    model.train()
    train_loss = 0
    num_loops = 0
    for batch_idx, (t_amps, t_waveforms, t_ch_locs, center_loc, _, exp_ids, _, _) in enumerate(train_loader):
        t_amps = t_amps.to(device)
        t_waveforms = t_waveforms.to(device)
        t_ch_locs = t_ch_locs.to(device)
        exp_ids = exp_ids.to(device)
        optimizer.zero_grad()
        recon_amps, x_mu, x_var, y_mu, y_var, z_mu, z_var = model(t_amps, t_waveforms, t_ch_locs, exp_ids)
        loss = loss_function(recon_amps, t_amps, x_mu, x_var, y_mu, y_var, z_mu, z_var, epoch, model.prior_varx, model.prior_vary, model.prior_varz, device)
        loss.backward()
        train_loss += loss.item()/t_amps.shape[0]
        num_loops += 1
        optimizer.step()
    train_losses.append(train_loss/num_loops)
    return model
