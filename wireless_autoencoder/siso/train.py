import math
from inspect import getmembers
from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.interpolate import interp1d


from parameters import *
from model import Wireless_Autoencoder, device
from helpers import jakes_flat, bler_chart, losses_chart, plot_constellation

def train(model, dl, num_epoch, lr, loss_fn, optim_fn=torch.optim.Adam):
    # for p in model.reverse_filter.parameters(): p.requires_grad = False
    optim = optim_fn(model.parameters(), lr=lr, amsgrad=True)
    losses = []
    for epoch in range(num_epoch):
        for i, (x, y) in enumerate(dl):
            x = x.to(device)
            y = y.to(device)
            '''
                When channel samples parameter is set, same signal passes through different channel realization.
            '''
            if channel_parameters['channel_samples'] is not None:
                loss = 0
                for i in range(channel_parameters['channel_samples']):
                    output = model(x)
                    loss += loss_fn(output, y)
                loss = loss / channel_parameters['channel_samples'] 
            else:
                output = model(x)
                loss = loss_fn(output, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

        losses.append(loss.item())
        if (epoch + 1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch + 1, num_epoch, loss.item()))
    return losses


def run_train(seed=model_parameters['seed'], save=True):
    torch.manual_seed(seed)
    print('Device: ', 'CPU' if device == 'cpu' else 'GPU')
    train_labels = (torch.rand(model_parameters['train_size']) * model_parameters['m']).long()
    train_data = torch.sparse.torch.eye(model_parameters['m']).index_select(dim=0, index=train_labels)
    test_labels = (torch.rand(model_parameters['test_size']) * model_parameters['m']).long()
    test_data = torch.sparse.torch.eye(model_parameters['m']).index_select(dim=0, index=test_labels)
    train_ds = Data.TensorDataset(train_data, train_labels)
    test_ds = Data.TensorDataset(test_data, test_labels)
    train_dl = DataLoader(train_ds, batch_size=model_parameters['batch_size'], shuffle=True)

    model = Wireless_Autoencoder(model_parameters['m'], model_parameters['n_channel']).to(device)

    model.train()

    losses = train(model, train_dl, model_parameters['num_epochs'], lr=model_parameters['learning_rate'], loss_fn=F.cross_entropy,
                   optim_fn=torch.optim.Adam)

    if save:
        model.save(train_ds, test_ds)
        torch.save(torch.random.get_rng_state(), '../../data/siso/' + channel_parameters['channel_type'] + '/channel_random_state.pt')
    else:
        run_eval(model, test_ds, seed)


def run_eval(model=None, test_ds=None, seed=model_parameters['seed']):
    if test_ds is None:
        test_ds = torch.load('../../data/siso/' + channel_parameters['channel_type'] + '/wireless_autoencoder(' + str(model_parameters['n_channel']) + ',' + str(model_parameters['k']) + ')_test.pt')
    torch.manual_seed(seed)
    torch.random.set_rng_state(torch.load('../../data/siso/' + channel_parameters['channel_type'] + '/channel_random_state.pt'))
    print('Device: ', 'CPU' if device == 'cpu' else 'GPU')
    if model is None:
        model = Wireless_Autoencoder(model_parameters['m'], model_parameters['n_channel']).to(device)
        model.load()
    model.eval()
    bler_chart(model, test_ds)


def eval_parameters():
    if channel_parameters['channel_type'] == 'rayleigh':
        ebno_range = list(range(0, 22, 5))
    elif channel_parameters['channel_type'] == 'rician':
        ebno_range = list(range(0, 22, 5))
    else:
        ebno_range = list(range(-4, 9, 1))
    parameters = ['4,2', '6,3', '8,4']

    colors = {'4,2': '--bH', '6,3': '--rH', '8,4': '-ko'}


    for parameter in parameters:
        if channel_parameters['channel_type'] == 'awgn':
            if parameter == '4,2':
                bler = [0.26646484375, 0.221171875, 0.178203125, 0.13908203125, 0.10208984375, 0.07140625, 0.0446875, 0.0256640625, 0.0143359375, 0.00630859375, 0.0025390625, 0.00064453125, 0.0001953125]
            if parameter == '6,3':
                bler = [0.33966796875, 0.2820703125, 0.220234375, 0.16537109375, 0.1171875, 0.07365234375, 0.04265625, 0.0213671875, 0.010625, 0.00361328125, 0.001328125, 0.00037109375, 5.859375e-05]
            if parameter == '8,4':
                bler = [0.3958203125, 0.32232421875, 0.2448046875, 0.17669921875, 0.11544921875, 0.0704296875, 0.03830078125, 0.01779296875, 0.007109375, 0.00271484375, 0.00078125, 0.00013671875, 1.953125e-05]
        if channel_parameters['channel_type'] == 'rayleigh':
            if parameter == '4,2':
                bler = [0.427421875, 0.23005859375, 0.0971875, 0.03318359375, 0.0112109375]
            if parameter == '6,3':
                bler = [0.45791015625, 0.21572265625, 0.09033984375, 0.03006640625, 0.0102890625]
            if parameter == '8,4':
                bler = [0.48138671875, 0.224609375, 0.082265625, 0.028125, 0.00951171875]
        if channel_parameters['channel_type'] == 'rician':
            if parameter == '4,2':
                bler = [0.1523046875, 0.035234375, 0.00763671875, 0.0024046875, 0.000903125]
            if parameter == '6,3':
                bler = [0.15349609375, 0.0339453125, 0.00731328125, 0.00219921875, 0.0006421875]
            if parameter == '8,4':
                bler = [0.18876953125, 0.04037109375, 0.00693359375, 0.00193359375, 0.0005078125]

        plt.plot(ebno_range, bler, colors[parameter], clip_on=False,
                 label="Autoencoder (" + parameter + ")")

    plt.yscale('log')
    plt.xlabel("$E_{b}/N_{0}$ (dB)", fontsize=16)
    plt.ylabel('Block Error Rate', fontsize=16)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid()
    plt.legend(loc="lower left", ncol=1, fontsize=18)
    plt.tight_layout()
    plt.savefig("results/png/autoencoder_bler_" + channel_parameters['channel_type'] + "_parameters.png")
    plt.savefig("results/eps/autoencoder_bler_" + channel_parameters['channel_type'] + "_parameters.eps", format="eps")
    plt.savefig("results/svg/autoencoder_bler_" + channel_parameters['channel_type'] + "_parameters.svg", format="svg")
    plt.show()

'''
    Run training or evaluation
'''
# run_train()
run_eval()
# eval_parameters()

