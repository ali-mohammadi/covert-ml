import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import sys

from parameters import *
from model import Wireless_Autoencoder, device
from helpers import jakes_flat, bler_chart, losses_chart, plot_constellation


def train(model, dl, num_epoch, lr, loss_fn, optim_fn=torch.optim.Adam):
    optim = optim_fn(model.parameters(), lr=lr, amsgrad=True)
    losses = []
    for epoch in range(num_epoch):
        for i, (x, y) in enumerate(dl):
            x = x.to(device)
            y = y.to(device)
            '''
                When channel samples parameter is set, same signal passes through different channel realization.
            '''
            if channel_parameters['channel_samples']  is not None:
                loss = 0
                for i in range(channel_parameters['channel_samples'] ):
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
        torch.save(torch.random.get_rng_state(), '../data/' + channel_parameters['channel_type'] + '/channel_random_state.pt')


def run_eval(model=None, test_ds=torch.load('../data/' + channel_parameters['channel_type'] + '/wireless_autoencoder(' + str(model_parameters['n_channel']) + ',' + str(model_parameters['k']) + ')_test.pt'), seed=model_parameters['seed']):
    torch.manual_seed(seed)
    torch.random.set_rng_state(torch.load('../data/' + channel_parameters['channel_type'] + '/channel_random_state.pt'))
    print('Device: ', 'CPU' if device == 'cpu' else 'GPU')
    if model is None:
        model = Wireless_Autoencoder(model_parameters['m'], model_parameters['n_channel']).to(device)
        model.load()
    model.eval()
    bler_chart(model, test_ds)


'''
    Run training or evaluation
'''
# run_train()
run_eval()