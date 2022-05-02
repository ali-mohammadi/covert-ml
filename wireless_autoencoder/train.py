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
        model.batch_number = 0
        for i, (x, y) in enumerate(dl):
            x = x.to(device)
            y = y.to(device)

            if (model_parameters['channel_samples'] != None):
                loss = 0
                for i in range(model_parameters['channel_samples']):
                    output = model(x)
                    loss += loss_fn(output, y)
                loss = loss / model_parameters['channel_samples']
            else:
                output = model(x)
                loss = loss_fn(output, y)

            optim.zero_grad()
            loss.backward()
            optim.step()
            model.batch_number += 1
        losses.append(loss.item())
        if (epoch + 1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch + 1, num_epoch, loss.item()))
    return losses


def run_train(seed=model_parameters['seed']):
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
    model.fading = torch.randn((model_parameters['train_size'], 1), dtype=torch.cfloat).to(device)
    # model.fading = jakes_flat(Ns=(model_parameters['train_size']))

    losses = train(model, train_dl, model_parameters['num_epochs'], lr=model_parameters['learning_rate'], loss_fn=F.cross_entropy,
                   optim_fn=torch.optim.Adam)
    if model_parameters['channel_type'] == 'rayleigh':
        losses += train(model, train_dl, model_parameters['num_epochs'], lr=model_parameters['learning_rate'] * 1e-1, loss_fn=F.cross_entropy, optim_fn=torch.optim.Adam)
        losses += train(model, train_dl, model_parameters['num_epochs'], lr=model_parameters['learning_rate'] * 1e-2, loss_fn=F.cross_entropy, optim_fn=torch.optim.Adam)

    model.save(train_ds, test_ds, save_signals=True)

    model_parameters['_batch_size'] = model_parameters['batch_size']
    model.eval()
    model.batch_number = 0
    model_parameters['batch_size'] = model_parameters['test_size']
    bler_chart(model, test_ds, manual_seed=seed)
    losses_chart(losses)
    model_parameters['batch_size'] = model_parameters['_batch_size']


def run_eval(seed=model_parameters['seed']):
    torch.manual_seed(seed)
    model = Wireless_Autoencoder(model_parameters['m'], model_parameters['n_channel'])
    model.eval()
    model.load()
    model.fading = torch.randn((model_parameters['test_size'], 1), dtype=torch.cfloat)
    # model.fading = jakes_flat(Ns=(model_parameters['train_size']))
    model.batch_number = 0
    model_parameters['batch_size'] = model_parameters['test_size']
    test_ds = torch.load('../data/' + model_parameters['channel_type'] + '/test_' + str(model_parameters['n_channel']) + '-' + str(model_parameters['k']) + '.pt')
    bler_chart(model, test_ds)


'''
    Run training or evaluation
'''
run_train()
# run_eval()

'''
    Run train
'''
# fading = torch.randn((model_parameters['test_size'], 1), dtype=torch.cfloat).to(device)
# model_parameters['batch_size'] = model_parameters['test_size']
# manual_seed = model_parameters['seed']
# run_eval()