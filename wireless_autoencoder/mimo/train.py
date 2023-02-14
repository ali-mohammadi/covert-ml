import math

import numpy
import torch
# torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import sys

from parameters import *
from shared_parameters import system_parameters
from model import Wireless_Autoencoder, Wireless_Decoder, device
from helpers import jakes_flat, losses_chart, plot_constellation, plot_constellations, blers_chart


decoder_model = None


def train(models, dls, num_epoch, lr, loss_fn, optim_fn=torch.optim.Adam, losses=None, losses_ind=None, decoder_model=None):
    params = list()

    # for i in range(model_parameters['n_user']):
    #     params += list(models[i].parameters())

    for i in range(model_parameters['n_user']):
        params += list(models[i].encoder.parameters())
    params += list(decoder_model.parameters())

    optim = optim_fn(params, lr=lr, amsgrad=True)

    if losses is None:
        losses = []
    if losses_ind is None:
        losses_ind = [[] for _ in range(model_parameters['n_user'])]

    coefficients = [1 / model_parameters['n_user'] for _ in range(model_parameters['n_user'])]
    for epoch in range(num_epoch):
        for _, xys in enumerate(zip(*dls)):
            if channel_parameters['channel_type'] == 'awgn':
                encodes = []
                signals = []
                labels = []
                for i, ((x, y), model) in enumerate(zip(xys, models)):
                    x = x.to(device)
                    y = y.to(device)
                    encode = model.encoder(x)
                    encodes.append(encode)
                    signals.append(model.channel(encode))
                    labels.append(y)

                for i in range(model_parameters['n_user']):
                    for j, encode in enumerate(encodes):
                        if j != i:
                            signals[i] += encode

                signals = torch.stack(signals)
                signals = signals.transpose(0, 1)
                signals = signals.contiguous().view(signals.size()[0], -1)
            else:
                encodes = []
                labels = []
                for i, ((x, y), model) in enumerate(zip(xys, models)):
                    x = x.to(device)
                    y = y.to(device)
                    encode = model.encoder(x)
                    encodes.append(encode)
                    labels.append(y)

                encodes = torch.stack(encodes)
                encodes.transpose_(0, 1)

                h = torch.randn((x.size()[0], model_parameters['n_user'], model_parameters['n_user']), dtype=torch.cfloat).to(device)
                signals = []
                for i in range(model_parameters['n_user']):
                    signals.append(models[i].channel(encodes, h=h)[:, i, :])
                signals = torch.stack(signals).transpose(1, 0)

                signals = signals.view((-1, model_parameters['n_user'], model_parameters['n_channel'], 2))
                signals = torch.view_as_complex(signals)
                transform = torch.inverse(h) @ signals
                signals = torch.view_as_real(transform).view(transform.size()[0], -1)

            loss_ind = [[0] for _ in range(model_parameters['n_user'])]
            for i in range(model_parameters['n_user']):
                ''' 
                    Separate receivers
                '''
                # if channel_parameters['channel_type'] == 'awgn':
                #     decode = models[i].decoder_net(signals[i])
                # else:
                #     decode = models[i].decoder_net(signals, torch.view_as_real(h).view(-1, 2 * model_parameters['n_user'] * model_parameters['n_user']))
                '''
                    Central receiver
                '''
                decode = decoder_model(signals, i)
                loss_ind[i] = coefficients[i] * loss_fn(decode, labels[i]) * 10


            loss = sum(loss_ind)
            # loss = max(loss_ind)

            optim.zero_grad()
            loss.backward()
            optim.step()

            with torch.no_grad():
                for i in range(model_parameters['n_user']):
                    coefficients[i] = loss_ind[i].item() / loss.item()

        losses.append(loss.item())
        for i in range(model_parameters['n_user']):
            losses_ind[i].append(loss_ind[i].item())

        if (epoch + 1) % 10 == 0:
            output = 'Epoch [{}/{}], '.format(epoch + 1, num_epoch)
            for i in range(model_parameters['n_user']):
                output += (' L' + str(i + 1) + ': {:.5f},'.format(loss_ind[i].item()))
            output += ' Loss: {:.5f}'.format(loss.item())
            print(output)

    return losses, losses_ind


def run_train(seed=model_parameters['seed'], save=True):
    torch.manual_seed(seed)
    print('Device: ', 'CPU' if device == 'cpu' else 'GPU')
    train_labels = (torch.rand(model_parameters['train_size']) * model_parameters['m']).long()
    train_data = torch.sparse.torch.eye(model_parameters['m']).index_select(dim=0, index=train_labels)
    test_labels = (torch.rand(model_parameters['test_size']) * model_parameters['m']).long()
    test_data = torch.sparse.torch.eye(model_parameters['m']).index_select(dim=0, index=test_labels)
    train_ds = Data.TensorDataset(train_data, train_labels)
    test_ds = Data.TensorDataset(test_data, test_labels)

    train_dls = []
    models = []

    for i in range(model_parameters['n_user']):
        train_dls.append(DataLoader(train_ds, batch_size=model_parameters['batch_size'], shuffle=True))
        model = Wireless_Autoencoder(model_parameters['m'], model_parameters['n_channel'], i + 1).to(device)
        model.train()
        models.append(model)

    global decoder_model
    decoder_model = Wireless_Decoder().to(device)
    decoder_model.train()

    train_dls = tuple(train_dls)
    models = tuple(models)

    losses, losses_ind = train(models, train_dls, model_parameters['num_epochs'], lr=model_parameters['learning_rate'],
                               loss_fn=F.cross_entropy,
                               optim_fn=torch.optim.Adam,
                               decoder_model=decoder_model)


    losses_chart(losses)
    losses_chart(losses_ind, True)

    if save:
        decoder_model.save()
        for i, model in enumerate(models):
            model.save(train_ds, test_ds)
            if i == 0:
                torch.save(torch.random.get_rng_state(),
                           '../../data/' + system_parameters['system_type'] + '/' + channel_parameters[
                               'channel_type'] + '/' + str(model_parameters['n_user']) + '-user/channel_random_state.pt')

    run_eval(models, test_ds, seed)
    return decoder_model


def run_eval(models=None, test_ds=None, seed=model_parameters['seed']):
    torch.manual_seed(seed)
    torch.random.set_rng_state(torch.load(
        '../../data/' + system_parameters['system_type'] + '/' + channel_parameters['channel_type'] + '/' + str(
            model_parameters['n_user']) + '-user/channel_random_state.pt'))
    print('Device: ', 'CPU' if device == 'cpu' else 'GPU')
    if test_ds is None:
        test_ds = torch.load('../../data/' + system_parameters['system_type'] + '/' + channel_parameters[
            'channel_type'] + '/' + str(model_parameters['n_user']) + '-user/wireless_autoencoder(' + str(model_parameters['n_channel']) + ',' + str(
            model_parameters['k']) + ')_test.pt')
    if models is None:
        models = []
        for i in range(model_parameters['n_user']):
            model = Wireless_Autoencoder(model_parameters['m'], model_parameters['n_channel'], i + 1).to(device)
            model.load()
            model.eval()
            models.append(model)
    else:
        for model in models:
            model.eval()

    global decoder_model
    if decoder_model is None:
        decoder_model = Wireless_Decoder().to(device)
        decoder_model.load()
        decoder_model.eval()

    test_dses = []
    for i in range(model_parameters['n_user']):
        idx = torch.randperm(len(test_ds))
        test_dses.append(Data.TensorDataset(test_ds[idx][0], test_ds[idx][1]))

    blers_chart(models, test_dses, decoder_model)


'''
    Run training or evaluation
'''
# run_train(save=False)
run_eval()


# plot_constellations()
