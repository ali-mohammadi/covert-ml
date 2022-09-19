import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import sys

from parameters import *
from shared_parameters import system_parameters
from model import Wireless_Autoencoder, device
from helpers import jakes_flat, losses_chart, plot_constellation, blers_chart

assert system_parameters['system_type'] == 'mimo'

def train(models, dls, num_epoch, lr, loss_fn, optim_fn=torch.optim.Adam, losses=None, losses_ind=None):
    params = list()
    for model in models:
        params += list(model.parameters())

    optim = optim_fn(params, lr=lr, amsgrad=True)

    if losses is None:
        losses = []
    if losses_ind is None:
        losses_ind = [[] for _ in range(model_parameters['n_user'])]

    coefficients = [1] * model_parameters['n_user']
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

                loss_ind = [0] * model_parameters['n_user']
                for i in range(model_parameters['n_user']):
                    decode = models[i].decoder_net(signals[i])
                    loss_ind[i] = coefficients[i] * loss_fn(decode, labels[i]) * 10
                    if epoch % 100 == 0 and _ == 3:
                        demodulate = models[i].demodulate(signals[i])
                        print("Autoencoder" + str(i) + " Acc: ", torch.sum(demodulate != labels[i]).item() / len(demodulate))
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
                # h_r = torch.normal(np.sqrt(0.5), 0, (x.size()[0], model_parameters['n_user'], model_parameters['n_user']))
                # h_i = torch.normal(np.sqrt(0.5), 0, (x.size()[0], model_parameters['n_user'], model_parameters['n_user']))
                # h = torch.complex(h_r, h_i).to(device)
                signals = Wireless_Autoencoder().channel(encodes, h=h)

                loss_ind = [0] * model_parameters['n_user']
                for i in range(model_parameters['n_user']):
                    decode = models[i].decoder_net(signals, torch.view_as_real(h).view(-1, 2 * model_parameters['n_user'] * model_parameters['n_user']))
                    loss_ind[i] = coefficients[i] * loss_fn(decode, labels[i])

            loss = sum(loss_ind)

            optim.zero_grad()
            loss.backward()
            optim.step()

            # with torch.no_grad():
            #     for i in range(model_parameters['n_user']):
            #         coefficients[i] = loss_ind[i].item() / loss.item()

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

    train_dls = tuple(train_dls)
    models = tuple(models)

    losses, losses_ind = train(models, train_dls, model_parameters['num_epochs'], lr=model_parameters['learning_rate'],
                               loss_fn=F.cross_entropy,
                               optim_fn=torch.optim.Adam)
    losses, losses_ind = train(models, train_dls, model_parameters['num_epochs'],
                               lr=model_parameters['learning_rate'] * 1e-1,
                               loss_fn=F.cross_entropy,
                               optim_fn=torch.optim.Adam, losses=losses, losses_ind=losses_ind)



    losses_chart(losses)
    # losses_chart([loss[-50:] for loss in losses_ind], True)
    losses_chart(losses_ind, True)

    if save:
        for i, model in enumerate(models):
            model.save(train_ds, test_ds)
            if i == 0:
                torch.save(torch.random.get_rng_state(),
                           '../../data/' + system_parameters['system_type'] + '/' + channel_parameters[
                               'channel_type'] + '/channel_random_state.pt')
        run_eval(models, test_ds)
    else:
        run_eval(models, test_ds, seed)


def run_eval(models=None, test_ds=None, seed=model_parameters['seed']):
    torch.manual_seed(seed)
    print('Device: ', 'CPU' if device == 'cpu' else 'GPU')
    if test_ds is None:
        test_ds = torch.load('../../data/' + system_parameters['system_type'] + '/' + channel_parameters[
            'channel_type'] + '/wireless_autoencoder(' + str(model_parameters['n_channel']) + ',' + str(
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

    test_dses = []
    for i in range(model_parameters['n_user']):
        idx = torch.randperm(len(test_ds))
        test_dses.append(Data.TensorDataset(test_ds[idx][0], test_ds[idx][1]))

    # torch.random.set_rng_state(torch.load('../../data/' + system_parameters['system_type'] + '/' + channel_parameters['channel_type'] + '/channel_random_state.pt'))
    blers_chart(models, test_dses)


'''
    Run training or evaluation
'''
run_train()
# run_eval()

def plot_constellations():
    with torch.no_grad():
        n_samples = 100
        models = []
        for i in range(model_parameters['n_user']):
            model = Wireless_Autoencoder(model_parameters['m'], model_parameters['n_channel'], i + 1).to(device)
            model.load()
            model.eval()
            models.append(model)
        encodes = []
        for i in range(model_parameters['n_user']):
            x = torch.sparse.torch.eye(model_parameters['m']).index_select(dim=0, index=torch.ones(n_samples).int() * 2).to(device)
            encode = models[i].encoder(x)
            encode = models[0].channel(encode)
            encodes.append(encode)

        encodes = torch.stack(encodes)
        encodes.transpose_(0, 1)

        h = torch.randn((n_samples, model_parameters['n_user'], model_parameters['n_user']),
                        dtype=torch.cfloat).to(device)
        # h_r = torch.normal(np.sqrt(0.5), 0,
        #                    (x.size()[0], model_parameters['n_user'], model_parameters['n_user']))
        # h_i = torch.normal(np.sqrt(0.5), 0,
        #                    (x.size()[0], model_parameters['n_user'], model_parameters['n_user']))
        # h = torch.complex(h_r, h_i).to(device)
        # signals = models[0].channel(encodes, ebno=channel_parameters['ebno'], h=h)
        signals = encodes
        plot_constellation([(signals[:, 0, :], 'o', 'tab:red', 20, 1), (signals[:, 1, :], 'o', 'tab:blue', 20, 2)])