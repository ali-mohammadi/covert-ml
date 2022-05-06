import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
import sys

from wireless_autoencoder.parameters import model_parameters
from wireless_covert.parameters import covert_parameters
from wireless_autoencoder.model import Wireless_Autoencoder
from models import Alice, Bob, Willie, device
from helpers import frange, jakes_flat, bler_chart, accuracies_chart, losses_chart, plot_constellation, reset_grads, discriminator_accuracy, \
    classifier_accuracy


def initialize():
    AE = Wireless_Autoencoder().to(device)
    AE.load()

    autoencoder_ds = torch.load(
        '../data/' + covert_parameters['channel_type'] + '/train_signals.pt')
    x = autoencoder_ds[:][0]
    y = autoencoder_ds[:][1]

    autoencoder_tds = torch.load(
        '../data/' + covert_parameters['channel_type'] + '/test_signals.pt')
    test_x = autoencoder_tds[:][0]
    test_y = autoencoder_tds[:][1]

    if covert_parameters['low_rate_multiplier'] is not None:
        x = x.reshape(-1, covert_parameters['n_channel'] * 2)
        _y = y.reshape(-1, covert_parameters['low_rate_multiplier'])
        test_x = test_x.reshape(-1, covert_parameters['n_channel'] * 2)
        _test_y = test_y.reshape(-1, covert_parameters['low_rate_multiplier'])

    x = x.to(device)
    y = y.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)

    m = (torch.rand(len(x)) * covert_parameters['m']).long().to(device)
    test_m = (torch.rand(len(test_x)) * covert_parameters['m']).long().to(device)

    # train_ds = TensorDataset(x, y)
    # train_dl = DataLoader(train_ds, model_parameters['batch_size']=model_parameters['model_parameters['batch_size']'], shuffle=False)

    W = Willie().to(device)
    A = Alice().to(device)
    B = Bob().to(device)

    return (A, B, W, AE), ((x, y), (test_x, test_y), (m, test_m))


def train(models, datasets, num_epochs=covert_parameters['num_epochs'], learning_rate=covert_parameters['learning_rate'],
          optim_fn=torch.optim.Adam, lambdas=(1, 1, 1)):
    A, B, W, AE = models
    train_ds, test_ds, m_ds = datasets

    x, y = train_ds
    test_x, test_y = test_ds
    m, test_m = m_ds

    w_optimizer = optim_fn(W.parameters(), lr=learning_rate)
    a_optimizer = optim_fn(A.parameters(), lr=learning_rate)
    b_optimizer = optim_fn(B.parameters(), lr=learning_rate)

    b_criterion = nn.CrossEntropyLoss().to(device)
    ae_criterion = nn.CrossEntropyLoss().to(device)
    w_criterion = nn.BCELoss().to(device)

    ae_lambda, bob_lambda, willie_lambda = lambdas

    accs = {'willie': [], 'bob': [], 'model': []}
    losses = {'willie': [], 'bob': [], 'alice': []}

    '''
        Overriding batch size (Batch training is not supported yet).
    '''
    covert_parameters['batch_size'] = x.size()[0]

    for epoch in range(num_epochs):
        '''
            Randomly switching noise power
        '''
        # model_parameters['ebno'] = random.randint(5, 10)

        A.train()
        B.train()
        W.train()
        AE.train()

        AE.batch_number = 0
        fading = torch.randn((x.size()[0], 1), dtype=torch.cfloat)
        fading_idx = torch.randperm(fading.shape[0])
        AE.fading = fading[fading_idx]

        for i in range(1):
            z = torch.randn(covert_parameters['batch_size'], covert_parameters['n_channel'] * 2).to(device)
            alice_output = A(z, m)

            normal_labels = torch.ones(covert_parameters['batch_size'], 1).to(device)
            covert_labels = torch.zeros(covert_parameters['batch_size'], 1).to(device)

            willie_output = W(AE.channel(x, covert_parameters['channel_type']))
            willie_normal_loss = w_criterion(willie_output, normal_labels)
            willie_output = W(AE.channel(alice_output + x, covert_parameters['channel_type']))
            willie_covert_loss = w_criterion(willie_output, covert_labels)
            willie_loss = willie_normal_loss + willie_covert_loss

            reset_grads(a_optimizer, b_optimizer, w_optimizer)
            willie_loss.backward()
            w_optimizer.step()

        z = torch.randn(covert_parameters['batch_size'], covert_parameters['n_channel'] * 2).to(device)
        alice_output = A(z, m)

        willie_output = W(AE.channel(alice_output + x, covert_parameters['channel_type']))
        bob_output = B(AE.channel(alice_output + x, covert_parameters['channel_type']))

        if covert_parameters['low_rate_multiplier'] is not None:
            alice_output = AE.channel(
                alice_output.reshape(-1, model_parameters['n_channel'] * 2) + x.reshape(-1, model_parameters[
                    'n_channel'] * 2), covert_parameters['channel_type'])
        else:
            alice_output = AE.channel(alice_output + x, covert_parameters['channel_type'])

        alice_bob_loss = ae_lambda * ae_criterion(AE.decoder_net(alice_output),
                                                  y) + willie_lambda * w_criterion(
            willie_output, torch.ones(covert_parameters['batch_size'], 1).to(device)) + bob_lambda * b_criterion(
            bob_output, m)

        reset_grads(a_optimizer, b_optimizer, w_optimizer)
        alice_bob_loss.backward()
        a_optimizer.step()

        z = torch.randn(covert_parameters['batch_size'], covert_parameters['n_channel'] * 2).to(device)
        alice_output = A(z, m)

        bob_output = B(AE.channel(alice_output + x, covert_parameters['channel_type']))
        bob_loss = b_criterion(bob_output, m)

        reset_grads(a_optimizer, b_optimizer, w_optimizer)
        bob_loss.backward()
        b_optimizer.step()

        if epoch % 10 == 0:
            with torch.no_grad():
                test_z = torch.randn(len(test_x), covert_parameters['n_channel'] * 2).to(device)

                A.eval()
                B.eval()
                W.eval()
                AE.eval()

                AE.batch_number = 0

                alice_output_test = A(test_z, test_m)
                
                willie_acc = discriminator_accuracy(
                    W(AE.channel(alice_output_test + test_x, covert_parameters['channel_type'])),
                    W(AE.channel(test_x, covert_parameters['channel_type'])))


                bob_acc = classifier_accuracy(
                    B(AE.channel(alice_output_test + test_x, covert_parameters['channel_type'])),
                    test_m)

                accs['willie'].append(round(willie_acc, 2))
                accs['bob'].append(round(bob_acc, 2))
                
                if covert_parameters['low_rate_multiplier'] is not None:
                    alice_output_test = AE.channel(
                        alice_output_test.reshape(-1, model_parameters['n_channel'] * 2) + test_x.reshape(-1,
                                                                                                            model_parameters[
                                                                                                                'n_channel'] * 2),
                        covert_parameters['channel_type'])
                else:
                    alice_output_test = AE.channel(alice_output_test + test_x, covert_parameters['channel_type'])
                # alice_output_test = AE.channel(test_x, covert_parameters['channel_type'])
                model_acc = classifier_accuracy(AE.decoder_net(alice_output_test), test_y)

                accs['model'].append(round(model_acc, 2))

                losses['willie'].append(willie_loss.item())
                losses['bob'].append(bob_loss.item())
                losses['alice'].append(alice_bob_loss.item())

            print(
                'Epoch[{}/{}], Alice Loss:{:.4f}, Willie Loss:{:.4f}, Willie Acc:{:.2f}, Bob Acc:{:.2f}, Model Acc:{:.2f}'.format(
                    epoch, num_epochs, alice_bob_loss.item(),
                    willie_loss.item(), willie_acc, bob_acc, model_acc))

    return accs, losses


def run_train(seed=covert_parameters['seed']):
    torch.manual_seed(seed)
    print('Device: ', 'CPU' if device == 'cpu' else 'GPU')
    models, datasets = initialize()
    A, B, E, AE = models

    ae_lambda, bob_lambda, willie_lambda = 1, 1.5, 1.25
    accs, losses = train(models, datasets, lambdas=(ae_lambda, bob_lambda, willie_lambda))

    torch.save(A.state_dict(), '../models/' + covert_parameters['channel_type'] + '/wireless_covert_alice.pt')
    torch.save(B.state_dict(), '../models/' + covert_parameters['channel_type'] + '/wireless_covert_bob.pt')
    torch.save(E.state_dict(), '../models/' + covert_parameters['channel_type'] + '/wireless_covert_willie.pt')

    accuracies_chart(accs)


def evaluate(ebno):
    models, datasets = initialize()
    A, B, W, AE = models
    _, test_ds, m_ds = datasets

    A.eval()
    B.eval()
    W.eval()
    AE.eval()

    test_x, test_y = test_ds
    m, test_m = m_ds

    AE.batch_number = 0
    fading = torch.randn((test_x.size()[0], 1), dtype=torch.cfloat)
    fading_idx = torch.randperm(fading.shape[0])
    AE.fading = fading[fading_idx].to(device)
    # AE.fading = fading

    A.load_state_dict(torch.load('../models/' + covert_parameters['channel_type'] + '/wireless_covert_alice.pt'))
    B.load_state_dict(torch.load('../models/' + covert_parameters['channel_type'] + '/wireless_covert_bob.pt'))
    W.load_state_dict(torch.load('../models/' + covert_parameters['channel_type'] + '/wireless_covert_willie.pt'))

    test_z = torch.randn(len(test_x), covert_parameters['n_channel'] * 2).to(device)

    willie_acc = discriminator_accuracy(W(AE.channel(A(test_z, test_m) + test_x, covert_parameters['channel_type'], ebno=ebno)),
                                 W(AE.channel(test_x, covert_parameters['channel_type'], ebno=ebno)))

    bob_acc = classifier_accuracy(B(AE.channel(A(test_z, test_m) + test_x, covert_parameters['channel_type'], ebno=ebno)),
                           test_m)

    alice_output_test = A(test_z, test_m)
    if covert_parameters['low_rate_multiplier'] is not None:
        alice_output_test = AE.channel(
            alice_output_test.reshape(-1, model_parameters['n_channel'] * 2) + test_x.reshape(-1, model_parameters[
                'n_channel'] * 2), covert_parameters['channel_type'], ebno=ebno)
    else:
        alice_output_test = AE.channel(alice_output_test + test_x, covert_parameters['channel_type'], ebno=ebno)
    # alice_output_test = AE.channel(test_x, covert_parameters['channel_type'], ebno=ebno)
    model_acc = classifier_accuracy(AE.decoder_net(alice_output_test), test_y)

    return model_acc, bob_acc, willie_acc


def run_eval():
    if covert_parameters['channel_type'] == 'rayleigh':
        ebno_range = list(frange(0, 21, 2))
    else:
        ebno_range = list(frange(-4, 8, 0.5))

    accs = {'model': [], 'bob': [], 'willie': []}

    for i in range(0, len(ebno_range)):
        model_acc, bob_acc, willie_acc = evaluate(ebno=ebno_range[i])
        accs['model'].append(1 - model_acc)
        accs['bob'].append(1 - bob_acc)
        accs['willie'].append(round(willie_acc, 2))

    for x in ['model', 'bob', 'willie']:
        if x == 'model':
            plot_label = 'Autoencoder' + "(" + str(model_parameters['n_channel']) + "," + str(model_parameters['k']) + ")"
        elif x == 'bob':
            plot_label = 'Bob' + "(" + str(covert_parameters['n_channel']) + "," + str(covert_parameters['k']) + ")"
        else:
            plot_label = 'Willie'

        if x == 'willie':
            plot_x = list(ebno_range)
            plot_y = accs[x]
            plot_x_new = np.linspace(min(plot_x), max(plot_x), 100)
            a_BSpline = make_interp_spline(plot_x, plot_y)
            plot_y = a_BSpline(plot_x_new)
            plot_x = plot_x_new
            plot_fmt = 'g-'
        elif x == 'bob':
            plot_x = list(ebno_range)
            plot_y = accs[x]
            plot_fmt = 'ro'
        else:
            plot_x = list(ebno_range)
            plot_y = accs[x]
            plot_fmt = 'bo'

        plt.plot(plot_x, plot_y, plot_fmt,
                 label=plot_label)

        if x == 'model' or x == 'bob':
            plt.yscale('log')

        plt.xlabel('Eb/N0')
        if x == 'model' or x == 'bob':
            plt.ylabel('Block Error Rate')
        else:
            plt.ylabel('Accuracy')

        plt.grid()
        plt.legend(loc="upper right", ncol=1)

        if x == 'model':
            plt.savefig("results/autoencoder_bler_" + covert_parameters['channel_type'] + ".png")
        elif x == 'bob':
            plt.savefig(
                "results/bob_bler_" +
                covert_parameters['channel_type'] + ".png")
        else:
            plt.savefig(
                "results/willie_accuracy_" +
                covert_parameters['channel_type'] + ".png")

        plt.show()


'''
    Run training or evaluation
'''
# for i in range(0, 100):
run_train()
run_eval()
