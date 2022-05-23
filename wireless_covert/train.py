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
from channel_parameters import channel_parameters
from wireless_autoencoder.model import Wireless_Autoencoder
from models import Alice, Bob, Willie, device
from helpers import frange, jakes_flat, bler_chart, accuracies_chart, losses_chart, plot_constellation, reset_grads, discriminator_accuracy, \
    classifier_accuracy


def initialize():
    AE = Wireless_Autoencoder()
    AE.load()
    AE.eval()
    with torch.no_grad():
        autoencoder_ds = torch.load(
            '../data/' + channel_parameters['channel_type'] + '/wireless_autoencoder(' + str(model_parameters['n_channel']) + ',' + str(model_parameters['k']) + ')_train.pt')
        x = AE.encoder(autoencoder_ds[:][0])
        y = autoencoder_ds[:][1]

        autoencoder_tds = torch.load(
            '../data/' + channel_parameters['channel_type'] + '/wireless_autoencoder(' + str(model_parameters['n_channel']) + ',' + str(model_parameters['k']) + ')_test.pt')
        test_x = AE.encoder(autoencoder_tds[:][0])
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

    W = Willie().to(device)
    A = Alice().to(device)
    B = Bob().to(device)
    AE = AE.to(device)

    return (A, B, W, AE), ((x, y), (test_x, test_y), (m, test_m))


def train(models, datasets, num_epochs=covert_parameters['num_epochs'], learning_rate=covert_parameters['learning_rate'],
          optim_fn=torch.optim.Adam, lambdas=(1, 1, 1)):
    A, B, W, AE = models
    train_set, test_set, m_set = datasets

    x, y = train_set
    test_x, test_y = test_set
    m, test_m = m_set

    if covert_parameters['batch_size'] is not None:
        dl = DataLoader(Data.TensorDataset(x, y, m), batch_size=covert_parameters['batch_size'], shuffle=True)
    else:
        covert_parameters['batch_size'] = len(x)
        dl = False

    w_optimizer = optim_fn(W.parameters(), lr=learning_rate, amsgrad=True)
    a_optimizer = optim_fn(A.parameters(), lr=learning_rate, amsgrad=True)
    b_optimizer = optim_fn(B.parameters(), lr=learning_rate, amsgrad=True)

    b_criterion = nn.CrossEntropyLoss().to(device)
    ae_criterion = nn.CrossEntropyLoss().to(device)
    w_criterion = nn.BCELoss().to(device)

    ae_lambda, bob_lambda, willie_lambda = lambdas

    accs = {'willie': [], 'bob': [], 'autoencoder': []}
    losses = {'willie': [], 'bob': [], 'alice': []}

    def train_batch(x, y, m):
        '''
            Randomly switching noise power to be robust against different noise levels
        '''
        if channel_parameters['channel_type'] == 'awgn':
            covert_parameters['ebno'] = torch.randint(-2, 8, (1,))
        else:
            covert_parameters['ebno'] = torch.randint(10, 20, (1,))
        # covert_parameters['ebno'] = channel_parameters['ebno']

        reset_grads(a_optimizer, b_optimizer, w_optimizer)

        z = torch.randn(covert_parameters['batch_size'], covert_parameters['n_channel'] * 2).to(device)
        alice_output = A(z, m)

        normal_labels = torch.ones(covert_parameters['batch_size'], 1).to(device)
        covert_labels = torch.zeros(covert_parameters['batch_size'], 1).to(device)

        willie_output = W(AE.channel(x, channel_parameters['channel_type'], ebno=covert_parameters['ebno']))
        willie_normal_loss = w_criterion(willie_output, normal_labels)
        willie_output = W(AE.channel(alice_output + x, channel_parameters['channel_type'], ebno=covert_parameters['ebno']))
        willie_covert_loss = w_criterion(willie_output, covert_labels)

        willie_loss = willie_normal_loss + willie_covert_loss
        willie_loss.backward()
        w_optimizer.step()

        reset_grads(a_optimizer, b_optimizer, w_optimizer)
        z = torch.randn(covert_parameters['batch_size'], covert_parameters['n_channel'] * 2).to(device)
        alice_output = A(z, m)
        x = x.detach()

        willie_output = W(AE.channel(alice_output + x, channel_parameters['channel_type'], ebno=covert_parameters['ebno']))
        bob_output = B(AE.channel(alice_output + x, channel_parameters['channel_type'], ebno=covert_parameters['ebno']))

        if covert_parameters['low_rate_multiplier'] is not None:
            alice_output = AE.channel(
                alice_output.reshape(-1, model_parameters['n_channel'] * 2) + x.reshape(-1, model_parameters[
                    'n_channel'] * 2), channel_parameters['channel_type'], ebno=covert_parameters['ebno'])
        else:
            alice_output = AE.channel(alice_output + x, channel_parameters['channel_type'], ebno=covert_parameters['ebno'])

        alice_bob_loss = ae_lambda * ae_criterion(AE.decoder_net(alice_output),
                                                  y) + willie_lambda * w_criterion(
            willie_output, torch.ones(covert_parameters['batch_size'], 1).to(device)) + bob_lambda * b_criterion(
            bob_output, m)

        alice_bob_loss.backward()
        a_optimizer.step()

        reset_grads(a_optimizer, b_optimizer, w_optimizer)
        z = torch.randn(covert_parameters['batch_size'], covert_parameters['n_channel'] * 2).to(device)
        alice_output = A(z, m)
        x = x.detach()

        bob_output = B(AE.channel(alice_output + x, channel_parameters['channel_type'], ebno=covert_parameters['ebno']))
        bob_loss = b_criterion(bob_output, m)
        bob_loss.backward()
        b_optimizer.step()

        return willie_loss, bob_loss, alice_bob_loss

    for epoch in range(num_epochs):
        A.train()
        B.train()
        W.train()

        if dl:
            for i, (x, y, m) in enumerate(dl):
                willie_loss, bob_loss, alice_bob_loss = train_batch(x, y, m)
        else:
            willie_loss, bob_loss, alice_bob_loss = train_batch(x, y, m)

        if epoch != 0 and epoch % 500 == 0:
            w_optimizer.param_groups[0]['lr'] /= 2
            b_optimizer.param_groups[0]['lr'] /= 2
            a_optimizer.param_groups[0]['lr'] /= 2

        if epoch % 10 == 0:
            with torch.no_grad():
                test_z = torch.randn(len(test_x), covert_parameters['n_channel'] * 2).to(device)

                A.eval()
                B.eval()
                W.eval()

                alice_output_test = A(test_z, test_m)

                willie_acc = discriminator_accuracy(
                    W(AE.channel(alice_output_test + test_x, channel_parameters['channel_type'])),
                    W(AE.channel(test_x, channel_parameters['channel_type'])))

                bob_acc = classifier_accuracy(
                    B(AE.channel(alice_output_test + test_x, channel_parameters['channel_type'])),
                    test_m)

                accs['willie'].append(round(willie_acc, 2))
                accs['bob'].append(round(bob_acc, 2))

                if covert_parameters['low_rate_multiplier'] is not None:
                    alice_output_test = AE.channel(
                        alice_output_test.reshape(-1, model_parameters['n_channel'] * 2) + test_x.reshape(-1,
                                                                                                            model_parameters[
                                                                                                                'n_channel'] * 2),
                        channel_parameters['channel_type'])
                else:
                    alice_output_test = AE.channel(alice_output_test + test_x, channel_parameters['channel_type'])
                # alice_output_test = AE.channel(test_x, channel_parameters['channel_type'])
                model_acc = classifier_accuracy(AE.decoder_net(alice_output_test), test_y)

                accs['autoencoder'].append(round(model_acc, 2))

                losses['willie'].append(willie_loss.item())
                losses['bob'].append(bob_loss.item())
                losses['alice'].append(alice_bob_loss.item())

            print(
                'Epoch[{}/{}], Alice Loss:{:.4f}, Willie Loss:{:.4f}, Willie Acc:{:.2f}, Bob Acc:{:.2f}, Model Acc:{:.2f}'.format(
                    epoch, num_epochs, alice_bob_loss.item(),
                    willie_loss.item(), willie_acc, bob_acc, model_acc))

    return accs, losses


def run_train(seed=covert_parameters['seed'], save=True):
    torch.manual_seed(seed)
    print('Device: ', 'CPU' if device == 'cpu' else 'GPU')
    models, datasets = initialize()
    A, B, E, AE = models

    if channel_parameters['channel_type'] == "awgn":
        ae_lambda, bob_lambda, willie_lambda = 0.4, covert_parameters['m'] * 0.1, 0.8
    else:
        ae_lambda, bob_lambda, willie_lambda = 0.4, covert_parameters['m'] * 0.2, 0.6

    accs, losses = train(models, datasets, lambdas=(ae_lambda, bob_lambda, willie_lambda))

    if save:
        torch.save(A.state_dict(), '../models/' + channel_parameters['channel_type'] + '/wireless_covert_alice(' + str(covert_parameters['n_channel']) + ',' + str(covert_parameters['k']) + ').pt')
        torch.save(B.state_dict(), '../models/' + channel_parameters['channel_type'] + '/wireless_covert_bob(' + str(covert_parameters['n_channel']) + ',' + str(covert_parameters['k']) + ').pt')
        torch.save(E.state_dict(), '../models/' + channel_parameters['channel_type'] + '/wireless_covert_willie(' + str(covert_parameters['n_channel']) + ',' + str(covert_parameters['k']) + ').pt')

    accuracies_chart(accs)


def evaluate(ebno):
    models, datasets = initialize()

    A, B, W, AE = models
    A.eval(), B.eval(), W.eval(), AE.eval()

    _, test_ds, m_ds = datasets
    test_x, test_y = test_ds
    m, test_m = m_ds

    A.load_state_dict(torch.load('../models/' + channel_parameters['channel_type'] + '/wireless_covert_alice(' + str(covert_parameters['n_channel']) + ',' + str(covert_parameters['k']) + ').pt'))
    B.load_state_dict(torch.load('../models/' + channel_parameters['channel_type'] + '/wireless_covert_bob(' + str(covert_parameters['n_channel']) + ',' + str(covert_parameters['k']) + ').pt'))
    W.load_state_dict(torch.load('../models/' + channel_parameters['channel_type'] + '/wireless_covert_willie(' + str(covert_parameters['n_channel']) + ',' + str(covert_parameters['k']) + ').pt'))

    test_z = torch.randn(len(test_x), covert_parameters['n_channel'] * 2).to(device)

    willie_acc = discriminator_accuracy(W(AE.channel(A(test_z, test_m) + test_x, channel_parameters['channel_type'], ebno=ebno)),
                                 W(AE.channel(test_x, channel_parameters['channel_type'], ebno=ebno)))

    bob_acc = classifier_accuracy(B(AE.channel(A(test_z, test_m) + test_x, channel_parameters['channel_type'], ebno=ebno)),
                           test_m)

    alice_output_test = A(test_z, test_m)
    if covert_parameters['low_rate_multiplier'] is not None:
        covert_signal_test = AE.channel(
            alice_output_test.reshape(-1, model_parameters['n_channel'] * 2) + test_x.reshape(-1, model_parameters[
                'n_channel'] * 2), channel_parameters['channel_type'], ebno=ebno)
    else:
        covert_signal_test = AE.channel(alice_output_test + test_x, channel_parameters['channel_type'], ebno=ebno)

    # normal_signal_test = AE.channel(test_x, channel_parameters['channel_type'], ebno=ebno)
    # ae_acc = classifier_accuracy(AE.decoder_net(normal_signal_test), test_y)
    ae_covert_acc = classifier_accuracy(AE.decoder_net(covert_signal_test), test_y)

    return ae_covert_acc, bob_acc, willie_acc


def run_eval(seed=covert_parameters['seed']):
    torch.manual_seed(seed)
    if channel_parameters['channel_type'] == 'awgn':
        ebno_range = list(frange(-4, 5, 1))
    else:
        ebno_range = list(frange(0, 22, 5))

    accs = {'autoencoder': [], 'autoencoder_covert': [], 'bob': [], 'willie': []}

    if channel_parameters['channel_type'] == 'awgn':
        accs['autoencoder'] = [0.14544921875, 0.09330078125, 0.0578515625, 0.02900390625, 0.01353515625, 0.00513671875, 0.00193359375, 0.00048828125, 7.8125e-05]
    else:
        accs['autoencoder'] = [0.1699609375, 0.05806640625, 0.01875, 0.00599609375, 0.0017578125]

    for i in range(0, len(ebno_range)):
        ae_covert_acc, bob_acc, willie_acc = evaluate(ebno=ebno_range[i])
        accs['autoencoder_covert'].append(1 - ae_covert_acc)
        accs['bob'].append(1 - bob_acc)
        accs['willie'].append(round(willie_acc, 2))

    with open('results/txt/accuracies(' + str(covert_parameters['n_channel']) + ',' + str(covert_parameters['k'])  + ')_' + channel_parameters['channel_type'] + '.txt', 'w') as f:
        for key, item in accs.items():
            f.write("%s:%s\n" % (key, item))

    for x in ['autoencoder', 'autoencoder_covert', 'bob', 'willie']:
        if x == 'willie':
            plot_label = 'Willie'
            plot_x = list(ebno_range)
            plot_y = accs[x]
            plot_x_new = np.linspace(min(plot_x), max(plot_x), 100)
            a_BSpline = make_interp_spline(plot_x, plot_y)
            plot_y = a_BSpline(plot_x_new)
            plot_x = plot_x_new
            plot_fmt = 'g-'
        elif x == 'bob':
            plot_label = 'Bob' + " (" + str(covert_parameters['n_channel']) + "," + str(covert_parameters['k']) + ")"
            plot_x = list(ebno_range)
            plot_y = accs[x]
            plot_fmt = '-ro'
        elif x == 'autoencoder_covert':
            plot_label = "Autoencoder" + " (" + str(model_parameters['n_channel']) + "," + str(
                model_parameters['k']) + ") - After Covert Communication"
            plot_x = list(ebno_range)
            plot_y = accs[x]
            plot_fmt = '-ko'
        elif x == 'autoencoder':
            plot_label = "Autoencoder" + " (" + str(model_parameters['n_channel']) + "," + str(
                model_parameters['k']) + ") - Before Covert Communication"
            plot_x = list(ebno_range)
            plot_y = accs[x]
            plot_fmt = '--bD'


        plt.plot(plot_x, plot_y, plot_fmt,
                 label=plot_label)

        if x == 'autoencoder':
            continue

        plt.xlim(ebno_range[0], ebno_range[-1])

        if x == 'autoencoder_covert' or x == 'bob':
            plt.yscale('log')

        plt.xlabel("$E_{b}/N_{0}$ (dB)")
        if x == 'autoencoder_covert' or x == 'bob':
            plt.ylabel('Block Error Rate')
        else:
            plt.ylabel('Accuracy')

        plt.grid()
        plt.legend(loc="upper right", ncol=1)

        plt.show()

def eval_rate_change():
    rates = [1, 2, 4]
    accs = {'autoencoder': [], 'autoencoder_covert': [], 'bob': [], 'willie': []}

    chart_conf = {
        'autoencoder': {
            'c': ['black'],
            'l':  [(0, (5, 7))],
            'm': [None]
        },
        'autoencoder_covert': {
            'c': ['blue', 'tab:olive', 'tab:orange'],
            'l': ['solid', 'dashdot', 'dotted'],
            'm': ['v', '<', 'd']
        },
        'bob': {
            'c': ['red', 'tab:olive', 'tab:orange'],
            'l': ['solid', 'dashdot', 'dotted'],
            'm': ['v', '<', 'd']
        },
        'willie': {
            'c': ['green', 'tab:olive', 'tab:orange'],
            'l': ['solid', 'dashdot', 'dotted'],
            'm': [None, None, None]
        }
    }

    if channel_parameters['channel_type'] == 'awgn':
        ebno_range = list(frange(-4, 5, 1))
    else:
        ebno_range = list(frange(0, 22, 5))

    for rate in rates:
        f = open('results/txt/accuracies(' + str(covert_parameters['n_channel']) + ',' + str(rate) + ')_' + channel_parameters['channel_type'] + '.txt')
        for line in f.readlines():
            acc = line.split(':')
            if acc[0] == 'autoencoder':
                accs[acc[0]] = eval(acc[1].strip())
            else:
                accs[acc[0]].append(eval(acc[1].strip()))

    for x in accs:
        for i in range(len(rates)):
            if x == 'willie':
                plot_label = 'Willie' + " (" + str(covert_parameters['n_channel']) + "," + str(
                    str(rates[i])) + ")"
                plot_x = list(ebno_range)
                plot_y = accs[x][i]
                plot_x_new = np.linspace(min(plot_x), max(plot_x), 100)
                a_BSpline = make_interp_spline(plot_x, plot_y)
                plot_y = a_BSpline(plot_x_new)
                plot_x = plot_x_new
            elif x == 'bob':
                plot_label = "Bob (" + str(covert_parameters['n_channel']) + "," + str(
                    rates[i]) + ")"
                plot_x = list(ebno_range)
                plot_y = accs[x][i]
            elif x == 'autoencoder_covert':
                plot_label = "Autoencoder (" + str(model_parameters['n_channel']) + "," + str(
                    model_parameters['k']) + ") — Alice (" + str(model_parameters['n_channel']) + "," + str(rates[i]) + ")"
                plot_x = list(ebno_range)
                plot_y = accs[x][i]
            elif x == 'autoencoder':
                plot_label = "Autoencoder (" + str(model_parameters['n_channel']) + "," + str(
                    model_parameters['k']) + ")"
                plot_x = list(ebno_range)
                plot_y = accs[x]

            plt.plot(plot_x, plot_y, linestyle=chart_conf[x]['l'][i], marker=chart_conf[x]['m'][i], color=chart_conf[x]['c'][i],
                             label=plot_label, clip_on=False)

            if x == 'autoencoder':
                break

            if rates[i] == 4:
                plt.xlim(ebno_range[0], ebno_range[-1])

                if x == 'autoencoder_covert' or x == 'bob':
                    plt.yscale('log')
                    plt.legend(loc="lower left", ncol=1)
                else:
                    plt.legend(loc="upper left", ncol=1)

                plt.xlabel("$E_{b}/N_{0}$ (dB)")
                if x == 'autoencoder_covert' or x == 'bob':
                    plt.ylabel('Block Error Rate')
                else:
                    plt.ylabel('Accuracy')

                plt.grid()



                if x == 'autoencoder_covert':
                    plt.savefig("results/png/covert_autoencoder_bler_" + channel_parameters['channel_type'] + ".png")
                    plt.savefig("results/eps/covert_autoencoder_bler_" + channel_parameters['channel_type'] + ".eps",
                                format="eps")
                elif x == 'bob':
                    plt.savefig(
                        "results/png/bob_bler_" +
                        channel_parameters['channel_type'] + ".png")
                    plt.savefig(
                        "results/eps/bob_bler_" +
                        channel_parameters['channel_type'] + ".eps", format="eps")
                else:
                    plt.savefig(
                        "results/png/willie_accuracy_" +
                        channel_parameters['channel_type'] + ".png")
                    plt.savefig(
                        "results/eps/willie_accuracy_" +
                        channel_parameters['channel_type'] + ".eps", format="eps")

                plt.show()
        
def eval_constellation(ebno=channel_parameters['ebno'], points=500):
    models, _ = initialize()

    A, B, W, AE = models

    A.load_state_dict(torch.load('../models/' + channel_parameters['channel_type'] + '/wireless_covert_alice(' + str(covert_parameters['n_channel']) + ',' + str(covert_parameters['k']) + ').pt'))
    B.load_state_dict(torch.load('../models/' + channel_parameters['channel_type'] + '/wireless_covert_bob(' + str(covert_parameters['n_channel']) + ',' + str(covert_parameters['k']) + ').pt'))
    W.load_state_dict(torch.load('../models/' + channel_parameters['channel_type'] + '/wireless_covert_willie(' + str(covert_parameters['n_channel']) + ',' + str(covert_parameters['k']) + ').pt'))

    A.eval(), B.eval(), W.eval(), AE.eval()

    for i in range(model_parameters['m']):
        with torch.no_grad():
            test_z = torch.randn(points, covert_parameters['n_channel'] * 2).to(device)
            test_m = (torch.rand(points) * covert_parameters['m']).long().to(device)
            s = torch.sparse.torch.eye(model_parameters['m']).index_select(dim=0, index=torch.tensor(i))
            test_x = torch.zeros(points, model_parameters['n_channel'] * 2)
            test_x[:] = AE.encoder(s.cuda())[0].detach()
            test_x = test_x.cuda()

            encoder_signal = [test_x[0]]
            random_state = torch.random.get_rng_state()
            covert_signal = AE.channel(test_x + A(test_z, test_m), channel=channel_parameters['channel_type'], ebno=ebno)
            plt = plot_constellation([(covert_signal, 'x', 'tab:red', 2), (encoder_signal, 'o', 'k', 30)])
            plt.xlabel("In-phase")
            plt.ylabel('Quadrature')
            plt.savefig('results/eps/constellation/' + channel_parameters['channel_type'] + '_covert_' + str(i) + '.eps',
                        format='eps')
            plt.savefig(
                'results/eps/constellation/' + channel_parameters['channel_type'] + '_covert_' + str(i) + '.png',
                format='png')
            plt.close()

            torch.random.set_rng_state(random_state)
            normal_signal = AE.channel(test_x, channel=channel_parameters['channel_type'], ebno=ebno)
            plt = plot_constellation([(normal_signal, 'x', 'tab:green', 2), (encoder_signal, 'o', 'k', 30)])
            plt.xlabel("In-phase")
            plt.ylabel('Quadrature')
            plt.savefig('results/eps/constellation/' + channel_parameters['channel_type'] + '_normal_' + str(i) + '.eps',
                        format='eps')
            plt.savefig(
                'results/eps/constellation/' + channel_parameters['channel_type'] + '_normal_' + str(i) + '.png',
                format='png')
            plt.close()


'''
    Run) training or evaluation
'''
# run_train()
# run_eval()

eval_constellation()
# eval_rate_change()