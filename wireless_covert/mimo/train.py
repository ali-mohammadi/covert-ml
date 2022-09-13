import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np

from wireless_autoencoder.mimo.parameters import model_parameters
from wireless_covert.mimo.parameters import covert_parameters
from shared_parameters import channel_parameters, system_parameters
from wireless_autoencoder.mimo.model import Wireless_Autoencoder
from models import Alice, Bob, Willie, device
from helpers import frange, accuracies_chart, plot_constellation, reset_grads, discriminator_accuracy, \
    classifier_accuracy

assert system_parameters['system_type'] == 'mimo'


def initialize():
    AEs = []
    with torch.no_grad():
        xs = []
        ys = []
        autoencoder_ds = torch.load(
            '../../data/' + system_parameters['system_type'] + '/' + channel_parameters[
                'channel_type'] + '/wireless_autoencoder(' + str(
                model_parameters['n_channel']) + ',' + str(model_parameters['k']) + ')_train.pt')
        for i in range(model_parameters['n_user']):
            AE = Wireless_Autoencoder()
            AE.index = i + 1
            AE.load()
            AE.eval()
            AEs.append(AE)
            idx = torch.randperm(len(autoencoder_ds))
            xs.append(AE.encoder(autoencoder_ds[idx][0]))
            ys.append(autoencoder_ds[idx][1])
            # xs.append(AE.encoder(autoencoder_ds[:][0]))
            # ys.append(autoencoder_ds[:][1])
        if channel_parameters['channel_type'] == 'awgn':
            x = torch.sum(torch.stack(xs), 0)
        else:
            x = torch.stack(xs)
            x.transpose_(0, 1)
        y = torch.stack(ys)

        test_xs = []
        test_ys = []
        autoencoder_tds = torch.load(
            '../../data/' + system_parameters['system_type'] + '/' + channel_parameters[
                'channel_type'] + '/wireless_autoencoder(' + str(model_parameters['n_channel']) + ',' + str(
                model_parameters['k']) + ')_test.pt')
        for i in range(model_parameters['n_user']):
            idx = torch.randperm(len(autoencoder_ds))
            test_xs.append(AEs[i].encoder(autoencoder_tds[idx][0]))
            test_ys.append(autoencoder_tds[idx][1])
        if channel_parameters['channel_type'] == 'awgn':
            test_x = torch.sum(torch.stack(test_xs), 0)
        else:
            test_x = torch.stack(test_xs)
            test_x.transpose_(0, 1)
        test_y = torch.stack(test_ys)

    if covert_parameters['low_rate_multiplier'] is not None:
        pass  # todo: To be implemented for MIMO systems.

    x = x.to(device)
    y = y.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)

    m = (torch.rand(len(x)) * covert_parameters['m']).long().to(device)
    test_m = (torch.rand(len(test_x)) * covert_parameters['m']).long().to(device)

    W = Willie().to(device)
    A = Alice().to(device)
    B = Bob().to(device)
    for AE in AEs:
        AE.to(device)

    return (A, B, W, AEs), ((x, y), (test_x, test_y), (m, test_m))


def train(models, datasets, num_epochs=covert_parameters['num_epochs'],
          learning_rate=covert_parameters['learning_rate'],
          optim_fn=torch.optim.Adam, lambdas=(1, 1, 1)):
    A, B, W, AEs = models
    AE = AEs[0]
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
    losses = {'willie': [], 'bob': [], 'alice': [], 'ae': []}

    def train_batch(x, y, m):
        '''
            Randomly switching noise power to be robust against different noise levels
        '''
        if channel_parameters['channel_type'] == 'awgn':
            covert_parameters['channel_k'] = None
            covert_parameters['ebno'] = torch.randint(-2, 8, (1,))
        elif channel_parameters['channel_type'] == 'rician':
            covert_parameters['channel_k'] = channel_parameters['channel_k']
            # covert_parameters['channel_k'] = torch.randint(1, 5, (1,)).numpy()[0]
            covert_parameters['ebno'] = torch.randint(10, 20, (1,))
        else:
            covert_parameters['channel_k'] = None
            covert_parameters['ebno'] = torch.randint(5, 25, (1,))
            # covert_parameters['ebno'] = channel_parameters['ebno']

        '''
            Train Willie
        '''
        reset_grads(a_optimizer, b_optimizer, w_optimizer)
        if channel_parameters['channel_type'] == 'rayleigh':
            h = torch.randn((x.size()[0], model_parameters['n_user'], model_parameters['n_user']),
                            dtype=torch.cfloat).to(device)

        z = torch.randn(covert_parameters['batch_size'], covert_parameters['n_channel'] * 2).to(device)
        alice_output = A(z, m)

        normal_labels = torch.ones(covert_parameters['batch_size'], 1).to(device)
        covert_labels = torch.zeros(covert_parameters['batch_size'], 1).to(device)

        if channel_parameters['channel_type'] == 'awgn':
            willie_output = W(AE.channel(x, channel_parameters['channel_type'], ebno=covert_parameters['ebno'],
                                         k=covert_parameters['channel_k']))
        else:
            willie_output = W(AE.channel(x, channel_parameters['channel_type'], ebno=covert_parameters['ebno'],
                                         k=covert_parameters['channel_k'], h=h)[:, 0, :])

        willie_normal_loss = w_criterion(willie_output, normal_labels)

        if channel_parameters['channel_type'] == 'awgn':
            willie_output = W(
                AE.channel(alice_output + x, channel_parameters['channel_type'], ebno=covert_parameters['ebno'],
                           k=covert_parameters['channel_k']))
        else:
            alice_output = alice_output.unsqueeze(1)
            alice_output = torch.concat((alice_output, torch.zeros(x.size()[0], x.size()[1] - 1, x.size()[2]).to(device)), dim=1)
            willie_output = W(
                AE.channel(alice_output + x, channel_parameters['channel_type'], ebno=covert_parameters['ebno'],
                           k=covert_parameters['channel_k'], h=h)[:, 0, :])

        willie_covert_loss = w_criterion(willie_output, covert_labels)

        willie_loss = willie_normal_loss + willie_covert_loss
        willie_loss.backward()
        w_optimizer.step()

        '''
            Train Alice
        '''
        reset_grads(a_optimizer, b_optimizer, w_optimizer)
        z = torch.randn(covert_parameters['batch_size'], covert_parameters['n_channel'] * 2).to(device)
        alice_output = A(z, m)
        x = x.detach()

        if channel_parameters['channel_type'] == 'awgn':
            willie_output = W(
                AE.channel(alice_output + x, channel_parameters['channel_type'],
                           ebno=covert_parameters['ebno'],
                           k=covert_parameters['channel_k']))
            bob_output = B(
                AE.channel(alice_output + x, channel_parameters['channel_type'], ebno=covert_parameters['ebno'],
                           k=covert_parameters['channel_k']))
        else:
            alice_output = alice_output.unsqueeze(1)
            alice_output = torch.concat((alice_output, torch.zeros(x.size()[0], x.size()[1] - 1, x.size()[2]).to(device)), dim=1)
            willie_output = W(
                AE.channel(alice_output + x, channel_parameters['channel_type'], ebno=covert_parameters['ebno'],
                           k=covert_parameters['channel_k'], h=h)[:, 0, :])
            bob_output = B(
                AE.channel(alice_output + x, channel_parameters['channel_type'], ebno=covert_parameters['ebno'],
                           k=covert_parameters['channel_k'], h=h)[:, 0, :])

        if covert_parameters['low_rate_multiplier'] is not None:
            pass  # todo: To be implemented.
        else:
            if channel_parameters['channel_type'] == 'awgn':
                alice_output = AE.channel(alice_output + x, channel_parameters['channel_type'],
                                          ebno=covert_parameters['ebno'], k=covert_parameters['channel_k'])
            else:
                alice_output = AE.channel(alice_output + x, channel_parameters['channel_type'],
                                          ebno=covert_parameters['ebno'], k=covert_parameters['channel_k'], h=h)

        if channel_parameters['channel_type'] == 'awgn':
            ae_loss = ae_criterion(AE.decoder_net(alice_output), y)
        else:
            ae_loss = 0
            for i in range(model_parameters['n_user']):
                ae_loss += ae_criterion(AEs[i].decoder_net(alice_output, torch.view_as_real(h).view(-1, 2 *
                                                                                                    model_parameters[
                                                                                                        'n_user'] *
                                                                                                    model_parameters[
                                                                                                        'n_user'])),
                                        y[i])

        alice_bob_loss = ae_lambda * ae_loss + willie_lambda * w_criterion(
            willie_output, torch.ones(covert_parameters['batch_size'], 1).to(device)) + bob_lambda * b_criterion(
            bob_output, m)

        alice_bob_loss.backward()
        a_optimizer.step()

        '''
            Train Bob
        '''
        reset_grads(a_optimizer, b_optimizer, w_optimizer)
        z = torch.randn(covert_parameters['batch_size'], covert_parameters['n_channel'] * 2).to(device)
        alice_output = A(z, m)
        x = x.detach()

        if channel_parameters['channel_type'] == 'awgn':
            bob_output = B(
                AE.channel(alice_output + x, channel_parameters['channel_type'], ebno=covert_parameters['ebno'],
                           k=covert_parameters['channel_k']))
        else:
            alice_output = alice_output.unsqueeze(1)
            alice_output = torch.concat((alice_output, torch.zeros(x.size()[0], x.size()[1] - 1, x.size()[2]).to(device)), dim=1)
            bob_output = B(
                AE.channel(alice_output + x, channel_parameters['channel_type'], ebno=covert_parameters['ebno'],
                           k=covert_parameters['channel_k'], h=h)[:, 0, :])
        bob_loss = b_criterion(bob_output, m)
        bob_loss.backward()
        b_optimizer.step()

        return willie_loss, bob_loss, alice_bob_loss, ae_loss

    for epoch in range(num_epochs):
        A.train()
        B.train()
        W.train()

        if dl:
            for i, (x, y, m) in enumerate(dl):
                willie_loss, bob_loss, alice_bob_loss, ae_loss = train_batch(x, y, m)
        else:
            willie_loss, bob_loss, alice_bob_loss, ae_loss = train_batch(x, y, m)

        if epoch != 0 and epoch % (num_epochs / 10) == 0:
            w_optimizer.param_groups[0]['lr'] /= 2
            b_optimizer.param_groups[0]['lr'] /= 2
            a_optimizer.param_groups[0]['lr'] /= 2

        if epoch % 10 == 0:
            with torch.no_grad():
                test_z = torch.randn(len(test_x), covert_parameters['n_channel'] * 2).to(device)
                if channel_parameters['channel_type'] == 'rayleigh':
                    h = torch.randn((test_x.size()[0], model_parameters['n_user'], model_parameters['n_user']),
                                    dtype=torch.cfloat).to(device)
                A.eval()
                B.eval()
                W.eval()

                alice_output_test = A(test_z, test_m)

                if channel_parameters['channel_type'] == 'awgn':
                    willie_acc = discriminator_accuracy(
                        W(AE.channel(alice_output_test + test_x, channel_parameters['channel_type'])),
                        W(AE.channel(test_x, channel_parameters['channel_type'])))
                else:
                    alice_output_test = alice_output_test.unsqueeze(1)
                    alice_output_test = torch.concat(
                        (alice_output_test, torch.zeros(x.size()[0], x.size()[1] - 1, x.size()[2]).to(device)), dim=1)
                    willie_acc = discriminator_accuracy(
                        W(AE.channel(alice_output_test + test_x, channel_parameters['channel_type'], h=h)[:, 0, :]),
                        W(AE.channel(test_x, channel_parameters['channel_type'], h=h)[:, 0, :]))

                if channel_parameters['channel_type'] == 'awgn':
                    bob_acc = classifier_accuracy(
                        B(AE.channel(alice_output_test + test_x, channel_parameters['channel_type'])),
                        test_m)
                else:
                    bob_acc = classifier_accuracy(
                        B(AE.channel(alice_output_test + test_x, channel_parameters['channel_type'], h=h)[:, 0, :]),
                        test_m)

                accs['willie'].append(round(willie_acc, 2))
                accs['bob'].append(round(bob_acc, 2))

                if covert_parameters['low_rate_multiplier'] is not None:
                    pass  # todo: To be implemented for MIMO systems.
                else:
                    if channel_parameters['channel_type'] == 'awgn':
                        alice_output_test = AE.channel(alice_output_test + test_x, channel_parameters['channel_type'])
                    else:
                        alice_output_test = AE.channel(alice_output_test + test_x, channel_parameters['channel_type'],
                                                       h=h)
                # alice_output_test = AE.channel(test_x, channel_parameters['channel_type'])

                model_accs = []
                for i in range(model_parameters['n_user']):
                    if channel_parameters['channel_type'] == 'awgn':
                        model_acc = classifier_accuracy(AEs[i].decoder_net(alice_output_test), test_y[i])
                    else:
                        model_acc = classifier_accuracy(AEs[i].decoder_net(alice_output_test,
                                                                           torch.view_as_real(h).view(-1, 2 *
                                                                                                      model_parameters[
                                                                                                          'n_user'] *
                                                                                                      model_parameters[
                                                                                                          'n_user'])),
                                                        test_y[i])
                    model_accs.append(model_acc)
                model_acc = sum(model_accs) / model_parameters['n_user']
                accs['autoencoder'].append(round(model_acc, 2))

                losses['willie'].append(willie_loss.item())
                losses['bob'].append(bob_loss.item())
                losses['alice'].append(alice_bob_loss.item())
                losses['ae'].append(ae_loss.item())

            print(
                'Epoch[{}/{}], Alice Loss:{:.4f}, Bob Loss: {:.4f}, Willie Loss:{:.4f}, AE Loss:{:.4f}, Willie Acc:{:.2f}, Bob Acc:{:.2f}, Model Acc:{:.2f}'.format(
                    epoch, num_epochs, alice_bob_loss.item(), bob_loss.item(),
                    willie_loss.item(), ae_loss.item(), willie_acc, bob_acc, model_acc))

    return accs, losses


def run_train(seed=covert_parameters['seed'], save=True):
    torch.manual_seed(seed)
    print('Device: ', 'CPU' if device == 'cpu' else 'GPU')
    models, datasets = initialize()
    A, B, E, AE = models

    if channel_parameters['channel_type'] == "awgn":
        ae_lambda, bob_lambda, willie_lambda = 0.4, covert_parameters['m'] * 0.1, 0.8
    else:
        if model_parameters['n_user'] < 4:
            ae_lambda, bob_lambda, willie_lambda = 0.4, covert_parameters['m'] * model_parameters['n_user'] * 0.1, 0.6
        else:
            ae_lambda, bob_lambda, willie_lambda = 0.4, covert_parameters['m'] * model_parameters['n_user'] * 0.2, 0.6
        # ae_lambda, bob_lambda, willie_lambda = 0.6, covert_parameters['m'] * 0.4, 0.6
        # ae_lambda, bob_lambda, willie_lambda = 0, 1, 0

    accs, losses = train(models, datasets, lambdas=(ae_lambda, bob_lambda, willie_lambda))

    if save:
        torch.save(A.state_dict(), '../../models/' + system_parameters['system_type'] + '/' + channel_parameters[
            'channel_type'] + '/wireless_covert_alice(' + str(covert_parameters['n_channel']) + ',' + str(
            covert_parameters['k']) + ').pt')
        torch.save(B.state_dict(), '../../models/' + system_parameters['system_type'] + '/' + channel_parameters[
            'channel_type'] + '/wireless_covert_bob(' + str(covert_parameters['n_channel']) + ',' + str(
            covert_parameters['k']) + ').pt')
        torch.save(E.state_dict(), '../../models/' + system_parameters['system_type'] + '/' + channel_parameters[
            'channel_type'] + '/wireless_covert_willie(' + str(covert_parameters['n_channel']) + ',' + str(
            covert_parameters['k']) + ').pt')

    accuracies_chart(accs)


def evaluate(ebno):
    models, datasets = initialize()

    A, B, W, AEs = models
    A.eval(), B.eval(), W.eval()
    for AE in AEs:
        AE.eval()

    _, test_ds, m_ds = datasets
    test_x, test_y = test_ds
    m, test_m = m_ds

    h = torch.randn((test_x.size()[0], model_parameters['n_user'], model_parameters['n_user']),
                    dtype=torch.cfloat).to(device)

    A.load_state_dict(torch.load('../../models/' + system_parameters['system_type'] + '/' + channel_parameters[
        'channel_type'] + '/wireless_covert_alice(' + str(covert_parameters['n_channel']) + ',' + str(
        covert_parameters['k']) + ').pt'))
    B.load_state_dict(torch.load('../../models/' + system_parameters['system_type'] + '/' + channel_parameters[
        'channel_type'] + '/wireless_covert_bob(' + str(covert_parameters['n_channel']) + ',' + str(
        covert_parameters['k']) + ').pt'))
    W.load_state_dict(torch.load('../../models/' + system_parameters['system_type'] + '/' + channel_parameters[
        'channel_type'] + '/wireless_covert_willie(' + str(covert_parameters['n_channel']) + ',' + str(
        covert_parameters['k']) + ').pt'))

    test_z = torch.randn(len(test_x), covert_parameters['n_channel'] * 2).to(device)

    if channel_parameters['channel_type'] == 'awgn':
        alice_output_test = A(test_z, test_m)
        willie_acc = discriminator_accuracy(
            W(AE.channel(alice_output_test + test_x, channel_parameters['channel_type'], ebno=ebno)),
            W(AE.channel(test_x, channel_parameters['channel_type'], ebno=ebno)))

        bob_acc = classifier_accuracy(
            B(AE.channel(alice_output_test + test_x, channel_parameters['channel_type'], ebno=ebno)),
            test_m)
    else:
        alice_output_test = A(test_z, test_m).unsqueeze(1)
        alice_output_test = torch.concat((alice_output_test, torch.zeros(test_x.size()[0], test_x.size()[1] - 1, test_x.size()[2]).to(device)), dim=1)
        willie_acc = discriminator_accuracy(
            W(AEs[0].channel(alice_output_test + test_x, channel_parameters['channel_type'], ebno=ebno, h=h)[:,
              0, :]),
            W(AEs[0].channel(test_x, channel_parameters['channel_type'], ebno=ebno, h=h)[:, 0, :]))

        bob_acc = classifier_accuracy(
            B(AEs[0].channel(alice_output_test + test_x, channel_parameters['channel_type'], ebno=ebno, h=h)[:,
              0, :]),
            test_m)

    if covert_parameters['low_rate_multiplier'] is not None:
        pass  # todo: To be implemented for MIMO systems.
    else:
        if channel_parameters['channel_type'] == 'awgn':
            covert_signal_test = AE.channel(alice_output_test + test_x, channel_parameters['channel_type'], ebno=ebno)
        else:
            covert_signal_test = AEs[0].channel(alice_output_test + test_x, channel_parameters['channel_type'],
                                            ebno=ebno, h=h)

    # normal_signal_test = AEs[0].channel(test_x, channel_parameters['channel_type'],
    #                                     ebno=ebno, h=h)
    # ae_acc = classifier_accuracy(AE.decoder_net(normal_signal_test), test_y)
    ae_covert_accs = []
    for i in range(model_parameters['n_user']):
        if channel_parameters['channel_type'] == 'awgn':
            ae_covert_accs.append(classifier_accuracy(AEs[i].decoder_net(covert_signal_test), test_y[i]))
        else:
            # covert_signal_test = AEs[i].channel(test_x, channel_parameters['channel_type'],
            #                                                                         ebno=ebno, h=h)
            ae_covert_accs.append(classifier_accuracy(
                AEs[i].decoder_net(covert_signal_test, torch.view_as_real(h).view(-1, 2 *
                                                                                  model_parameters[
                                                                                      'n_user'] *
                                                                                  model_parameters[
                                                                                      'n_user'])), test_y[i]))
            # break
    # ae_covert_acc = ae_covert_acc / model_parameters['n_user']

    return ae_covert_accs, bob_acc, willie_acc


def run_eval(seed=covert_parameters['seed']):
    torch.manual_seed(seed)
    if channel_parameters['channel_type'] == 'awgn':
        ebno_range = list(frange(-4, 5, 1))
    elif channel_parameters['channel_type'] == 'rician':
        ebno_range = list(frange(-5, 22, 5))
    else:
        ebno_range = list(frange(0, 22, 5))

    accs = {'autoencoder': [], 'autoencoder_covert': [], 'bob': [], 'willie': []}
    accs['autoencoder_covert'] = [[] for _ in range(model_parameters['n_user'])]
    accs['autoencoder'] = [[] for _ in range(model_parameters['n_user'])]

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    if channel_parameters['channel_type'] == 'awgn':
        accs['autoencoder'] = [0.14544921875, 0.09330078125, 0.0578515625, 0.02900390625, 0.01353515625, 0.00513671875,
                               0.00193359375, 0.00048828125, 7.8125e-05]
    elif channel_parameters['channel_type'] == 'rician':
        if channel_parameters['channel_k'] == 1:
            accs['autoencoder'] = [0.67341796875, 0.421640625, 0.19515625, 0.06857421875, 0.02060546875, 0.00626953125]
        if channel_parameters['channel_k'] == 2:
            accs['autoencoder'] = [0.44544921875, 0.188359375, 0.05201171875, 0.0108203125, 0.00216796875,
                                   0.00056640625]
        if channel_parameters['channel_k'] == 4:
            accs['autoencoder'] = [0.17064453125, 0.03505859375, 0.00390625, 0.000234375, 0.0, 0.0]
        if channel_parameters['channel_k'] == 8:
            accs['autoencoder'] = [0.03015625, 0.00234375, 7.8125e-05, 0.0, 0.0, 0.0]
    else:
        if model_parameters['n_user'] == 2:
            accs['autoencoder'][0] = [0.0605078125, 0.00951171875, 0.00123046875, 0.0001953125, 9.765625e-05]
            accs['autoencoder'][1] = [0.08046875, 0.01361328125, 0.00169921875, 0.000234375, 0.0001953125]
        if model_parameters['n_user'] == 3:
            accs['autoencoder'][0] = [0.0321875, 0.0029296875, 0.000390625, 5.859375e-05, 3.90625e-05]
            accs['autoencoder'][1] = [0.02875, 0.0025390625, 0.00029296875, 7.8125e-05, 0.0]
            accs['autoencoder'][2] = [0.02580078125, 0.0025, 0.0003515625, 9.765625e-05, 0.0001171875]
        if model_parameters['n_user'] == 4:
            accs['autoencoder'][0] = [0.01150390625, 0.00103515625, 0.0001953125, 0.00013671875, 0.00013671875]
            accs['autoencoder'][1] = [0.012578125, 0.00154296875, 0.0005859375, 0.00029296875, 0.000234375]
            accs['autoencoder'][2] = [0.01283203125, 0.001171875, 0.00017578125, 9.765625e-05, 0.00015625]
            accs['autoencoder'][3] = [0.013046875, 0.001328125, 0.0001953125, 0.00013671875, 9.765625e-05]

    for i in range(0, len(ebno_range)):
        ae_covert_accs, bob_acc, willie_acc = evaluate(ebno=ebno_range[i])
        for i in range(model_parameters['n_user']):
            accs['autoencoder_covert'][i].append(1 - ae_covert_accs[i])
        accs['bob'].append(1 - bob_acc)
        accs['willie'].append(round(willie_acc, 2))

    with open(
            'results/txt/accuracies(' + str(covert_parameters['n_channel']) + ',' + str(covert_parameters['k']) + ')_' +
            channel_parameters['channel_type'] + '.txt', 'w') as f:
        for key, item in accs.items():
            if key == 'autoencoder_covert':
                for i in range(model_parameters['n_user']):
                    f.write("%s%s:%s\n" % (key, str(i), item[i]))
            else:
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
            for i in range(model_parameters['n_user']):
                plot_label = "AE" + str(i + 1) + " (" + str(model_parameters['n_channel']) + "," + str(
                    model_parameters['k']) + ") - After Covert Communication"
                plot_x = list(ebno_range[0:-1])
                plot_y = accs[x][i][0:-1]
                # if channel_parameters['channel_k'] == 4:
                #     plot_y[-1] = 0.0
                plot_fmt = '-o'
                plot_color = colors[i]
                plt.plot(plot_x, plot_y, plot_fmt, color=plot_color,
                         label=plot_label)
        elif x == 'autoencoder':
            for i in range(model_parameters['n_user']):
                plot_label = "AE" + str(i + 1) + " (" + str(model_parameters['n_channel']) + "," + str(
                    model_parameters['k']) + ") - Before Covert Communication"
                plot_x = list(ebno_range[0:-1])
                plot_y = accs[x][i][0:-1]
                plot_fmt = '--D'
                plot_color = colors[i]
                plt.plot(plot_x, plot_y, plot_fmt, color=plot_color,
                         label=plot_label)

        if x == 'autoencoder':
            continue

        if x != 'autoencoder_covert' and x != 'autoencoder':
            plt.plot(plot_x, plot_y, plot_fmt,
                     label=plot_label)

        plt.xlim(ebno_range[0], ebno_range[-1])

        if x == 'autoencoder_covert' or x == 'bob':
            plt.yscale('log')

        label_fontsize = 16
        label_fontsize = 12
        plt.xlabel("$E_{b}/N_{0}$ (dB)", fontsize=label_fontsize)
        if x == 'autoencoder_covert' or x == 'bob':
            plt.ylabel('Block Error Rate', fontsize=label_fontsize)
        else:
            plt.ylabel('Accuracy', fontsize=label_fontsize)

        tick_fontsize = 14
        tick_fontsize = 10
        plt.xticks(fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        plt.grid()
        plt.legend(loc="upper right", ncol=1, fontsize=label_fontsize)
        plt.tight_layout()
        plt.show()


def eval_rate_change():
    rates = [1, 2, 4]
    accs = {'autoencoder': [], 'autoencoder_covert': [], 'bob': [], 'willie': []}

    chart_conf = {
        'autoencoder': {
            'c': ['black'],
            'l': [(0, (5, 7))],
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
    elif channel_parameters['channel_type'] == 'rician':
        ebno_range = list(frange(-5, 22, 5))
    else:
        ebno_range = list(frange(0, 22, 5))

    for rate in rates:
        f = open('results/txt/accuracies(' + str(covert_parameters['n_channel']) + ',' + str(rate) + ')_' +
                 channel_parameters['channel_type'] + '.txt')
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
                plot_label = "AE (" + str(model_parameters['n_channel']) + "," + str(
                    model_parameters['k']) + ") — Alice (" + str(model_parameters['n_channel']) + "," + str(
                    rates[i]) + ")"
                plot_x = list(ebno_range)
                plot_y = accs[x][i]
            elif x == 'autoencoder':
                plot_label = "AE (" + str(model_parameters['n_channel']) + "," + str(
                    model_parameters['k']) + ")"
                plot_x = list(ebno_range)
                plot_y = accs[x]

            plt.plot(plot_x, plot_y, linestyle=chart_conf[x]['l'][i], marker=chart_conf[x]['m'][i],
                     color=chart_conf[x]['c'][i],
                     label=plot_label, clip_on=False)

            if x == 'autoencoder':
                break

            if rates[i] == 4:
                plt.xlim(ebno_range[0], ebno_range[-1])

                if x == 'autoencoder_covert' or x == 'bob':
                    plt.yscale('log')
                    plt.legend(loc="lower left", ncol=1, fontsize=16)
                else:
                    plt.legend(loc="upper left", ncol=1, fontsize=16)

                plt.xlabel("$E_{b}/N_{0}$ (dB)", fontsize=16)
                if x == 'autoencoder_covert' or x == 'bob':
                    plt.ylabel('Block Error Rate', fontsize=16)
                else:
                    plt.ylabel('Accuracy', fontsize=16)

                plt.grid()
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.tight_layout()

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


def eval_constellation(ebno=channel_parameters['ebno'], points=500, s=None, save=True):
    models, _ = initialize()

    A, B, W, AE = models

    A.load_state_dict(torch.load('../models/' + channel_parameters['channel_type'] + '/wireless_covert_alice(' + str(
        covert_parameters['n_channel']) + ',' + str(covert_parameters['k']) + ').pt'))
    B.load_state_dict(torch.load('../models/' + channel_parameters['channel_type'] + '/wireless_covert_bob(' + str(
        covert_parameters['n_channel']) + ',' + str(covert_parameters['k']) + ').pt'))
    W.load_state_dict(torch.load('../models/' + channel_parameters['channel_type'] + '/wireless_covert_willie(' + str(
        covert_parameters['n_channel']) + ',' + str(covert_parameters['k']) + ').pt'))

    A.eval(), B.eval(), W.eval(), AE.eval()

    for i in range(0, model_parameters['m']):
        if s is not None:
            i = s
        with torch.no_grad():
            test_z = torch.randn(points, covert_parameters['n_channel'] * 2).to(device)
            test_m = (torch.rand(points) * covert_parameters['m']).long().to(device)
            s = torch.sparse.torch.eye(model_parameters['m']).index_select(dim=0, index=torch.tensor(i))
            test_x = torch.zeros(points, model_parameters['n_channel'] * 2)
            test_x[:] = AE.encoder(s.cuda())[0].detach()
            test_x = test_x.cuda()

            encoder_signal = [test_x[0]]
            random_state = torch.random.get_rng_state()
            covert_signal = AE.channel(test_x + A(test_z, test_m), channel=channel_parameters['channel_type'],
                                       ebno=ebno)
            plt = plot_constellation([(covert_signal, 'x', 'tab:red', 2), (encoder_signal, 'o', 'k', 30)])
            plt.xlabel("In-phase", fontsize=18)
            plt.ylabel('Quadrature', fontsize=18)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.tight_layout()
            if save:
                plt.savefig(
                    'results/eps/constellation/' + channel_parameters['channel_type'] + '_covert_' + str(i) + '.eps',
                    format='eps')
                plt.savefig(
                    'results/png/constellation/' + channel_parameters['channel_type'] + '_covert_' + str(i) + '.png',
                    format='png')
            else:
                plt.show()
            plt.close()

            torch.random.set_rng_state(random_state)
            normal_signal = AE.channel(test_x, channel=channel_parameters['channel_type'], ebno=ebno)
            plt = plot_constellation([(normal_signal, 'x', 'tab:green', 2), (encoder_signal, 'o', 'k', 30)])
            plt.xlabel("In-phase", fontsize=18)
            plt.ylabel('Quadrature', fontsize=18)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.tight_layout()
            if save:
                plt.savefig(
                    'results/eps/constellation/' + channel_parameters['channel_type'] + '_normal_' + str(i) + '.eps',
                    format='eps')
                plt.savefig(
                    'results/png/constellation/' + channel_parameters['channel_type'] + '_normal_' + str(i) + '.png',
                    format='png')
            else:
                plt.show()
            plt.close()
        if s is not None:
            break


'''
    Run training or evaluation
'''
run_train()
run_eval()

# eval_constellation(s=1, save=False)
# eval_rate_change()
