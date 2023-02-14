import datetime

import torch
# torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np

from wireless_autoencoder.mimo.parameters import model_parameters
from wireless_covert.mimo.parameters import covert_parameters
from shared_parameters import channel_parameters, system_parameters
from wireless_autoencoder.mimo.model import Wireless_Autoencoder, Wireless_Decoder
from models import Alice, Bob, Willie, device
from helpers import frange, accuracies_chart, plot_constellation, reset_grads, discriminator_accuracy, \
    classifier_accuracy

assert system_parameters['system_type'] == 'mimo'

decoder_model = None


def initialize():
    AEs = []
    with torch.no_grad():
        xs = []
        ys = []
        autoencoder_ds = torch.load(
            '../../data/' + system_parameters['system_type'] + '/' + channel_parameters[
                'channel_type'] + '/' + str(model_parameters['n_user']) + '-user/wireless_autoencoder(' + str(
                model_parameters['n_channel']) + ',' + str(model_parameters['k']) + ')_train.pt')
        for i in range(model_parameters['n_user']):
            AE = Wireless_Autoencoder()
            AE.index = i + 1
            AE.load()
            AE.eval()
            if torch.get_default_dtype() == torch.float32:
                AE = AE.float()
            AEs.append(AE)
            idx = torch.randperm(len(autoencoder_ds))
            if torch.get_default_dtype() == torch.float32:
                x = AE.encoder(autoencoder_ds[idx][0].float())
            else:
                x = AE.encoder(autoencoder_ds[idx][0])
            y = autoencoder_ds[idx][1]
            xs.append(x)
            ys.append(y)

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
                'channel_type'] + '/' + str(model_parameters['n_user']) + '-user/wireless_autoencoder(' + str(
                model_parameters['n_channel']) + ',' + str(
                model_parameters['k']) + ')_test.pt')
        for i in range(model_parameters['n_user']):
            idx = torch.randperm(len(autoencoder_tds))
            if torch.get_default_dtype() == torch.float32:
                test_x = AEs[i].encoder(autoencoder_tds[idx][0].float())
            else:
                test_x = AEs[i].encoder(autoencoder_tds[idx][0])
            test_y = autoencoder_tds[idx][1]
            test_xs.append(test_x)
            test_ys.append(test_y)
        if channel_parameters['channel_type'] == 'awgn':
            test_x = torch.sum(torch.stack(test_xs), 0)
        else:
            test_x = torch.stack(test_xs)
            test_x.transpose_(0, 1)
        test_y = torch.stack(test_ys)

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

    global decoder_model
    decoder_model = Wireless_Decoder()
    decoder_model.load()
    decoder_model.eval()
    decoder_model.to(device)

    return (A, B, W, AEs), ((x, y), (test_x, test_y), (m, test_m))


def train(models, datasets, num_epochs=covert_parameters['num_epochs'],
          learning_rate=covert_parameters['learning_rate'],
          optim_fn=torch.optim.Adam, lambdas=(1, 1, 1)):
    A, B, W, AEs = models
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

    ae_coefficients = [1 / model_parameters['n_user'] for _ in range(model_parameters['n_user'])]

    # ae_coefficients[0] = 0.75

    def train_batch(x, y, m):
        '''
            Randomly switching noise power to be robust against different noise levels
        '''
        if channel_parameters['channel_type'] == 'awgn':
            covert_parameters['channel_k'] = None
            covert_parameters['ebno'] = torch.randint(0, 12, (1,))
        else:
            covert_parameters['channel_k'] = None
            covert_parameters['ebno'] = torch.randint(10, 25, (1,))
            # covert_parameters['ebno'] = channel_parameters['ebno']

        '''
            Train Willie
        '''
        reset_grads(a_optimizer, b_optimizer, w_optimizer)
        if channel_parameters['channel_type'] == 'rayleigh':
            h = torch.randn((x.size()[0], model_parameters['n_user'], model_parameters['n_user']),
                            dtype=torch.cfloat).to(device)
        else:
            h = None

        z = torch.randn(covert_parameters['batch_size'], covert_parameters['n_channel'] * 2).to(device)
        alice_output = A(z, m, h)
        if channel_parameters['channel_type'] == 'rayleigh':
            alice_output = alice_output.unsqueeze(1)
            # alice_output = torch.concat(
            #     (alice_output, torch.zeros(x.size()[0], x.size()[1] - 1, x.size()[2]).to(device)), dim=1)

        normal_labels = torch.ones(covert_parameters['batch_size'], 1).to(device)
        covert_labels = torch.zeros(covert_parameters['batch_size'], 1).to(device)

        ''' METHOD 1 '''
        willie_loss_ind = []
        for i in range(model_parameters['n_user']):
            if channel_parameters['channel_type'] == 'awgn':
                willie_output_normal = W(
                    AEs[i].channel(x, channel_parameters['channel_type'], ebno=covert_parameters['ebno']))
            else:
                willie_output_normal = W(
                    AEs[i].channel(x, channel_parameters['channel_type'], ebno=covert_parameters['ebno'], h=h)[:, i, :])

            if channel_parameters['channel_type'] == 'awgn':
                willie_output_covert = W(
                    AEs[i].channel(alice_output + x, channel_parameters['channel_type'],
                                   ebno=covert_parameters['ebno']))
            else:
                willie_output_covert = W(
                    AEs[i].channel(alice_output + x, channel_parameters['channel_type'], ebno=covert_parameters['ebno'],
                                   h=h)[:, i, :])

            willie_loss_ind.append(
                w_criterion(willie_output_normal, normal_labels) + w_criterion(willie_output_covert, covert_labels))

        willie_loss = sum(willie_loss_ind)
        ''' METHOD 1 '''
        ''' METHOD 2 '''
        # signals_normal = []
        # signals_covert = []
        # for i in range(model_parameters['n_user']):
        #     if channel_parameters['channel_type'] == 'awgn':
        #         signals_normal.append(AEs[i].channel(x, channel_parameters['channel_type'], ebno=covert_parameters['ebno'],
        #                                      k=covert_parameters['channel_k']))
        #         signals_covert.append(AEs[i].channel(alice_output + x, channel_parameters['channel_type'], ebno=covert_parameters['ebno'],
        #                              k=covert_parameters['channel_k']))
        #     else:
        #         signals_normal.append(AEs[i].channel(x, channel_parameters['channel_type'], ebno=covert_parameters['ebno'],
        #                                      k=covert_parameters['channel_k'], h=h)[:, i, :])
        #         signals_covert.append(AEs[i].channel(alice_output + x, channel_parameters['channel_type'], ebno=covert_parameters['ebno'],
        #                              k=covert_parameters['channel_k'], h=h)[:, i, :])
        #
        # signals_normal = torch.stack(signals_normal).transpose(0, 1).reshape(-1, model_parameters['n_user'] * model_parameters['n_channel'] * 2)
        # signals_covert = torch.stack(signals_covert).transpose(0, 1).reshape(-1, model_parameters['n_user'] * model_parameters['n_channel'] * 2)
        # willie_loss = w_criterion(W(signals_normal), normal_labels) + w_criterion(W(signals_covert), covert_labels)
        ''' METHOD 2 '''

        willie_loss.backward()
        w_optimizer.step()

        '''
            Train Alice
        '''
        reset_grads(a_optimizer, b_optimizer, w_optimizer)
        z = torch.randn(covert_parameters['batch_size'], covert_parameters['n_channel'] * 2).to(device)
        x = x.detach()
        alice_output = A(z, m, h)
        if channel_parameters['channel_type'] != 'awgn':
            alice_output = alice_output.unsqueeze(1)
            # alice_output = torch.concat(
            #     (alice_output, torch.zeros(x.size()[0], x.size()[1] - 1, x.size()[2]).to(device)), dim=1)

        ''' METHOD 1 '''
        willie_loss_ind = []
        for i in range(model_parameters['n_user']):
            if channel_parameters['channel_type'] == 'awgn':
                willie_output = W(
                    AEs[i].channel(alice_output + x, channel_parameters['channel_type'],
                                   ebno=covert_parameters['ebno']))

            else:
                willie_output = W(
                    AEs[i].channel(alice_output + x, channel_parameters['channel_type'], ebno=covert_parameters['ebno'],
                                   h=h)[:, i, :])

            willie_loss_ind.append(w_criterion(willie_output, normal_labels))

        willie_dloss = sum(willie_loss_ind)
        ''' METHOD 1 '''

        ''' METHOD 2 '''
        # signals_covert = []
        # for i in range(model_parameters['n_user']):
        #     if channel_parameters['channel_type'] == 'awgn':
        #         signals_covert.append(
        #             AEs[i].channel(alice_output + x, channel_parameters['channel_type'], ebno=covert_parameters['ebno'],
        #                            k=covert_parameters['channel_k']))
        #     else:
        #         signals_covert.append(
        #             AEs[i].channel(alice_output + x, channel_parameters['channel_type'], ebno=covert_parameters['ebno'],
        #                            k=covert_parameters['channel_k'], h=h)[:, i, :])
        # signals_covert = torch.stack(signals_covert).transpose(0, 1).reshape(-1, model_parameters['n_user'] *
        #                                                                       model_parameters['n_channel'] * 2)
        # willie_dloss = w_criterion(W(signals_covert), normal_labels)
        ''' METHOD 2 '''

        ''' METHOD 1 '''
        bob_outputs = []
        for i in range(model_parameters['n_user']):
            if channel_parameters['channel_type'] == 'awgn':
                bob_outputs.append(B(
                    AEs[i].channel(alice_output + x, channel_parameters['channel_type'],
                                   ebno=covert_parameters['ebno'])))
            else:
                bob_outputs.append(B(
                    AEs[i].channel(alice_output + x, channel_parameters['channel_type'], ebno=covert_parameters['ebno'],
                                   h=h)[:, i, :]))
            break
        bob_output = torch.sum(torch.stack(bob_outputs), 0)
        bob_loss = b_criterion(bob_output, m)
        ''' METHOD 1 '''

        ''' METHOD 2 '''
        # bob_loss = b_criterion(B(signals_covert), m)
        ''' METHOD 2 '''

        if channel_parameters['channel_type'] == 'awgn':
            alice_outputs = []
            for i in range(model_parameters['n_user']):
                alice_outputs.append(AEs[i].channel(alice_output + x, channel_parameters['channel_type'],
                                                    ebno=covert_parameters['ebno']))

            alice_output = torch.stack(alice_outputs)
            alice_output.transpose_(0, 1)
            alice_output = alice_output.reshape(-1, model_parameters['n_user'] * model_parameters['n_channel'] * 2)
        else:
            alice_output = AEs[i].channel(alice_output + x, channel_parameters['channel_type'],
                                          ebno=covert_parameters['ebno'], h=h)

        if channel_parameters['channel_type'] == 'awgn':
            ae_loss = 0
            ae_ind_loss = [[] for _ in range(model_parameters['n_user'])]

            for i in range(model_parameters['n_user']):
                # ae_ind_loss[i] = ae_criterion(
                #     AEs[i].decoder_net(alice_output, torch.view_as_real(h).view(-1, 2 *
                #                                                                 model_parameters[
                #                                                                     'n_user'] *
                #                                                                 model_parameters[
                #                                                                     'n_user'])),
                #     y[i]) * 10

                ae_ind_loss[i] = ae_criterion(decoder_model(alice_output, i), y[i]) * 10

                ae_loss += ae_coefficients[i] * ae_ind_loss[i]
        else:
            ae_loss = 0
            ae_ind_loss = [[] for _ in range(model_parameters['n_user'])]

            alice_output = alice_output.view(
                (-1, model_parameters['n_user'], model_parameters['n_channel'], 2))
            alice_output = torch.view_as_complex(alice_output)
            transform = torch.inverse(h) @ alice_output
            alice_output = torch.view_as_real(transform).view(transform.size()[0], -1)

            for i in range(model_parameters['n_user']):
                # ae_ind_loss[i] = ae_criterion(
                #     AEs[i].decoder_net(alice_output, torch.view_as_real(h).view(-1, 2 *
                #                                                                 model_parameters[
                #                                                                     'n_user'] *
                #                                                                 model_parameters[
                #                                                                     'n_user'])),
                #     y[i]) * 10

                ae_ind_loss[i] = ae_criterion(decoder_model(alice_output, i), y[i]) * 10

                ae_loss += ae_coefficients[i] * ae_ind_loss[i]

            with torch.no_grad():
                for i in range(model_parameters['n_user']):
                    ae_coefficients[i] = ae_ind_loss[i].item() / sum(ae_ind_loss)

        alice_bob_loss = ae_lambda * ae_loss + willie_lambda * willie_dloss + bob_lambda * bob_loss

        alice_bob_loss.backward()
        a_optimizer.step()

        '''
            Train Bob
        '''
        reset_grads(a_optimizer, b_optimizer, w_optimizer)
        z = torch.randn(covert_parameters['batch_size'], covert_parameters['n_channel'] * 2).to(device)
        alice_output = A(z, m, h)
        if channel_parameters['channel_type'] == 'rayleigh':
            alice_output = alice_output.unsqueeze(1)
            # alice_output = torch.concat(
            #     (alice_output, torch.zeros(x.size()[0], x.size()[1] - 1, x.size()[2]).to(device)), dim=1)
        x = x.detach()

        ''' METHOD 1 '''
        bob_outputs = []
        for i in range(model_parameters['n_user']):
            if channel_parameters['channel_type'] == 'awgn':
                bob_outputs.append(B(
                    AEs[i].channel(alice_output + x, channel_parameters['channel_type'],
                                   ebno=covert_parameters['ebno'])))
            else:
                bob_outputs.append(B(
                    AEs[i].channel(alice_output + x, channel_parameters['channel_type'], ebno=covert_parameters['ebno'],
                                   h=h)[:, i, :]))
            break

        bob_output = torch.sum(torch.stack(bob_outputs), 0)
        bob_loss = b_criterion(bob_output, m)
        ''' METHOD 1 '''
        ''' METHOD 2 '''
        # signals_covert = []
        # for i in range(model_parameters['n_user']):
        #     if channel_parameters['channel_type'] == 'awgn':
        #         signals_covert.append(
        #             AEs[i].channel(alice_output + x, channel_parameters['channel_type'], ebno=covert_parameters['ebno'],
        #                            k=covert_parameters['channel_k']))
        #     else:
        #         signals_covert.append(
        #             AEs[i].channel(alice_output + x, channel_parameters['channel_type'], ebno=covert_parameters['ebno'],
        #                            k=covert_parameters['channel_k'], h=h)[:, i, :])
        # signals_covert = torch.stack(signals_covert).transpose(0, 1).reshape(-1, model_parameters['n_user'] *
        #                                                                       model_parameters['n_channel'] * 2)
        #
        # bob_loss = b_criterion(B(signals_covert), m)
        ''' METHOD 2 '''

        bob_loss.backward()
        b_optimizer.step()

        return willie_loss, bob_loss, alice_bob_loss, ae_loss

    for epoch in range(num_epochs):
        A.train()
        B.train()
        W.train()

        if dl:
            for index, (x, y, m) in enumerate(dl):
                willie_loss, bob_loss, alice_bob_loss, ae_loss = train_batch(x, y, m)
        else:
            willie_loss, bob_loss, alice_bob_loss, ae_loss = train_batch(x, y, m)

        if epoch != 0 and epoch % 500 == 0:
            w_optimizer.param_groups[0]['lr'] /= 2
            b_optimizer.param_groups[0]['lr'] /= 2
            a_optimizer.param_groups[0]['lr'] /= 2

        if epoch % 10 == 0:
            with torch.no_grad():
                test_z = torch.randn(len(test_x), covert_parameters['n_channel'] * 2).to(device)
                if channel_parameters['channel_type'] == 'rayleigh':
                    h = torch.randn((test_x.size()[0], model_parameters['n_user'], model_parameters['n_user']),
                                    dtype=torch.cfloat).to(device)
                else:
                    h = None

                A.eval()
                B.eval()
                W.eval()

                alice_output_test = A(test_z, test_m, h)
                if channel_parameters['channel_type'] != 'awgn':
                    alice_output_test = alice_output_test.unsqueeze(1)
                    # alice_output_test = torch.concat(
                    #     (alice_output_test,
                    #      torch.zeros(test_x.size()[0], test_x.size()[1] - 1, test_x.size()[2]).to(device)), dim=1)

                ''' METHOD 1 '''
                willie_accs = []
                for i in range(model_parameters['n_user']):
                    if channel_parameters['channel_type'] == 'awgn':
                        willie_accs.append(discriminator_accuracy(
                            W(AEs[i].channel(alice_output_test + test_x, channel_parameters['channel_type'])),
                            W(AEs[i].channel(test_x, channel_parameters['channel_type']))))
                    else:
                        willie_accs.append(discriminator_accuracy(
                            W(AEs[i].channel(alice_output_test + test_x, channel_parameters['channel_type'], h=h)[:, i,
                              :]),
                            W(AEs[i].channel(test_x, channel_parameters['channel_type'], h=h)[:, i, :])))

                willie_acc = sum(willie_accs) / len(willie_accs)

                bob_outputs = []
                for i in range(model_parameters['n_user']):
                    if channel_parameters['channel_type'] == 'awgn':
                        bob_outputs.append(
                            B(AEs[i].channel(alice_output_test + test_x, channel_parameters['channel_type'])))
                    else:
                        bob_outputs.append(B(
                            AEs[i].channel(alice_output_test + test_x, channel_parameters['channel_type'], h=h)[:, i,
                            :]))
                    break
                bob_outputs = torch.sum(torch.stack(bob_outputs), dim=0)
                bob_acc = classifier_accuracy(bob_outputs, test_m)

                ''' METHOD 1 '''

                ''' METHOD 2 '''
                # signals_covert_test = []
                # signals_normal_test = []
                # for i in range(model_parameters['n_user']):
                #     if channel_parameters['channel_type'] == 'awgn':
                #         signals_covert_test.append(
                #             AEs[i].channel(alice_output_test + test_x, channel_parameters['channel_type']))
                #         signals_normal_test.append(
                #             AEs[i].channel(test_x, channel_parameters['channel_type']))
                #     else:
                #         signals_covert_test.append(
                #             AEs[i].channel(alice_output_test + test_x, channel_parameters['channel_type'], h=h)[:, i, :])
                #         signals_normal_test.append(
                #             AEs[i].channel(test_x, channel_parameters['channel_type'], h=h)[:, i, :])
                # signals_covert_test = torch.stack(signals_covert_test).transpose(0, 1).reshape(-1, model_parameters['n_user'] *
                #                                                                       model_parameters['n_channel'] * 2)
                # signals_normal_test = torch.stack(signals_normal_test).transpose(0, 1).reshape(-1, model_parameters[
                #     'n_user'] *
                #                                                                                 model_parameters[
                #                                                                                     'n_channel'] * 2)
                # bob_acc = classifier_accuracy(B(signals_covert_test), test_m)
                # willie_acc = discriminator_accuracy(W(signals_covert_test), W(signals_normal_test))
                ''' METHOD 2 '''

                accs['willie'].append(round(willie_acc, 2))
                accs['bob'].append(round(bob_acc, 2))

                if channel_parameters['channel_type'] == 'awgn':
                    alice_output_tests = []
                    for _ in range(model_parameters['n_user']):
                        alice_output_tests.append(
                            AEs[_].channel(alice_output_test + test_x, channel_parameters['channel_type']))
                    alice_output_test = torch.stack(alice_output_tests)
                    alice_output_test.transpose_(0, 1)
                    alice_output_test = alice_output_test.reshape(-1, model_parameters['n_user'] * model_parameters[
                        'n_channel'] * 2)
                else:
                    alice_output_test = AEs[i].channel(alice_output_test + test_x, channel_parameters['channel_type'],
                                                       h=h)
                # alice_output_test = AE.channel(test_x, channel_parameters['channel_type'])

                model_accs = []

                if channel_parameters['channel_type'] != 'awgn':
                    alice_output_test = alice_output_test.view(
                        (-1, model_parameters['n_user'], model_parameters['n_channel'], 2))
                    alice_output_test = torch.view_as_complex(alice_output_test)
                    transform = torch.inverse(h) @ alice_output_test
                    alice_output_test = torch.view_as_real(transform).view(transform.size()[0], -1)

                for i in range(model_parameters['n_user']):
                    if channel_parameters['channel_type'] == 'awgn':
                        # model_acc = classifier_accuracy(AEs[i].decoder_net(alice_output_test), test_y[i])
                        model_acc = classifier_accuracy(decoder_model(alice_output_test, i), test_y[i])
                    else:
                        # model_acc = classifier_accuracy(AEs[i].decoder_net(alice_output_test,
                        #                                                    torch.view_as_real(h).view(-1, 2 *
                        #                                                                               model_parameters[
                        #                                                                                   'n_user'] *
                        #                                                                               model_parameters[
                        #                                                                                   'n_user'])),
                        #                                 test_y[i])
                        model_acc = classifier_accuracy(decoder_model(alice_output_test, i), test_y[i])
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
            # print(', '.join(['Model' + str(i + 1) + ' Acc: {:.2f}'.format(model_accs[i]) for i in range(model_parameters['n_user'])]))

    return accs, losses


def run_train(seed=covert_parameters['seed'], save=True):
    time = datetime.datetime.now()
    torch.manual_seed(seed)
    print('Device: ', 'CPU' if device == 'cpu' else 'GPU')
    models, datasets = initialize()
    A, B, E, AE = models

    if channel_parameters['channel_type'] == "awgn":
        if model_parameters['n_user'] == 2:
            ae_lambda, bob_lambda, willie_lambda = 0.1, 0.2, 0.6
        if model_parameters['n_user'] == 4:
            ae_lambda, bob_lambda, willie_lambda = 0.1, 0.2, 0.4
    else:
        if model_parameters['n_user'] == 2:
            ae_lambda, bob_lambda, willie_lambda = 0.01, 0.8, 0.3
        if model_parameters['n_user'] == 4:
            ae_lambda, bob_lambda, willie_lambda = 0.01, 0.8, 0.2

    accs, losses = train(models, datasets, lambdas=(ae_lambda, bob_lambda, willie_lambda))

    # if save:
    #     torch.save(A.state_dict(), '../../models/' + system_parameters['system_type'] + '/' + channel_parameters[
    #         'channel_type'] + '/' + str(model_parameters['n_user']) + '-user/wireless_covert_alice(' + str(
    #         covert_parameters['n_channel']) + ',' + str(
    #         covert_parameters['k']) + ').pt')
    #     torch.save(B.state_dict(), '../../models/' + system_parameters['system_type'] + '/' + channel_parameters[
    #         'channel_type'] + '/' + str(model_parameters['n_user']) + '-user/wireless_covert_bob(' + str(
    #         covert_parameters['n_channel']) + ',' + str(
    #         covert_parameters['k']) + ').pt')
    #     torch.save(E.state_dict(), '../../models/' + system_parameters['system_type'] + '/' + channel_parameters[
    #         'channel_type'] + '/' + str(model_parameters['n_user']) + '-user/wireless_covert_willie(' + str(
    #         covert_parameters['n_channel']) + ',' + str(
    #         covert_parameters['k']) + ').pt')
    #
    # print('Training Time: ' + str(datetime.datetime.now() - time))
    accuracies_chart(accs)


def evaluate(models, datasets, ebno):
    A, B, W, AEs = models
    A.eval(), B.eval(), W.eval()
    for AE in AEs:
        AE.eval()

    _, test_ds, m_ds = datasets
    test_x, test_y = test_ds
    m, test_m = m_ds

    if channel_parameters['channel_type'] == 'rayleigh':
        h = torch.randn((test_x.size()[0], model_parameters['n_user'], model_parameters['n_user']),
                        dtype=torch.cfloat).to(device)
    else:
        h = None

    A.load_state_dict(torch.load('../../models/' + system_parameters['system_type'] + '/' + channel_parameters[
        'channel_type'] + '/' + str(model_parameters['n_user']) + '-user/wireless_covert_alice(' + str(
        covert_parameters['n_channel']) + ',' + str(
        covert_parameters['k']) + ').pt'))
    B.load_state_dict(torch.load('../../models/' + system_parameters['system_type'] + '/' + channel_parameters[
        'channel_type'] + '/' + str(model_parameters['n_user']) + '-user/wireless_covert_bob(' + str(
        covert_parameters['n_channel']) + ',' + str(
        covert_parameters['k']) + ').pt'))
    W.load_state_dict(torch.load('../../models/' + system_parameters['system_type'] + '/' + channel_parameters[
        'channel_type'] + '/' + str(model_parameters['n_user']) + '-user/wireless_covert_willie(' + str(
        covert_parameters['n_channel']) + ',' + str(
        covert_parameters['k']) + ').pt'))

    test_z = torch.randn(len(test_x), covert_parameters['n_channel'] * 2).to(device)
    alice_output_test = A(test_z, test_m, h)
    if channel_parameters['channel_type'] == 'rayleigh':
        alice_output_test = alice_output_test.unsqueeze(1)

    ''' METHOD 1 '''
    bob_outputs = []
    willie_accs = []
    for i in range(model_parameters['n_user']):
        if channel_parameters['channel_type'] == 'awgn':
            willie_accs.append(discriminator_accuracy(
                W(AEs[i].channel(alice_output_test + test_x, channel_parameters['channel_type'], ebno=ebno)),
                W(AEs[i].channel(test_x, channel_parameters['channel_type'], ebno=ebno))))
            if i == 0:
                bob_outputs.append(
                    B(AEs[i].channel(alice_output_test + test_x, channel_parameters['channel_type'], ebno=ebno)))
        else:
            # alice_output_test = torch.concat(
            #     (alice_output_test, torch.zeros(test_x.size()[0], test_x.size()[1] - 1, test_x.size()[2]).to(device)),
            #     dim=1)
            willie_accs.append(discriminator_accuracy(
                W(AEs[i].channel(alice_output_test + test_x, channel_parameters['channel_type'], ebno=ebno, h=h)[:,
                  i, :]),
                W(AEs[i].channel(test_x, channel_parameters['channel_type'], ebno=ebno, h=h)[:, i, :])))
            if i == 0:
                bob_outputs.append(
                    B(AEs[i].channel(alice_output_test + test_x, channel_parameters['channel_type'], ebno=ebno, h=h)[:,
                      i, :]))
    willie_acc = sum(willie_accs) / len(willie_accs)
    bob_output = torch.sum(torch.stack(bob_outputs), 0)
    bob_acc = classifier_accuracy(bob_output, test_m)
    ''' METHOD 1 '''
    ''' METHOD 2 '''
    # signals_covert_test = []
    # for i in range(model_parameters['n_user']):
    #     if channel_parameters['channel_type'] == 'awgn':
    #         signals_covert_test.append(
    #             AEs[i].channel(alice_output_test + test_x, channel_parameters['channel_type'], ebno=ebno))
    #     else:
    #         signals_covert_test.append(
    #             AEs[i].channel(alice_output_test + test_x, channel_parameters['channel_type'], ebno=ebno, h=h)[:, i, :])
    # signals_covert_test = torch.stack(signals_covert_test).transpose(0, 1).reshape(-1, model_parameters['n_user'] *
    #                                                                                model_parameters['n_channel'] * 2)
    # bob_acc = classifier_accuracy(B(signals_covert_test), test_m)
    ''' METHOD 2 '''

    if channel_parameters['channel_type'] == 'awgn':
        covert_signal_tests = []
        normal_signal_tests = []
        for i in range(model_parameters['n_user']):
            covert_signal_tests.append(
                AEs[i].channel(alice_output_test + test_x, channel_parameters['channel_type'], ebno=ebno))
            normal_signal_tests.append(AEs[i].channel(test_x, channel_parameters['channel_type'], ebno=ebno))

        covert_signal_test = torch.stack(covert_signal_tests)
        covert_signal_test.transpose_(0, 1)
        covert_signal_test = covert_signal_test.reshape(-1, model_parameters['n_user'] * model_parameters[
            'n_channel'] * 2)

        normal_signal_test = torch.stack(normal_signal_tests)
        normal_signal_test.transpose_(0, 1)
        normal_signal_test = normal_signal_test.reshape(-1, model_parameters['n_user'] * model_parameters[
            'n_channel'] * 2)

    else:
        covert_signal_test = AEs[i].channel(alice_output_test + test_x, channel_parameters['channel_type'],
                                            ebno=ebno, h=h)
        normal_signal_test = AEs[i].channel(test_x, channel_parameters['channel_type'],
                                            ebno=ebno, h=h)

    # ae_acc = classifier_accuracy(AE.decoder_net(normal_signal_test), test_y)
    ae_covert_accs = []
    ae_accs = []

    if channel_parameters['channel_type'] != 'awgn':
        covert_signal_test = covert_signal_test.view((-1, model_parameters['n_user'], model_parameters['n_channel'], 2))
        covert_signal_test = torch.view_as_complex(covert_signal_test)
        transform = torch.inverse(h) @ covert_signal_test
        covert_signal_test = torch.view_as_real(transform).view(transform.size()[0], -1)

        normal_signal_test = normal_signal_test.view((-1, model_parameters['n_user'], model_parameters['n_channel'], 2))
        normal_signal_test = torch.view_as_complex(normal_signal_test)
        transform = torch.inverse(h) @ normal_signal_test
        normal_signal_test = torch.view_as_real(transform).view(transform.size()[0], -1)

    for i in range(model_parameters['n_user']):
        if channel_parameters['channel_type'] == 'awgn':
            # ae_covert_accs.append(classifier_accuracy(AEs[i].decoder_net(covert_signal_test), test_y[i]))
            # ae_accs.append(classifier_accuracy(AEs[i].decoder_net(normal_signal_test), test_y[i]))

            ae_covert_accs.append(classifier_accuracy(decoder_model(covert_signal_test, i), test_y[i]))
            ae_accs.append(classifier_accuracy(decoder_model(normal_signal_test, i), test_y[i]))
        else:
            # covert_signal_test = AEs[0].channel(test_x, channel_parameters['channel_type'],
            #                                                                         ebno=ebno, h=h)
            # ae_covert_accs.append(classifier_accuracy(
            #     AEs[i].decoder_net(covert_signal_test, torch.view_as_real(h).view(-1, 2 *
            #                                                                       model_parameters[
            #                                                                           'n_user'] *
            #                                                                       model_parameters[
            #                                                                           'n_user'])), test_y[i]))
            # ae_accs.append(classifier_accuracy(
            #     AEs[i].decoder_net(normal_signal_test, torch.view_as_real(h).view(-1, 2 *
            #                                                                       model_parameters[
            #                                                                           'n_user'] *
            #                                                                       model_parameters[
            #                                                                           'n_user'])), test_y[i]))

            ae_covert_accs.append(classifier_accuracy(decoder_model(covert_signal_test, i), test_y[i]))
            ae_accs.append(classifier_accuracy(decoder_model(normal_signal_test, i), test_y[i]))

            # break
    # ae_covert_acc = ae_covert_acc / model_parameters['n_user']
    return ae_accs, ae_covert_accs, bob_acc, willie_acc


def run_eval(seed=covert_parameters['seed']):
    torch.manual_seed(seed)
    models, datasets = initialize()

    if channel_parameters['channel_type'] == 'awgn':
        ebno_range = list(frange(-4, 9, 1.5))
    elif channel_parameters['channel_type'] == 'rician':
        ebno_range = list(frange(0, 22, 5))
    else:
        ebno_range = list(frange(0, 22, 5))

    accs = {'autoencoder': [], 'autoencoder_covert': [], 'bob': [], 'willie': []}
    accs['autoencoder_covert'] = [[] for _ in range(model_parameters['n_user'])]
    accs['autoencoder'] = [[] for _ in range(model_parameters['n_user'])]

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

    torch.random.set_rng_state(torch.load('../../data/' + system_parameters['system_type'] + '/' + channel_parameters[
        'channel_type'] + '/' + str(model_parameters['n_user']) + '-user/channel_random_state.pt'))
    for i in range(0, len(ebno_range)):
        ae_accs, ae_covert_accs, bob_acc, willie_acc = evaluate(models, datasets, ebno=ebno_range[i])
        for i in range(model_parameters['n_user']):
            accs['autoencoder_covert'][i].append(1 - ae_covert_accs[i])
            accs['autoencoder'][i].append(1 - ae_accs[i])
        accs['bob'].append(1 - bob_acc)
        accs['willie'].append(round(willie_acc, 2))

    with open(
            'results/txt/accuracies(' + str(covert_parameters['n_channel']) + ',' + str(covert_parameters['k']) + ')_' +
            channel_parameters['channel_type'] + '_' + str(model_parameters['n_user']) + 'u.txt', 'w') as f:
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
            for i in range(model_parameters['n_user']):
                plot_label = "AE" + str(i + 1) + " (" + str(model_parameters['n_channel']) + "," + str(
                    model_parameters['k']) + ") - After Covert Communication"
                plot_label = "AE" + str(i + 1) + " (" + str(model_parameters['n_channel']) + "," + str(
                    model_parameters['k']) + ") - After"
                # plot_x = list(ebno_range[0:-1])
                # plot_y = accs[x][i][0:-1]
                plot_x = list(ebno_range)
                plot_y = accs[x][i]
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
                plot_label = "AE" + str(i + 1) + " (" + str(model_parameters['n_channel']) + "," + str(
                    model_parameters['k']) + ") - Before"
                # plot_x = list(ebno_range[0:-1])
                # plot_y = accs[x][i][0:-1]
                plot_x = list(ebno_range)
                plot_y = accs[x][i]
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
        # label_fontsize = 12
        plt.xlabel("$E_{b}/N_{0}$ (dB)", fontsize=label_fontsize)
        if x == 'autoencoder_covert' or x == 'bob':
            plt.ylabel('Block Error Rate', fontsize=label_fontsize)
        else:
            plt.ylabel('Accuracy', fontsize=label_fontsize)

        tick_fontsize = 14
        # tick_fontsize = 10
        plt.xticks(fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        plt.grid()
        plt.legend(loc="lower left", ncol=1, fontsize=label_fontsize)
        plt.tight_layout()
        plt.savefig("results/png/tmp_" + x + '.png')
        plt.show()
        plt.close()


def eval_nuser_change():
    rates = [2, 4]
    accs = {'autoencoder': [], 'autoencoder_covert': [], 'bob': [], 'willie': []}

    chart_conf = {
        'autoencoder': {
            'c': ['blue', 'red'],
            'l': ['dashed', 'dashed'],
            'm': [None, None]
        },
        'autoencoder_covert': {
            'c': ['blue', 'red'],
            'l': ['solid', 'solid'],
            'm': ['o', 'd']
        },
        'bob': {
            'c': ['blue', 'red'],
            'l': ['dashed', 'dashed'],
            'm': ['o', 'd']
        },
        'willie': {
            'c': ['blue', 'red'],
            'l': ['dashed', 'dashed'],
            'm': [None, None]
        }
    }

    if channel_parameters['channel_type'] == 'awgn':
        ebno_range = list(frange(-4, 9, 1.5))
    elif channel_parameters['channel_type'] == 'rician':
        ebno_range = list(frange(-5, 22, 5))
    else:
        ebno_range = list(frange(0, 22, 5))

    for rate in rates:
        f = open(
            'results/txt/accuracies(' + str(covert_parameters['n_channel']) + ',' + str(covert_parameters['k']) + ')_' +
            channel_parameters['channel_type'] + '_' + str(rate) + 'u.txt')
        for line in f.readlines():
            acc = line.split(':')
            acc[1] = eval(acc[1].strip())
            if acc[0] == 'autoencoder' or acc[0] == 'autoencoder_covert':
                acc[1] = np.array(acc[1]).mean(axis=0).tolist()
            accs[acc[0]].append(acc[1])

    for x in accs:
        for i in range(len(rates)):
            if x == 'willie':
                plot_label = 'Willie' + " (" + str(covert_parameters['n_channel']) + "," + str(
                    covert_parameters['k']) + ") " + str(
                    str(rates[i])) + "-user"
                plot_x = list(ebno_range)
                plot_y = accs[x][i]
                plot_x_new = np.linspace(min(plot_x), max(plot_x), 100)
                a_BSpline = make_interp_spline(plot_x, plot_y)
                plot_y = a_BSpline(plot_x_new)
                plot_x = plot_x_new
            elif x == 'bob':
                plot_label = "Bob (" + str(covert_parameters['n_channel']) + "," + str(
                    covert_parameters['k']) + ") " + str(
                    str(rates[i])) + "-user"
                plot_x = list(ebno_range)
                plot_y = accs[x][i]
            elif x == 'autoencoder_covert':
                plot_label = "User (" + str(model_parameters['n_channel']) + "," + str(
                    model_parameters['k']) + ") " + str(rates[i]) + "-user" + " — Alice (" + str(
                    model_parameters['n_channel']) + "," + str(covert_parameters['k']) + ")"
                plot_x = list(ebno_range)
                plot_y = accs[x][i]
            elif x == 'autoencoder':
                plot_label = "User (" + str(model_parameters['n_channel']) + "," + str(
                    model_parameters['k']) + ") " + str(rates[i]) + "-user"
                plot_x = list(ebno_range)
                plot_y = accs[x][i]

            plt.plot(plot_x, plot_y, linestyle=chart_conf[x]['l'][i], marker=chart_conf[x]['m'][i],
                     color=chart_conf[x]['c'][i],
                     label=plot_label, clip_on=False)

            if x == 'autoencoder' and rates[i] == 4:
                break

            if rates[i] == 4:
                plt.xlim(ebno_range[0], ebno_range[-1])

                if x == 'autoencoder_covert' or x == 'bob':
                    if x == 'autoencoder_covert':
                        handles, labels = plt.gca().get_legend_handles_labels()
                        order = [0, 2, 1, 3]
                    plt.yscale('log')
                    if x == 'autoencoder_covert':
                        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc="lower left",
                                   ncol=1, fontsize=18)
                    else:
                        plt.legend(loc="lower left",
                                   ncol=1, fontsize=18)
                else:
                    plt.legend(loc="upper left", ncol=1, fontsize=18)

                plt.xlabel("$E_{b}/N_{0}$ (dB)", fontsize=18)
                if x == 'autoencoder_covert' or x == 'bob':
                    plt.ylabel('Block Error Rate', fontsize=18)
                else:
                    plt.ylabel('Accuracy', fontsize=18)

                plt.grid()
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
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

    A.load_state_dict(torch.load('../models/' + channel_parameters['channel_type'] + '/' + str(
        model_parameters['n_user']) + '-user/wireless_covert_alice(' + str(
        covert_parameters['n_channel']) + ',' + str(covert_parameters['k']) + ').pt'))
    B.load_state_dict(torch.load('../models/' + channel_parameters['channel_type'] + '/' + str(
        model_parameters['n_user']) + '-user/wireless_covert_bob(' + str(
        covert_parameters['n_channel']) + ',' + str(covert_parameters['k']) + ').pt'))
    W.load_state_dict(torch.load('../models/' + channel_parameters['channel_type'] + '/' + str(
        model_parameters['n_user']) + '-user/wireless_covert_willie(' + str(
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
# run_train()
# run_eval()

eval_nuser_change()
