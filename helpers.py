import torch
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from shared_parameters import channel_parameters, system_parameters

if system_parameters['system_type'] == "siso":
    from wireless_autoencoder.siso.parameters import model_parameters
else:
    from wireless_autoencoder.mimo.parameters import model_parameters
plt.rcParams["font.family"] = "Times New Roman"

if model_parameters['device'] == 'auto':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(model_parameters['device'])


def frange(x, y, jump):
    while x < y:
        yield x
        x += jump


'''
    :param:
        fd      : Doppler frequency
        Ts      : sampling period
        Ns      : number of samples
        t0      : initial time
        E0      : channel power
        phi_N   : initial phase of the maximum doppler frequency sinusoid
    
    :return: 
        h       : complex fading vector
'''


def jakes_flat(fd=926, Ts=1e-6, Ns=1, t0=0, E0=np.sqrt(1), phi_N=0):
    N0 = 8
    N = 4 * N0 + 2
    wd = 2 * np.pi * fd
    t = t0 + np.asarray([i for i in range(0, Ns)]) * Ts
    H = np.ones((Ns, 2))
    coff = E0 / np.sqrt(2 * N0 + 1)
    phi_n = np.asarray([np.pi * i / (N0 + 1) for i in range(1, N0 + 1)])
    w_n = np.asarray([wd * np.cos(2 * np.pi * i / N) for i in range(1, N0 + 1)])
    h_i = np.ones((N0 + 1, Ns))
    for i in range(N0):
        h_i[i, :] = 2 * np.cos(phi_n[i]) * np.cos(w_n[i] * t)
    h_i[N0, :] = np.sqrt(2) * np.cos(phi_N) * np.cos(wd * t)
    h_q = np.ones((N0 + 1, Ns))
    for i in range(N0):
        h_q[i, :] = 2 * np.sin(phi_n[i]) * np.cos(w_n[i] * t)
    h_q[N0, :] = np.sqrt(2) * np.sin(phi_N) * np.cos(wd * t)
    h_I = coff * np.sum(h_i, 0)
    h_Q = coff * np.sum(h_q, 0)
    H[:, 0] = h_I
    H[:, 1] = h_Q
    H = torch.from_numpy(H)
    return torch.unsqueeze(torch.view_as_complex(H.type(torch.float32)), dim=1)


def plot_constellation(data, *labels):
    image = plt.figure(figsize=(6, 6))
    for signals, marker, color, size, *index in data:
        signals = signals.cpu().detach().numpy()

        for i in range(0, model_parameters['n_channel']):
            plt.scatter(signals[:, i], signals[:, i + model_parameters['n_channel']], marker=marker, c=color, s=size,
                        label=('Autoencoder' + str(index[0]) if len(index) else ""))


    image.axes[0].set_xticks(list(frange(-8, 9, 2)))
    image.axes[0].set_yticks(list(frange(-8, 9, 2)))

    plt.xlim([-4, 4])
    plt.ylim([-4, 4])

    plt.legend()

    image.axes[0].axhline(0, linestyle='-', color='gray', alpha=0.5)
    image.axes[0].axvline(0, linestyle='-', color='gray', alpha=0.5)
    plt.xlabel('In-Phase')
    plt.ylabel('Quadrature')

    # return plt
    plt.show()


def bler_chart(model, test_ds, manual_seed=None):
    if channel_parameters['channel_type'] == 'rayleigh':
        ebno_range = list(frange(0, 22, 5))
    elif channel_parameters['channel_type'] == 'rician':
        ebno_range = list(frange(-5, 22, 5))
    else:
        ebno_range = list(frange(-4, 9, 1))
    ber = [None] * len(ebno_range)

    x, y = test_ds[:][0], test_ds[:][1]
    x = x.to(device)
    for j in range(0, len(ebno_range)):
        with torch.no_grad():
            encoded_signal = model.encoder(x)
            channel_signal = model.channel(encoded_signal, ebno=ebno_range[j])
            decoded_signal = model.demodulate(channel_signal)

            # decoded_signal = model(x)

            ber[j] = torch.sum(decoded_signal.detach().cpu() != y).item() / len(decoded_signal)
            print('SNR: {}, BER: {:.8f}'.format(ebno_range[j], ber[j]))

    print(ber)
    plt.xlim(ebno_range[0], ebno_range[-1])

    plt.plot(ebno_range[0:-1], ber[0:-1], '-ko', clip_on=True,
             label="Autoencoder (" + str(model_parameters['n_channel']) + "," + str(model_parameters['k']) + ")" + (
                 (" - seed: " + str(manual_seed)) if (manual_seed is not None) else ""))

    '''
        Tim O’Shea (Autoencoder + RTN) block error rate for Rayleigh fading channel with 3 taps.
    '''
    # plt.plot(ebno_range, [7.6 * 1e-1, 4e-1, 8 * 1e-2, 6 * 1e-3, 4 * 1e-4], '-s')

    plt.yscale('log')
    plt.xlabel("$E_{b}/N_{0}$ (dB)", fontsize=16)
    plt.ylabel('Block Error Rate', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid()
    plt.legend(loc="upper right", ncol=1, fontsize=16)
    plt.tight_layout()
    plt.savefig("results/png/autoencoder_bler_" + channel_parameters['channel_type'] + ".png")
    plt.savefig("results/eps/autoencoder_bler_" + channel_parameters['channel_type'] + ".eps", format="eps")
    plt.show()


def blers_chart(models, test_dses, manual_seed=None):
    if channel_parameters['channel_type'] == 'rayleigh':
        ebno_range = list(frange(0, 22, 5))
    elif channel_parameters['channel_type'] == 'rician':
        ebno_range = list(frange(-5, 22, 5))
    else:
        ebno_range = list(frange(-4, 9, 1))
    bers = [[None] * len(ebno_range) for _ in range(model_parameters['n_user'])]

    for j in range(0, len(ebno_range)):
        with torch.no_grad():
            if channel_parameters['channel_type'] == 'awgn':
                encodes = []
                signals = []
                for i in range(model_parameters['n_user']):
                    x = test_dses[i][:][0].to(device)
                    encode = models[i].encoder(x)
                    encodes.append(encode)
                    signals.append(models[i].channel(encode, ebno=ebno_range[j]))

                for i in range(model_parameters['n_user']):
                    for t, encode in enumerate(encodes):
                        if t != i:
                            signals[i] += encode

                for i in range(model_parameters['n_user']):
                    demodulate = models[i].demodulate(signals[i])
                    bers[i][j] = torch.sum(demodulate.detach().cpu() != test_dses[i][:][1]).item() / len(demodulate)
            else:
                encodes = []
                for i in range(model_parameters['n_user']):
                    x = test_dses[i][:][0].to(device)
                    encode = models[i].encoder(x)
                    encodes.append(encode)

                encodes = torch.stack(encodes)
                encodes.transpose_(0, 1)

                h = torch.randn((x.size()[0], model_parameters['n_user'], model_parameters['n_user']),
                                dtype=torch.cfloat).to(device)
                # h_r = torch.normal(np.sqrt(0.5), 0,
                #                    (x.size()[0], model_parameters['n_user'], model_parameters['n_user']))
                # h_i = torch.normal(np.sqrt(0.5), 0,
                #                    (x.size()[0], model_parameters['n_user'], model_parameters['n_user']))
                # h = torch.complex(h_r, h_i).to(device)
                signals = models[0].channel(encodes, ebno=ebno_range[j], h=h)

                for i in range(model_parameters['n_user']):
                    demodulate = models[i].demodulate(signals, torch.view_as_real(h).view(-1, 2 * model_parameters[
                        'n_user'] * model_parameters['n_user']))
                    bers[i][j] = torch.sum(demodulate.detach().cpu() != test_dses[i][:][1]).item() / len(demodulate)

    print(bers)

    for index, ber in enumerate(bers):
        plt.xlim(ebno_range[0], ebno_range[-1])
        plt.plot(ebno_range[0:-1], ber[0:-1], '-o', clip_on=True,
                 label="Autoencoder" + str(index) + " (" + str(model_parameters['n_channel']) + "," + str(
                     model_parameters['k']) + ")" + (
                           (" - seed: " + str(manual_seed)) if (manual_seed is not None) else ""))
    plt.yscale('log')
    plt.xlabel("$E_{b}/N_{0}$ (dB)", fontsize=16)
    plt.ylabel('Block Error Rate', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid()
    plt.legend(loc="upper right", ncol=1, fontsize=16)
    plt.tight_layout()
    plt.show()


def losses_chart(losses, multiple=False):
    if not multiple:
        losses = [losses]
    for idx, loss in enumerate(losses, start=1):
        plt.plot(list(range(1, len(loss) + 1)), loss,
                 label="Autoencoder" + (str(idx) if multiple else "") + "(" + str(
                     model_parameters['n_channel']) + "," + str(model_parameters['k']) + ")")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc="upper right", ncol=1)
    plt.show()


def accuracies_chart(accs):
    # plt.figure(figsize=(6, 6))
    fig, ax = plt.subplots()
    plots = ['autoencoder', 'bob', 'willie']
    colors_map = {'autoencoder': 'k', 'bob': 'r', 'willie': 'g'}
    # plots = ['bob', 'willie']
    for plot in plots:
        plt.plot(range(0, len(accs[plot])), accs[plot], colors_map[plot], label=str(plot).capitalize() + " Accuracy")
    # plt.title("Models Accuracies")
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("Accuracy", fontsize=18)
    plt.legend(loc="center right", fontsize=16)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: round(x * 10)))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig("results/png/training_progress_" + channel_parameters['channel_type'] + ".png")
    plt.savefig("results/eps/training_progress_" + channel_parameters['channel_type'] + ".eps", format="eps")
    plt.show()
    # plt.figure()
    # plt.plot(range(0, len(results['willie_losses'])), results['willie_losses'], label="Willie Losses")
    # plt.xlabel("Epoch")
    # plt.ylabel("Losses")
    # plt.legend(loc="lower right")
    # plt.show()


def accuracy(pred, label):
    return torch.sum(pred == label).item() / len(pred)


def discriminator_accuracy(fake_preds, real_preds):
    fake_preds = fake_preds.cpu()
    real_preds = real_preds.cpu()
    fake_preds[fake_preds > 0.5] = 1
    fake_preds[fake_preds <= 0.5] = 0
    real_preds[real_preds > 0.5] = 1
    real_preds[real_preds <= 0.5] = 0
    discriminator_fake_accuracy = accuracy(fake_preds, torch.zeros(len(fake_preds), 1))
    discriminator_real_accuracy = accuracy(real_preds, torch.ones(len(real_preds), 1))
    return (discriminator_fake_accuracy + discriminator_real_accuracy) / 2


def classifier_accuracy(pred, label):
    _, preds = torch.max(pred, dim=1)
    return accuracy(preds, label)


def reset_grads(*args):
    for arg in args:
        arg.zero_grad()
