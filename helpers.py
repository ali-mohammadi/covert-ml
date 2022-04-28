import torch
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from wireless_autoencoder.parameters import model_parameters

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


def plot_constellation(signals):
    image = plt.figure(figsize=(6, 6))
    for signal in signals:
        for i in range(0, model_parameters['n_channel']):
            plt.scatter(signal[0][i], signal[1][i], marker='x', c="blue")

    image.axes[0].set_xticks(list(range(-4, 5)))
    image.axes[0].set_yticks(list(range(-4, 5)))

    image.axes[0].axhline(0, linestyle='-', color='gray')
    image.axes[0].axvline(0, linestyle='-', color='gray')
    plt.show()


def bler_chart(model, test_ds, manual_seed=None):
    if model_parameters['channel_type'] == 'rayleigh':
        ebno_range = list(frange(0, 21, 2))
    else:
        ebno_range = list(frange(-4, 8, 0.5))
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

    plt.plot(ebno_range, ber, 'bo',
             label="Autoencoder(" + str(model_parameters['n_channel']) + "," + str(model_parameters['k']) + ")" + ("  " + str(manual_seed))) if (manual_seed != None) else ""
    plt.yscale('log')
    plt.xlabel('Eb/N0')
    plt.ylabel('Block Error Rate')
    plt.grid()
    plt.legend(loc="upper right", ncol=1)
    plt.savefig("results/autoencoder_" + model_parameters['n_channel'] + "_" + model_parameters['k'] + "__bler__" + model_parameters['channel_type'] + ".png")
    plt.show()


def losses_chart(losses):
    plt.plot(list(range(1, len(losses) + 1)), losses,
             label="Autoencoder(" + str(model_parameters['n_channel']) + "," + str(model_parameters['k']) + ")")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc="upper right", ncol=1)
    plt.show()


def accuracies_chart(accs):
    # plt.figure(figsize=(6, 6))
    fig, ax = plt.subplots()
    plots = ['model', 'bob', 'willie']
    # plots = ['bob', 'willie']
    for plot in plots:
        plt.plot(range(0, len(accs[plot])), accs[plot], label=str(plot).capitalize() + " Accuracy")
    plt.title("Models Accuracies")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="center right")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: round(x * 10)))
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