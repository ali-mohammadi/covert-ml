import math
import os
import torch
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.rcParams["font.family"] = "Times New Roman"
from shared_parameters import channel_parameters

if "siso" in os.getcwd():
    from wireless_autoencoder.siso.parameters import model_parameters
else:
    from wireless_autoencoder.mimo.parameters import model_parameters

if model_parameters['device'] == 'auto':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(model_parameters['device'])


def frange(x, y, jump):
    while x < y:
        yield x
        x += jump


def jakes_flat(fd=926, Ts=1e-6, Ns=1, t0=0, E0=np.sqrt(1), phi_N=0):
    """
        :param:
            fd      : Doppler frequency
            Ts      : sampling period
            Ns      : number of samples
            t0      : initial time
            E0      : channel power
            phi_N   : initial phase of the maximum doppler frequency sinusoid

        :return:
            h       : complex fading vector
    """
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


def plot_constellation(data):
    image = plt.figure(figsize=(6, 6))
    model_parameters['n_channel'] = 1
    for signals, marker, color, size, *index in data:
        signals = signals.cpu().detach().numpy()

        for i in range(0, model_parameters['n_channel']):
            plt.scatter(signals[:, i], signals[:, i + model_parameters['n_channel']], marker=marker, c=color, s=size,
                        label=('Autoencoder' + str(index[0]) if len(index) and i == 0 else ""))
            break



    # image.axes[0].set_xticks(list(frange(-9, 9, 1)))
    # image.axes[0].set_yticks(list(frange(-9, 9, 1)))
    #
    # plt.xlim([-4, 4])
    # plt.ylim([-4, 4])

    plt.legend()

    image.axes[0].axhline(0, linestyle='-', color='gray', alpha=0.5)
    image.axes[0].axvline(0, linestyle='-', color='gray', alpha=0.5)
    plt.xlabel('In-Phase')
    plt.ylabel('Quadrature')

    # return plt
    plt.show()


def plot_constellations():
    with torch.no_grad():
        n_samples = 1
        models = []
        for i in range(model_parameters['n_user']):
            model = Wireless_Autoencoder(model_parameters['m'], model_parameters['n_channel'], i + 1).to(device)
            model.load()
            model.eval()
            models.append(model)
        encodes = []
        for i in range(model_parameters['n_user']):
            for j in range(model_parameters['m']):
                j = 5
                x = torch.sparse.torch.eye(model_parameters['m']).index_select(dim=0, index=torch.ones(n_samples).int() * j).to(device)
                encode = models[i].encoder(x)
                # encode = models[0].channel(encode)
                encodes.append(encode)
                break


        encodes = torch.stack(encodes)
        encodes.transpose_(0, 1)
        signals = encodes


        # h = torch.randn((n_samples, model_parameters['n_user'], model_parameters['n_user']),
        #                 dtype=torch.cfloat).to(device)
        # signals = models[0].channel(encodes, ebno=channel_parameters['ebno'], h=h)

        ''' PCA (Dimensionality reduction) - BEGIN '''
        encodes.transpose_(1, 0)
        encodes = encodes.reshape(-1, model_parameters['n_channel'] * 2)
        pca = decomposition.PCA(n_components=2)
        signals_pca = pca.fit_transform(encodes.cpu())
        signals_pca = signals_pca * 4


        tmp_signals = signals_pca
        signals = []
        for signal in tmp_signals:
            signal = np.expand_dims(signal, axis=0)
            signal = np.repeat(signal, 200, axis=0)
            signal = torch.from_numpy(signal).to(device)
            encode = models[0].channel(signal)
            # encode = signal
            signals.append(encode)
        signals = torch.stack(signals)
        # signals = torch.from_numpy(signals)
        signals.transpose_(0, 1)
        ''' PCA (Dimensionality reduction) - END '''

        plot_constellation([(signals[:, 0, :], 'o', 'tab:red', 20, 1), (signals[:, 1, :], 'o', 'tab:blue', 20, 2), (signals[:, 2, :], 'o', 'tab:green', 20, 3), (signals[:, 3, :], 'o', 'tab:orange', 20, 4)])


def bler_chart(model, test_ds, manual_seed=None, return_bler=False):
    if channel_parameters['channel_type'] == 'rayleigh':
        ebno_range = list(frange(0, 22, 5))
    elif channel_parameters['channel_type'] == 'rician':
        ebno_range = list(frange(0, 22, 5))
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
            # print('SNR: {}, BER: {:.8f}'.format(ebno_range[j], ber[j]))

    print(ber)
    if return_bler:
        return ber
    # ber = [math.floor(x * 10000) / 10000 if x < 0.000999 else x for x in ber]
    plt.xlim(ebno_range[0], ebno_range[-1])
    # plt.ylim(ber[-1], 0.5)

    plt.plot(ebno_range, ber, '-ko', clip_on=False,
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
    # plt.savefig("results/png/autoencoder_bler_" + channel_parameters['channel_type'] + ".png")
    # plt.savefig("results/eps/autoencoder_bler_" + channel_parameters['channel_type'] + ".eps", format="eps")
    plt.show()


def blers_chart(models, test_dses, decoder_model, manual_seed=None):
    if decoder_model is not None:
        decoder_model.eval()

    if channel_parameters['channel_type'] == 'rayleigh':
        ebno_range = list(frange(0, 22, 5))
    elif channel_parameters['channel_type'] == 'rician':
        ebno_range = list(frange(0, 22, 5))
    else:
        ebno_range = list(frange(-4, 9, 1.5))
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

                signals = torch.stack(signals)
                signals = signals.transpose(0, 1)
                signals = signals.contiguous().view(signals.size()[0], -1)

                for i in range(model_parameters['n_user']):
                    # demodulate = models[i].demodulate(signals[i])
                    with torch.no_grad():
                        decode = decoder_model(signals, i)
                        _, demodulate = torch.max(decode, dim=1)

                    bers[i][j] = torch.sum(demodulate.detach().cpu() != test_dses[i][:][1]).item() / len(demodulate)
            else:
                encodes = []
                for i in range(model_parameters['n_user']):
                    x = test_dses[i][:][0].to(device)
                    encode = models[i].encoder(x)
                    encodes.append(encode)

                encodes = torch.stack(encodes)
                encodes.transpose_(0, 1)

                if channel_parameters['channel_type'] == 'rayleigh':
                    h = torch.randn((x.size()[0], model_parameters['n_user'], model_parameters['n_user']), dtype=torch.cfloat).to(device)
                else:
                    std = (1 / (channel_parameters['channel_k'] + 1)) ** 0.5
                    mean = (channel_parameters['channel_k'] / 2 * (channel_parameters['channel_k'] + 1)) ** 0.5
                    h = torch.normal(mean, std, (x.size()[0], model_parameters['n_user'], model_parameters['n_user']), dtype=torch.cfloat).to(device)

                # signals = []
                # for i in range(model_parameters['n_user']):
                #     signals.append(models[i].channel(encodes, ebno=ebno_range[j], h=h)[:, i, :])
                # signals = torch.stack(signals).transpose(1, 0)
                signals = models[0].channel(encodes, ebno=ebno_range[j], h=h)

                signals = signals.view((-1, model_parameters['n_user'], model_parameters['n_channel'], 2))
                signals = torch.view_as_complex(signals)
                transform = torch.inverse(h) @ signals
                signals = torch.view_as_real(transform).view(transform.size()[0], -1)

                for i in range(model_parameters['n_user']):
                    # demodulate = models[i].demodulate(signals, torch.view_as_real(h).view(-1, 2 * model_parameters[
                    #     'n_user'] * model_parameters['n_user']))
                    with torch.no_grad():
                        # decode = decoder_model(signals, i, torch.view_as_real(h).view(-1, 2 * model_parameters['n_user'] * model_parameters['n_user']))
                        decode = decoder_model(signals, i)
                        _, demodulate = torch.max(decode, dim=1)
                    bers[i][j] = torch.sum(demodulate.detach().cpu() != test_dses[i][:][1]).item() / len(demodulate)


    avg_ber = np.array(bers)
    avg_ber = avg_ber.mean(axis=0)
    avg_ber = avg_ber.tolist()
    # print(avg_ber)
    # exit(1)

    if True:
        '''
            Autoencoder saved results for demonstration purposes
        '''
        if channel_parameters['channel_type'] == 'rayleigh':
            bers = [[], [], [], [], [], []]
            bers[0] = [0.33905273437500005, 0.148203125, 0.0540625, 0.017705078125000002, 0.005615234375]
            bers[1] = [0.363525390625, 0.1622216796875, 0.05802734375, 0.0195068359375, 0.006201171875]
            bers[2] = [0.1451, 0.06388, 0.02278, 0.00746, 0.00248]
            bers[3] = [0.26044, 0.11938, 0.0413, 0.0135, 0.00454]
            bers[4] = [0.09048828125, 0.0356640625, 0.01224609375, 0.00390625, 0.0012890625]
            bers[5] = [0.169453125, 0.06681640625, 0.02337890625, 0.0073046875, 0.00232421875]
        elif channel_parameters['channel_type'] == 'rician':
            # bers = [[], [], [], [], [], []]
            bers = [[], [], [], []]
            bers[0] = [0.067451171875, 0.010107421875, 0.002109375, 0.000517578125, 0.000166015625]
            bers[1] = [0.1316650390625, 0.0232080078125, 0.0035205078125, 0.0008935546875, 0.0003125]
            bers[2] = [0.04341796875, 0.01439453125, 0.0040234375, 0.00140625, 0.00041015625]
            bers[3] = [0.08025390625, 0.02673828125, 0.00830078125, 0.00265625, 0.0008984375]
            # bers[4] = [0.082109375, 0.01890625, 0.00310546875, 0.00080078125, 0.00017578125]
            # bers[5] = [0.15662109375, 0.0340234375, 0.00615234375, 0.00166015625, 0.0003515625]
        else:
            bers = [[], [], [], []]
            bers[0] = [0.294208984375, 0.202646484375, 0.118037109375, 0.062529296875, 0.0254296875, 0.008779296875, 0.002001953125, 0.000439453125, 1.953125e-05]
            bers[1] = [0.28151855468750003, 0.189130859375, 0.11747070312499999, 0.061528320312499996, 0.0279150390625, 0.010634765625, 0.0032324218750000005, 0.0006982421875, 0.00015136718750000003]
            bers[2] = [0.1866, 0.14342, 0.10414, 0.06588, 0.03672, 0.01672, 0.00622, 0.00114, 0.00022]
            bers[3] = [0.34154, 0.27248, 0.202, 0.13278, 0.07924, 0.03576, 0.01344, 0.00296, 0.0003]

        line_styles = ['-bo', '-rD', '--k', '-.k', '--g', ':g']
        labels = ['2-user', '4-user', 'Simulated BPSK', 'Simulated QPSK', 'Autoencoder BPSK', 'Autoencoder QPSK']
        dash_style = (6, 8)
        for i in range(2):
            labels[i] = "Autoencoder(" + str(model_parameters['n_channel']) + "," + str(
                model_parameters['k']) + ")  " + labels[i]
        plt.xlim(ebno_range[0], ebno_range[-1])
        plt.ylim(min(ber[-1] for ber in bers) / 5, 0.5)
        for index, ber in enumerate(bers):
            if index == 2 or index == 4:
                extra_args = {'dashes': dash_style}
            else:
                extra_args = {}
            plt.plot(ebno_range, ber, line_styles[index], clip_on=False,
                     label=labels[index], **extra_args)
    else:
        plt.xlim(ebno_range[0], ebno_range[-1])
        plt.ylim(min(ber[-1] for ber in bers) / 5, 0.5)
        for index, ber in enumerate(bers):
            plt.plot(ebno_range, ber, '-o', clip_on=False,
                     label="Autoencoder" + str(index + 1) + " (" + str(model_parameters['n_channel']) + "," + str(
                         model_parameters['k']) + ")" + (
                               (" - seed: " + str(manual_seed)) if (manual_seed is not None) else ""))

    plt.yscale('log')
    plt.xlabel("$E_{b}/N_{0}$ (dB)", fontsize=18)
    plt.ylabel('Block Error Rate', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid()
    plt.legend(loc="lower left", ncol=1, fontsize=15)
    plt.tight_layout()
    plt.savefig("results/png/wireless_autoencoders(" + str(model_parameters['n_channel']) + "," + str(model_parameters['k']) + ")_" + channel_parameters['channel_type'] + ".png")
    plt.savefig("results/eps/wireless_autoencoders(" + str(model_parameters['n_channel']) + "," + str(model_parameters['k']) + ")_" + channel_parameters['channel_type'] + ".eps", format="eps")
    plt.show()
    plt.close()


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
    plt.savefig("results/png/losses(" + str(model_parameters['n_channel']) + "," + str(model_parameters['k']) + ").png")
    plt.show()
    plt.close()


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
    plt.close()
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
