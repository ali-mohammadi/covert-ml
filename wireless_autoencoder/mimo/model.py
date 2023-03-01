import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from wireless_autoencoder.mimo.parameters import model_parameters
from shared_parameters import channel_parameters, system_parameters

if model_parameters['device'] == 'auto':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(model_parameters['device'])

torch.manual_seed(model_parameters['seed'])

class LayerShape(nn.Module):
    def __init__(self, terminate: bool = False):
        super(LayerShape, self).__init__()
        self.terminate = terminate

    def forward(self, x):
        print('Layer shape:', x.size())
        if self.terminate:
            sys.exit()
        return x


class Wireless_Decoder(nn.Module):
    def __init__(self):
        torch.manual_seed(model_parameters['seed'])
        super(Wireless_Decoder, self).__init__()
        in_channel = model_parameters['m']
        compressed_dimension = model_parameters['n_user'] * model_parameters['n_channel'] * 2

        self.predecoder = nn.Sequential(
            nn.Linear(compressed_dimension, compressed_dimension),
            nn.Tanh(),
            nn.Unflatten(1, (1, -1)),
            nn.Conv1d(1, model_parameters['n_channel'], 2, 1),
            nn.Tanh(),
            nn.Conv1d(model_parameters['n_channel'], model_parameters['n_channel'], (4 if model_parameters['n_channel'] > 4 else 2), 2),
            nn.Tanh(),
            nn.Conv1d(model_parameters['n_channel'], model_parameters['n_channel'], 2, 1),
            nn.Tanh(),
            nn.Conv1d(model_parameters['n_channel'], model_parameters['n_channel'], 2, 1),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(model_parameters['n_channel'] * (model_parameters['n_user'] * model_parameters['n_channel'] - 4), compressed_dimension * 2),
            # nn.Linear(12, compressed_dimension * 2),
            nn.Tanh(),
            nn.Linear(compressed_dimension * 2, compressed_dimension * 2),
            nn.Tanh()
        )

        self.decoder = nn.ModuleList([nn.Sequential(
                nn.Linear(compressed_dimension * 2, compressed_dimension),
                nn.Tanh(),
                nn.Linear(compressed_dimension, in_channel)
            ).to(device) for _ in range(model_parameters['n_user'])])

    def forward(self, x, i):
        x = self.predecoder(x)
        return self.decoder[i](x)

    def load(self):
        self.load_state_dict(
            torch.load('../../models/' + system_parameters['system_type'] + '/' + channel_parameters[
                'channel_type'] + '/' + str(model_parameters['n_user']) + '-user/wireless_autodecoder(' + str(model_parameters['n_channel']) + ',' + str(
                model_parameters['k']) + ').pt'))

    def save(self):
        self.eval()
        torch.save(self.state_dict(),
                   '../../models/' + system_parameters['system_type'] + '/' + channel_parameters[
                       'channel_type'] + '/' + str(model_parameters['n_user']) + '-user/wireless_autodecoder(' + str(model_parameters['n_channel']) + ',' + str(
                       model_parameters['k']) + ').pt')

class Wireless_Autoencoder(nn.Module):
    def __init__(self, in_channel=model_parameters['m'], compressed_dimension=model_parameters['n_channel'], index=0):
        torch.manual_seed(model_parameters['seed'])
        super(Wireless_Autoencoder, self).__init__()
        self.index = index
        self.in_channel = in_channel
        self.compressed_dimension = compressed_dimension
        compressed_dimension = 2 * compressed_dimension

        self.encoder = nn.Sequential(
            nn.Linear(in_channel, in_channel),
            nn.ELU(),
            nn.Linear(in_channel, compressed_dimension),
            nn.ELU(),
            nn.Unflatten(1, (1, -1)),
            nn.Conv1d(1, model_parameters['n_channel'], 2, 1),
            nn.Tanh(),
            nn.Conv1d(model_parameters['n_channel'], model_parameters['n_channel'], (4 if model_parameters['n_channel'] > 4 else 2), 2),
            nn.Tanh(),
            nn.Conv1d(model_parameters['n_channel'], model_parameters['n_channel'], 2, 1),
            nn.Tanh(),
            nn.Conv1d(model_parameters['n_channel'], model_parameters['n_channel'], 2, 1),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(max(model_parameters['n_channel'] - 4, 1) * model_parameters['n_channel'], compressed_dimension * 2),
            # nn.Linear(12, compressed_dimension * 2),
            nn.Tanh(),
            nn.Linear(compressed_dimension * 2, compressed_dimension),
            nn.BatchNorm1d(compressed_dimension, affine=False)
        )

        # if channel_parameters['channel_type'] != 'awgn':
        #     '''
        #         Parameter Estimation with Dense layers
        #     '''
        #     self.parameter_est = nn.Sequential(
        #         nn.Linear(compressed_dimension, compressed_dimension * 2),
        #         nn.ELU(),
        #         nn.Linear(compressed_dimension * 2, compressed_dimension * 4),
        #         nn.Tanh(),
        #         nn.Linear(compressed_dimension * 4, int(compressed_dimension / 2)),
        #         nn.Tanh(),
        #         nn.Linear(int(compressed_dimension / 2), 2)
        #     )
        #
        #     '''
        #         Parameter Embedding with Dense layers
        #     '''
        #     self.parameter_emb = nn.Sequential(
        #         nn.Linear(2 * model_parameters['n_user'] * model_parameters['n_user'], 32),
        #         nn.Tanh(),
        #         nn.Linear(32, 64),
        #         nn.Sigmoid(),
        #         nn.Linear(64, 2),
        #     )


        '''
            Decoder with Dense layers
        # '''
        # self.decoder = nn.Sequential(
        #     nn.Linear(compressed_dimension, compressed_dimension * 2),
        #     nn.ReLU(),
        #     nn.Linear(compressed_dimension * 2, compressed_dimension * 2),
        #     nn.ReLU(),
        #     nn.Linear(compressed_dimension * 2, in_channel),
        # )

        '''
            Decoder with Conv layers
        '''
        # self.decoder = nn.Sequential(
        #     nn.Linear(compressed_dimension, compressed_dimension),
        #     nn.Tanh(),
        #     nn.Unflatten(1, (1, -1)),
        #     nn.Conv1d(1, model_parameters['n_channel'], 2, 1),
        #     nn.Tanh(),
        #     nn.Conv1d(model_parameters['n_channel'], model_parameters['n_channel'], (4 if model_parameters['n_channel'] > 4 else 2), 2),
        #     nn.Tanh(),
        #     nn.Conv1d(model_parameters['n_channel'], model_parameters['n_channel'], 2, 1),
        #     nn.Tanh(),
        #     nn.Conv1d(model_parameters['n_channel'], model_parameters['n_channel'], 2, 1),
        #     nn.Tanh(),
        #     nn.Flatten(),
        #     nn.Linear(max(model_parameters['n_channel'] - 4, 1) * model_parameters['n_channel'], compressed_dimension),
        #     nn.Tanh(),
        #     nn.Linear(compressed_dimension, compressed_dimension),
        #     nn.Tanh(),
        #     nn.Linear(compressed_dimension, in_channel),
        # )

    def forward(self, x):
        x = self.encoder(x)

        '''
            Strict normalization on each symbol (Not needed when BatchNorm1d layer is used)
        '''
        # n = ((self.compressed_dimension) ** 0.5) * (x / x.norm(dim=-1)[:, None])

        n = x
        c = self.channel(n, channel_parameters['channel_type'])
        x = self.decoder_net(c)
        return x

    def channel(self, x, channel=channel_parameters['channel_type'], r=model_parameters['r'],
                ebno=channel_parameters['ebno'], k=channel_parameters['channel_k'], h=None):
        if channel == 'awgn':
            return self.awgn(x, r, ebno)
        if channel == 'rayleigh':
            return self.rayleigh(x, r, ebno, h)
        if channel == 'rician':
            return self.rician(x, r, ebno, k)
        return x

    def awgn(self, x, r, ebno):
        ebno = 10.0 ** (ebno / 10.0)
        noise = torch.randn(*x.size(), requires_grad=False) / ((2 * r * ebno) ** 0.5)
        noise = noise.to(device)
        return x + noise

    def rayleigh(self, x, r, ebno, h):
        if ebno is not None:
            ebno = 10.0 ** (ebno / 10.0)
            noise = torch.randn((x.size()[0], x.size()[2]), requires_grad=False) / ((2 * r * ebno) ** 0.5)
            noise = noise.unsqueeze(1).to(device)
        else:
            noise = torch.zeros(*x.size(), requires_grad=False)
            noise = noise.to(device)

        if ebno is not None:
            n_channel = int(x.size()[2] / 2)

            x = x.view((-1, model_parameters['n_user'], n_channel, 2))
            x = torch.view_as_complex(x)
        else:
            n_channel = int(x.size()[1] / 2)
            x = x.view((-1, n_channel, 2))
            x = torch.view_as_complex(x)

        if h is None:
            if channel_parameters['channel_taps'] is not None:
                fading_batch = torch.randn((x.size()[0], channel_parameters['channel_taps']), dtype=torch.cfloat).to(
                    device)
                padded_signals = torch.nn.functional.pad(x, (
                    int(channel_parameters['channel_taps'] - 1), int(channel_parameters['channel_taps'] - 1), 0, 0))
                output_signal = torch.zeros(model_parameters['batch_size'],
                                            model_parameters['n_channel'] + fading_batch.size()[1] - 1,
                                            dtype=torch.cfloat).to(device)
                for i in range(output_signal.size()[1]):
                    flipped_batch_fading = torch.flip(fading_batch, [1])
                    for j in range(fading_batch.size()[1]):
                        output_signal[:, i] += flipped_batch_fading[:, j] * padded_signals[:, i + j]

                output_signal = output_signal[:, 0:-int(channel_parameters['channel_taps'] - 1)]
            else:
                fading_batch = torch.randn((x.size()[0], 1), dtype=torch.cfloat).to(device)
                output_signal = x * fading_batch
        else:
            if ebno is not None:
                output_signal = h @ x
            else:
                output_signal = h * x
                return torch.view_as_real(output_signal).view(-1, n_channel * 2) + noise

        return torch.view_as_real(output_signal).view(-1, model_parameters['n_user'],
                                                      n_channel * 2) + noise

    @staticmethod
    def transform(x, h):
        x = x.view((-1, model_parameters['n_channel'], 2))
        x = torch.view_as_complex(x)
        h = torch.view_as_complex(h).view(-1, 1)
        return torch.view_as_real(x / h).view(-1, model_parameters['n_channel'] * 2)

    def demodulate(self, x, h=None):
        with torch.no_grad():
            x = self.decoder_net(x, h)
            _, preds = torch.max(x, dim=1)
            return preds

    def decoder_net(self, x, h=None):
        if channel_parameters['channel_type'] == 'rayleigh' or channel_parameters['channel_type'] == 'rician':
            h = self.parameter_emb(h)
            t = self.transform(x[:, self.index - 1, :], h)
            return self.decoder(t)
        return self.decoder(x)

    def load(self):
        self.load_state_dict(
            torch.load('../../models/' + system_parameters['system_type'] + '/' + channel_parameters[
                'channel_type'] + '/' + str(model_parameters['n_user']) + '-user/wireless_autoencoder(' + str(model_parameters['n_channel']) + ',' + str(
                model_parameters['k']) + ')_' + str(self.index) + '.pt'))

    def save(self, train_ds, test_ds):
        self.eval()

        torch.save(self.state_dict(),
                   '../../models/' + system_parameters['system_type'] + '/' + channel_parameters[
                       'channel_type'] + '/' + str(model_parameters['n_user']) + '-user/wireless_autoencoder(' + str(model_parameters['n_channel']) + ',' + str(
                       model_parameters['k']) + ')_' + str(self.index) + '.pt')
        if self.index == 1:
            torch.save(train_ds, '../../data/' + system_parameters['system_type'] + '/' + channel_parameters[
                'channel_type'] + '/' + str(model_parameters['n_user']) + '-user/wireless_autoencoder(' + str(model_parameters['n_channel']) + ',' + str(
                model_parameters['k']) + ')_train.pt')
            torch.save(test_ds, '../../data/' + system_parameters['system_type'] + '/' + channel_parameters[
                'channel_type'] + '/' + str(model_parameters['n_user']) + '-user/wireless_autoencoder(' + str(model_parameters['n_channel']) + ',' + str(
                model_parameters['k']) + ')_test.pt')

        '''
            To save the encoder signals in separate data sets
        '''
        # self.to('cpu')
        # encoded_signal = self.encoder(test_ds[:][0])
        # signal_ds = TensorDataset(encoded_signal, test_ds[:][1])
        # torch.save(signal_ds,
        #            '../data/' + channel_parameters['channel_type'] + '/wireless_autoencoder(' + str(model_parameters['n_channel']) + ',' + str(model_parameters['k']) + ')_signals_test.pt')
        #
        # encoded_signal = self.encoder(train_ds[:][0])
        # signal_ds = TensorDataset(encoded_signal, train_ds[:][1])
        # torch.save(signal_ds,
        #            '../data/' + channel_parameters['channel_type'] + '/wireless_autoencoder(' + str(model_parameters['n_channel']) + ',' + str(model_parameters['k']) + ')_signals_train.pt')
        #
        # self.to(device)
