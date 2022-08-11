import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import sys

from wireless_autoencoder.parameters import model_parameters
from wireless_covert.parameters import covert_parameters
from channel_parameters import channel_parameters

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


class Wireless_Autoencoder(nn.Module):
    def __init__(self, in_channel=model_parameters['m'], compressed_dimension=model_parameters['n_channel']):
        super(Wireless_Autoencoder, self).__init__()
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
            nn.Conv1d(model_parameters['n_channel'], model_parameters['n_channel'], 4, 2),
            nn.Tanh(),
            nn.Conv1d(model_parameters['n_channel'], model_parameters['n_channel'], 2, 1),
            nn.Tanh(),
            nn.Conv1d(model_parameters['n_channel'], model_parameters['n_channel'], 2, 1),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(compressed_dimension * 2, compressed_dimension),
            nn.BatchNorm1d(compressed_dimension)
        )

        torch.nn.init.xavier_uniform_(self.encoder[0].weight)

        '''
            Parameter Estimation with Dense layers
        '''
        self.parameter_est = nn.Sequential(
            nn.Linear(compressed_dimension, compressed_dimension * 2),
            nn.ELU(),
            nn.Linear(compressed_dimension * 2, compressed_dimension * 4),
            nn.Tanh(),
            nn.Linear(compressed_dimension * 4, int(compressed_dimension / 2)),
            nn.Tanh(),
            nn.Linear(int(compressed_dimension / 2), 2)
        )

        '''
            Decoder with Dense layers
        '''
        # self.decoder = nn.Sequential(
        #     nn.Linear(compressed_dimension, compressed_dimension),
        #     nn.ReLU(),
        #     nn.Linear(compressed_dimension, in_channel),
        # )

        '''
            Decoder with Conv layers
        '''
        self.decoder = nn.Sequential(
            nn.Linear(compressed_dimension, compressed_dimension),
            nn.Tanh(),
            nn.Unflatten(1, (1, -1)),
            nn.Conv1d(1, model_parameters['n_channel'], 2, 1),
            nn.Tanh(),
            nn.Conv1d(model_parameters['n_channel'], model_parameters['n_channel'], 4, 2),
            nn.Tanh(),
            nn.Conv1d(model_parameters['n_channel'], model_parameters['n_channel'], 2, 1),
            nn.Tanh(),
            nn.Conv1d(model_parameters['n_channel'], model_parameters['n_channel'], 2, 1),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(compressed_dimension * 2, compressed_dimension),
            nn.Tanh(),
            nn.Linear(compressed_dimension, compressed_dimension),
            nn.Tanh(),
            nn.Linear(compressed_dimension, in_channel),
        )

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
                ebno=channel_parameters['ebno'], k=channel_parameters['channel_k']):

        if channel == 'awgn':
            return self.awgn(x, r, ebno)
        if channel == 'rayleigh':
            return self.rayleigh(x, r, ebno)
        if channel == 'rician':
            return self.rician(x, r, ebno, k)
        return x

    def awgn(self, x, r, ebno):
        ebno = 10.0 ** (ebno / 10.0)
        noise = torch.randn(*x.size(), requires_grad=False) / ((2 * r * ebno) ** 0.5)
        noise = noise.to(device)
        return x + noise

    def rayleigh(self, x, r, ebno):
        ebno = 10.0 ** (ebno / 10.0)
        noise = torch.randn(*x.size(), requires_grad=False) / ((2 * r * ebno) ** 0.5)
        noise = noise.to(device)

        n_channel = int(x.size()[1] / 2)

        x = x.view((-1, n_channel, 2))
        x = torch.view_as_complex(x)

        if channel_parameters['channel_taps'] is not None:
            fading_batch = torch.randn((x.size()[0], channel_parameters['channel_taps'] ), dtype=torch.cfloat).to(device)
            padded_signals = torch.nn.functional.pad(x, (int(channel_parameters['channel_taps'] - 1), int(channel_parameters['channel_taps'] - 1), 0, 0))
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

        return torch.view_as_real(output_signal).view(-1, n_channel * 2) + noise

    def rician(self, x, r, ebno, k=channel_parameters['channel_k']):
        ebno = 10.0 ** (ebno / 10.0)
        noise = torch.randn(*x.size(), requires_grad=False) / ((2 * r * ebno) ** 0.5)
        noise = noise.to(device)

        n_channel = int(x.size()[1] / 2)

        x = x.view((-1, n_channel, 2))
        x = torch.view_as_complex(x)

        std = (1 / 2 * (k + 1)) ** 0.5
        mean = (k / 2 * (k + 1)) ** 0.5
        fading_batch = torch.normal(mean, std, (x.size()[0], 1), dtype=torch.cfloat).to(device)

        return torch.view_as_real(x * fading_batch).view(-1, n_channel * 2) + noise

    def transform(self, x, h):
        x = x.view((-1, model_parameters['n_channel'], 2))
        x = torch.view_as_complex(x)
        '''
            Used when a vector of channel fading parameters is estimated for each transmission
        '''
        # h = h.view((-1, model_parameters['n_channel'], 2))
        '''
            Used when a single channel fading parameter is estimated for each transmission
        '''
        h = h.view((-1, 1, 2))

        h = torch.view_as_complex(h)
        '''
            Uncomment to set the 'h' to the actual parameters of the fading channel
        '''
        # h = self.fading[self.batch_number * model_parameters['batch_size']:(self.batch_number * model_parameters['batch_size']) + model_parameters['batch_size']]
        return torch.view_as_real(x / h).view(-1, model_parameters['n_channel'] * 2)

    def demodulate(self, x):
        with torch.no_grad():
            x = self.decoder_net(x)
            _, preds = torch.max(x, dim=1)
            return preds

    def decoder_net(self, x):
        if channel_parameters['channel_type'] == 'rayleigh' or channel_parameters['channel_type'] == 'rician':
            h = self.parameter_est(x)
            t = self.transform(x, h)
            return self.decoder(t)
        return self.decoder(x)

    def load(self):
        self.load_state_dict(
            torch.load('../models/' + channel_parameters['channel_type'] + '/wireless_autoencoder(' + str(model_parameters['n_channel']) + ',' + str(model_parameters['k']) + ').pt'))

    def save(self, train_ds, test_ds):
        self.eval()

        torch.save(self.state_dict(),
                   '../models/' + channel_parameters['channel_type'] + '/wireless_autoencoder(' + str(model_parameters['n_channel']) + ',' + str(model_parameters['k']) + ').pt')
        torch.save(train_ds, '../data/' + channel_parameters['channel_type'] + '/wireless_autoencoder(' + str(model_parameters['n_channel']) + ',' + str(model_parameters['k']) + ')_train.pt')
        torch.save(test_ds, '../data/' + channel_parameters['channel_type'] + '/wireless_autoencoder(' + str(model_parameters['n_channel']) + ',' + str(model_parameters['k']) + ')_test.pt')

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
