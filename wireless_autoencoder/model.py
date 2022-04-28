import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import sys

from wireless_autoencoder.parameters import model_parameters

if model_parameters['device'] == 'auto':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(model_parameters['device'])

class LayerShape(nn.Module):
    def forward(self, x):
        print('Layer shape:', x.size())
        sys.exit()

class Wireless_Autoencoder(nn.Module):
    def __init__(self, in_channel=model_parameters['m'], compressed_dimension=model_parameters['n_channel']):
        super(Wireless_Autoencoder, self).__init__()
        self.batch_number = 0
        self.fading = None
        self.in_channel = in_channel
        self.compressed_dimension = compressed_dimension
        compressed_dimension = 2 * compressed_dimension

        self.encoder = nn.Sequential(
            nn.Linear(in_channel, in_channel),
            nn.ELU(inplace=True),
            nn.Linear(in_channel, compressed_dimension),
            nn.BatchNorm1d(compressed_dimension)
        )

        torch.nn.init.xavier_uniform_(self.encoder[0].weight)

        '''
            Decoder with Dense layers
        '''
        # self.decoder = nn.Sequential(
        #     nn.Linear(int(compressed_dimension / 2), int(compressed_dimension / 2)),
        #     nn.ELU(),
        #     nn.Linear(int(compressed_dimension / 2), in_channel),
        # )

        '''
            Parameter Estimation with Dense layers
        '''
        # self.parameter_est = nn.Sequential(
        #         nn.Linear(model_parameters['n_channel'] * 2, model_parameters['n_channel'] * 4),
        #         nn.ReLU(),
        #         nn.Linear(model_parameters['n_channel'] * 4, model_parameters['n_channel'] * 4),
        #         nn.ReLU(),
        #         nn.Linear(model_parameters['n_channel'] * 4, model_parameters['n_channel'] * 2),
        # )

        '''
            Parameter Estimation with Conv layers
        '''
        self.parameter_est = nn.Sequential(
            nn.Unflatten(1, (1, -1)),
            nn.Conv1d(1, model_parameters['n_channel'] * 2, 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(model_parameters['n_channel'] * 2, model_parameters['n_channel'] * 2, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(model_parameters['n_channel'] * 2, model_parameters['n_channel'] * 2, 2, 4),
            nn.ReLU(inplace=True),
            nn.Conv1d(model_parameters['n_channel'] * 2, model_parameters['n_channel'] * 2, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(model_parameters['n_channel'] * 2, model_parameters['n_channel'] * 2, 3, 1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(model_parameters['n_channel'] * 2, model_parameters['n_channel'] * 2),
            # nn.Tanh(),
            # nn.ReLU(inplace=True),
            # nn.Linear(model_parameters['n_channel'] * 2, 2)
        )

        '''
            Decoder with Conv layers
        '''
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (1, -1)),
            nn.Conv1d(1, int(model_parameters['n_channel']), 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(int(model_parameters['n_channel']), int(model_parameters['n_channel']), 2, 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(int(model_parameters['n_channel']), int(model_parameters['n_channel']), 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(int(model_parameters['n_channel']), int(model_parameters['n_channel']), 1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(model_parameters['n_channel'] * model_parameters['n_channel'], compressed_dimension),
            nn.ReLU(inplace=True),
            nn.Linear(compressed_dimension, in_channel),
            nn.ELU(),
            nn.Linear(in_channel, in_channel),
        )

    def forward(self, x):
        x = self.encoder(x)
        '''
            Strict normalization on each symbol (Not needed when BatchNorm1d layer is used)
        '''
        # n = ((self.compressed_dimension) ** 0.5) * (x / x.norm(dim=-1)[:, None])
        n = x
        c = self.channel(n, model_parameters['channel_type'])
        if model_parameters['channel_type'] == 'rayleigh':
            # t = c
            h = self.parameter_est(c)
            t = self.transform(c, h)
        if model_parameters['channel_type'] == 'awgn':
            t = c
        x = self.decoder(t)
        return x

    def channel(self, x, channel=model_parameters['channel_type'], r=model_parameters['r'], ebno=model_parameters['ebno']):
        if channel == 'awgn':
            return self.awgn(x, r, ebno)
        if channel == 'rayleigh':
            return self.rayleigh(x, r, ebno)
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

        x = x.view((-1, model_parameters['n_channel'], 2))
        x = torch.view_as_complex(x)

        fading_batch = self.fading[self.batch_number * x.size()[0]:(self.batch_number * x.size()[0]) + x.size()[0]].to(device)
        # fading_batch = torch.randn((model_parameters['batch_size'], 1), dtype=torch.cfloat).to(device)


        if model_parameters['channel_taps'] != None:
            padded_signals = torch.nn.functional.pad(x, (2, 2, 0, 0))
            output_signal = torch.zeros(model_parameters['batch_size'], model_parameters['n_channel'] + fading_batch.size()[1] - 1, dtype=torch.cfloat).to(device)
            for i in range(output_signal.size()[1]):
                flipped_batch_fading = torch.flip(fading_batch, [1])
                for j in range(fading_batch.size()[1]):
                    output_signal[:, i] += flipped_batch_fading[:, j] * padded_signals[:, i + j]

            output_signal = output_signal[:, 0:-2]
        else:
            output_signal = x * fading_batch

        return torch.view_as_real(output_signal).view(-1, model_parameters['n_channel'] * 2) + noise

    def transform(self, x, h):
        x = x.view((-1, model_parameters['n_channel'], 2))
        x = torch.view_as_complex(x)
        h = h.view((-1, model_parameters['n_channel'], 2))
        h = torch.view_as_complex(h)
        # h = self.fading[self.batch_number * model_parameters['batch_size']:(self.batch_number * model_parameters['batch_size']) + model_parameters['batch_size']]
        return torch.view_as_real(x / h).view(-1, model_parameters['n_channel'] * 2)

    def demodulate(self, x):
        with torch.no_grad():
            x = self.decoder_net(x)
            _, preds = torch.max(x, dim=1)
            return preds

    def decoder_net(self, x):
        if model_parameters['channel_type'] == 'rayleigh':
            h = self.parameter_est(x)
            x = self.transform(x, h)
        return self.decoder(x)

    def load(self):
        self.load_state_dict(
            torch.load('../models/' + model_parameters['channel_type'] + '/wireless_autoencoder_' + str(model_parameters['n_channel']) + '-' + str(
                model_parameters['k']) + '.pth'))

    def save(self, train_ds, test_ds, save_signals=False,):
        self.eval()

        torch.save(self.state_dict(),
                   '../models/' + model_parameters['channel_type'] + '/wireless_autoencoder_' + str(model_parameters['n_channel']) + '-' + str(
                       model_parameters['k']) + '.pth')
        torch.save(train_ds, '../data/' + model_parameters['channel_type'] + '/train_' + str(
            model_parameters['n_channel']) + '-' + str(
            model_parameters['k']) + '.pt')
        torch.save(test_ds, '../data/' + model_parameters['channel_type'] + '/test_' + str(
            model_parameters['n_channel']) + '-' + str(
            model_parameters['k']) + '.pt')

        if save_signals:
            self.to('cpu')
            encoded_signal = self.encoder(test_ds[:][0])
            # norm_signal = ((model_parameters['n_channel']) ** 0.5) * (
            #             encoded_signal / encoded_signal.norm(dim=-1)[:, None])
            signal_ds = TensorDataset(encoded_signal, test_ds[:][1])
            torch.save(signal_ds,
                       '../data/' + model_parameters['channel_type'] + '/test_signals_' + str(
                           model_parameters['n_channel']) + '-' + str(
                           model_parameters['k']) + '.pt')


            encoded_signal = self.encoder(train_ds[:][0])
            # norm_signal = ((model_parameters['n_channel']) ** 0.5) * (
            #             encoded_signal / encoded_signal.norm(dim=-1)[:, None])
            signal_ds = TensorDataset(encoded_signal, train_ds[:][1])
            torch.save(signal_ds,
                       '../data/' + model_parameters['channel_type'] + '/train_signals_' + str(
                           model_parameters['n_channel']) + '-' + str(
                           model_parameters['k']) + '.pt')

            self.to(device)