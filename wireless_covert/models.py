import torch
import torch.nn as nn
import sys

from wireless_covert.parameters import covert_parameters

if covert_parameters['device'] == 'auto':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(covert_parameters['device'])

torch.manual_seed(covert_parameters['seed'])


class LayerShape(nn.Module):
    def __init__(self, terminate: bool = False):
        super(LayerShape, self).__init__()
        self.terminate = terminate

    def forward(self, x):
        print('Layer shape:', x.size())
        if self.terminate:
            sys.exit()
        return x


class Alice(nn.Module):
    def __init__(self):
        super(Alice, self).__init__()
        self.encode = nn.Sequential(
            nn.Linear(covert_parameters['n_channel'] * 2 + covert_parameters['m'], covert_parameters['n_channel'] * 4 + covert_parameters['m'] * 2),
            nn.ReLU(inplace=True),
            nn.Linear(covert_parameters['n_channel'] * 4 + covert_parameters['m'] * 2, covert_parameters['n_channel'] * 4 + covert_parameters['m'] * 2),
            nn.ReLU(inplace=True),
            nn.Linear(covert_parameters['n_channel'] * 4 + covert_parameters['m'] * 2, covert_parameters['n_channel'] * covert_parameters['m']),
            nn.ReLU(inplace=True),
            nn.Linear(covert_parameters['n_channel'] * covert_parameters['m'], covert_parameters['n_channel'] * 2),
            nn.Tanh(),
        )

    def forward(self, x, m):
        m_one_hot_sparse = torch.sparse.torch.eye(covert_parameters['m']).to(device)
        m_one_hot = m_one_hot_sparse.index_select(dim=0, index=m).to(device)
        encoded_signal = self.encode(torch.cat([x, m_one_hot], dim=1).to(device))

        '''
            Keeping covert signals power same as noise
        '''
        # ebno = 10.0 ** (covert_parameters['ebno'] / 10.0)
        # norm_signal = (covert_parameters['n_channel'] ** 0.5) * (encoded_signal / encoded_signal.norm(dim=-1)[:, None])
        # norm_signal = norm_signal / (2 * covert_parameters['r'] * ebno ** 0.5)

        return encoded_signal


class Willie(nn.Module):
    def __init__(self):
        super(Willie, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(covert_parameters['n_channel'] * 2, covert_parameters['n_channel'] * 2),
            nn.Tanh(),
            nn.Unflatten(1, (1, -1)),
            nn.Conv1d(1, covert_parameters['n_channel'], 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(covert_parameters['n_channel'], covert_parameters['n_channel'], 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(covert_parameters['n_channel'], covert_parameters['n_channel'], 4, 2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(covert_parameters['n_channel'], covert_parameters['n_channel'], 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(covert_parameters['n_channel'], covert_parameters['n_channel'], 2, 1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(covert_parameters['n_channel'] * (covert_parameters['n_channel'] - 4), covert_parameters['n_channel'] * 2),
            nn.Tanh(),
            nn.Linear(covert_parameters['n_channel'] * 2, 1),
            nn.Sigmoid()
        )

        # self.net = nn.Sequential(
        #     nn.Linear(covert_parameters['n_channel'] * 2, covert_parameters['n_channel'] * 2),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(covert_parameters['n_channel'] * 2, covert_parameters['n_channel'] * 4),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(covert_parameters['n_channel'] * 4, covert_parameters['n_channel'] * 8),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(covert_parameters['n_channel'] * 8, 1),
        #     nn.Sigmoid()
        # )

    def forward(self, x):
        return self.net(x)


class Bob(nn.Module):
    def __init__(self):
        super(Bob, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(covert_parameters['n_channel'] * 2, covert_parameters['n_channel'] * 2),
            nn.Tanh(),
            nn.Unflatten(1, (1, -1)),
            nn.Conv1d(1, covert_parameters['n_channel'], 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(covert_parameters['n_channel'], covert_parameters['n_channel'], 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(covert_parameters['n_channel'], covert_parameters['n_channel'], 4, 2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(covert_parameters['n_channel'], covert_parameters['n_channel'], 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(covert_parameters['n_channel'], covert_parameters['n_channel'], 2, 1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(covert_parameters['n_channel'] * (covert_parameters['n_channel'] - 4), covert_parameters['n_channel'] * 2),
            nn.Tanh(),
            nn.Linear(covert_parameters['n_channel'] * 2, covert_parameters['m']),
        )

        # self.net = nn.Sequential(
        #     nn.Linear(covert_parameters['n_channel'] * 2, covert_parameters['n_channel'] * 2),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(covert_parameters['n_channel'] * 2, covert_parameters['n_channel'] * 4),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(covert_parameters['n_channel'] * 4, covert_parameters['n_channel'] * 8),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(covert_parameters['n_channel'] * 8, covert_parameters['m']),
        # )

    def forward(self, x):
        return self.net(x)