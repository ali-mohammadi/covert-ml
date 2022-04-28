import torch
import torch.nn as nn

from wireless_autoencoder.parameters import model_parameters
from wireless_covert.parameters import covert_parameters

if model_parameters['device'] == 'auto':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(model_parameters['device'])


class Alice(nn.Module):
    def __init__(self):
        super(Alice, self).__init__()
        self.encode = nn.Sequential(
            nn.Linear(covert_parameters['n_channel'] * 2 + covert_parameters['m'], covert_parameters['n_channel'] * 4 + covert_parameters['m'] * 2),
            nn.ReLU(inplace=True),
            nn.Linear(covert_parameters['n_channel'] * 4 + covert_parameters['m'] * 2, covert_parameters['n_channel'] * 4 + covert_parameters['m'] * 2),
            nn.ReLU(inplace=True),
            nn.Linear(covert_parameters['n_channel'] * 4 + covert_parameters['m'] * 2, covert_parameters['n_channel'] * covert_parameters['m']),
            nn.CELU(),
            nn.Linear(covert_parameters['n_channel'] * 2, covert_parameters['n_channel'] * 2),
        )

    def forward(self, x, m):
        m_one_hot_sparse = torch.sparse.torch.eye(2).to(device)
        m_one_hot = m_one_hot_sparse.index_select(dim=0, index=m).to(device)

        encoded_signal = self.encode(torch.cat([x, m_one_hot], dim=1).to(device))

        '''
            Keeping covert signals power same as noise
        '''
        # ebno = 10.0 ** (model_parameters['ebno'] / 10.0)
        # norm_signal = (model_parameters['n_channel'] ** 0.5) * (encoded_signal / encoded_signal.norm(dim=-1)[:, None])
        # norm_signal = norm_signal / (2 * model_parameters['r'] * ebno ** 0.5)

        return encoded_signal


class Willie(nn.Module):
    def __init__(self):
        super(Willie, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(model_parameters['n_channel'] * 2, model_parameters['n_channel'] * 2),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (1, -1)),
            nn.Conv1d(1, int(model_parameters['n_channel'] / 2), 2, 2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(int(model_parameters['n_channel'] / 2), model_parameters['n_channel'], 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(model_parameters['n_channel'], model_parameters['n_channel'], 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(model_parameters['n_channel'], 1, 1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(model_parameters['n_channel'], 4),
            nn.LeakyReLU(0.2),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class Bob(nn.Module):
    def __init__(self):
        super(Bob, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(covert_parameters['n_channel'] * 2, covert_parameters['n_channel'] * 2),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (1, -1)),
            nn.Conv1d(1, int(covert_parameters['n_channel'] / 2), 2, 2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(int(covert_parameters['n_channel'] / 2), covert_parameters['n_channel'], 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(covert_parameters['n_channel'], covert_parameters['n_channel'], 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(covert_parameters['n_channel'], 1, 1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(covert_parameters['n_channel'], covert_parameters['m'] * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(covert_parameters['m'] * 2, covert_parameters['m']),
        )

    def forward(self, x):
        return self.net(x)