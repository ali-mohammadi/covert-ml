from shared_parameters import channel_parameters
from wireless_autoencoder.mimo.parameters import model_parameters

covert_parameters = {
    'n_channel': model_parameters['n_channel'],
    'k': 1,
    'num_epochs': 5000,
    'learning_rate': 1e-3,
    'batch_size': None,
    'device': 'auto',
    'seed': model_parameters['seed']
}

covert_parameters['m'] = 2 ** covert_parameters['k']
covert_parameters['r'] = covert_parameters['k'] / covert_parameters['n_channel']