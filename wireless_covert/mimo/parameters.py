from shared_parameters import channel_parameters
from wireless_autoencoder.mimo.parameters import model_parameters

covert_parameters = {
    'n_channel': 8,
    'k': 1,
    'num_epochs': 5000,
    'learning_rate': 1e-3,
    'batch_size': None,
    'device': 'auto'
}

'''
    Model Parameters for Rayleigh
'''
covert_parameters['seed'] = 0
if channel_parameters['channel_type'] == 'rayleigh':

    if model_parameters['n_user'] == 2:
        covert_parameters['seed'] = 3
    if model_parameters['n_user'] == 4:
        covert_parameters['seed'] = 2

''' 
    Model parameters for AWGN
'''
if channel_parameters['channel_type'] == 'awgn':
    if model_parameters['n_user'] == 2:
        covert_parameters['seed'] = 1
    if model_parameters['n_user'] == 4:
        covert_parameters['seed'] = 1


covert_parameters['m'] = 2 ** covert_parameters['k']
covert_parameters['r'] = covert_parameters['k'] / covert_parameters['n_channel']