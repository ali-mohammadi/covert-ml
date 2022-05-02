import torch

''' 
    Model Parameters for Rayleigh 
'''
model_parameters = {
    'n_channel': 8,
    'k': 4,
    'ebno': 14,
    'train_size': 1024 * 8,
    'test_size': 1024 * 6,
    'batch_size': 16,
    'num_epochs': 100,
    'learning_rate': 1e-3,
    'channel_type': 'rayleigh',
    'channel_samples': None,
    'channel_taps': None,
    'seed': 13,
    'device': 'auto'
}

''' 
    Model parameters for AWGN
'''
# model_parameters = {
#     'n_channel': 8,
#     'k': 4,
#     'ebno': 7,
#     'train_size': 1024 * 8,
#     'test_size': 1024 * 50,
#     'batch_size': 32,
#     'num_epochs': 100,
#     'learning_rate': 1e-3,
#     'channel_type': 'awgn',
#     'channel_samples': None,
#     'channel_taps': None,
#     'seed': 0
# }


'''
    Model parameters for multiple channel realization on each epoch for Rayleigh channel
'''
# model_parameters = {
#     'n_channel': 8,
#     'k': 4,
#     'ebno': 20,
#     'train_size': 1024 * 8,
#     'test_size': 1024 * 6,
#     'batch_size': 32,
#     'num_epochs': 100,
#     'learning_rate': 1e-3,
#     'channel_type': 'rayleigh',
#     'channel_samples': 10,
#     'channel_taps': None,
#     'seed': 0
# }

model_parameters['m'] = 2 ** model_parameters['k']
model_parameters['r'] = model_parameters['k'] / model_parameters['n_channel']

