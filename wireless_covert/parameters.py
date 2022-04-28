'''
    Model Parameters for Rayleigh
'''
# covert_parameters = {
#     'n_channel': 8,
#     'k': 1,
#     'ebno': 8,
#     'num_epochs': 4000,
#     'learning_rate': 2e-4,
#     'batch_size': 32,
#     'channel_type': 'rayleigh',
#     'channel_samples': None,
#     'channel_taps': None,
#     'low_rate_multiplier': None,
#     'seed': 13,
#     'device': 'auto'
# }

''' 
    Model parameters for AWGN
'''
covert_parameters = {
    'n_channel': 8,
    'k': 1,
    'ebno': 4,
    'num_epochs': 4000,
    'learning_rate': 2e-4,
    'batch_size': 32,
    'channel_type': 'awgn',
    'channel_samples': None,
    'channel_taps': None,
    'low_rate_multiplier': None,
    'seed': 0,
    'device': 'auto'
}

covert_parameters['m'] = 2 ** covert_parameters['k']
covert_parameters['r'] = covert_parameters['k'] / covert_parameters['n_channel']