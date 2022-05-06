from channel_parameters import channel_parameters

'''
    Model Parameters for Rayleigh
'''
if channel_parameters['channel_type'] == 'rayleigh':
    covert_parameters = {
        'n_channel': 8,
        'k': 1,
        'num_epochs': 4000,
        'learning_rate': 2e-4,
        'batch_size': 32,
        'low_rate_multiplier': None,
        'seed': 13,
        'device': 'auto'
    }

''' 
    Model parameters for AWGN
'''
if channel_parameters['channel_type'] == 'awgn':
    covert_parameters = {
        'n_channel': 8,
        'k': 1,
        'num_epochs': 4000,
        'learning_rate': 1e-4,
        'batch_size': 32,
        'low_rate_multiplier': None,
        'seed': 13,
        'device': 'auto'
    }

covert_parameters['m'] = 2 ** covert_parameters['k']
covert_parameters['r'] = covert_parameters['k'] / covert_parameters['n_channel']

covert_parameters['ebno'] = channel_parameters['ebno']
covert_parameters['channel_type'] = channel_parameters['channel_type']
covert_parameters['channel_samples'] = channel_parameters['channel_samples']
covert_parameters['channel_taps'] = channel_parameters['channel_taps']

if covert_parameters['low_rate_multiplier'] is not None:
    covert_parameters['n_channel'] *= covert_parameters['low_rate_multiplier']