from shared_parameters import channel_parameters

covert_parameters = {
    'n_channel': 8,
    'k': 1,
    'num_epochs': 5000,
    'learning_rate': 1e-3,
    'batch_size': None,
    'low_rate_multiplier': None,
    'device': 'auto'
}

'''
    Model Parameters for Rayleigh
'''
if channel_parameters['channel_type'] == 'rayleigh':

    if covert_parameters['k'] == 1:
        covert_parameters['seed'] = 0
    if covert_parameters['k'] == 2:
        covert_parameters['seed'] = 1
    if covert_parameters['k'] == 4:
        covert_parameters['seed'] = 2

'''
    Model Parameters for Rician
'''
if channel_parameters['channel_type'] == 'rician':

    if covert_parameters['k'] == 1:
        covert_parameters['seed'] = 0
    if covert_parameters['k'] == 2:
        covert_parameters['seed'] = 0
    if covert_parameters['k'] == 4:
        covert_parameters['seed'] = 1


''' 
    Model parameters for AWGN
'''
if channel_parameters['channel_type'] == 'awgn':
    if covert_parameters['k'] == 1:
        covert_parameters['seed'] = 0
    if covert_parameters['k'] == 2:
        covert_parameters['seed'] = 0
    if covert_parameters['k'] == 4:
        covert_parameters['seed'] = 0

covert_parameters['m'] = 2 ** covert_parameters['k']
covert_parameters['r'] = covert_parameters['k'] / covert_parameters['n_channel']

if covert_parameters['low_rate_multiplier'] is not None:
    covert_parameters['n_channel'] *= covert_parameters['low_rate_multiplier']