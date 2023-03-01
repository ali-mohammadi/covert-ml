from shared_parameters import channel_parameters
'''
    Model Parameters for Rayleigh 
'''
if channel_parameters['channel_type'] == 'rayleigh':
    model_parameters = {
        'n_user': 2,
        'n_channel': 8,
        'k': 4,
        'train_size': 1024 * 8,
        'test_size': 1024 * 50,
        'batch_size': 1024,
        'num_epochs': 100,
        'learning_rate': 1e-3,
        'seed': None,
        'device': 'auto'
    }
    if model_parameters['n_user'] == 2:
        model_parameters['seed'] = 2
    if model_parameters['n_user'] == 4:
        model_parameters['seed'] = 3

''' 
    Model parameters for AWGN
'''
if channel_parameters['channel_type'] == 'awgn':
    model_parameters = {
        'n_user': 2,
        'n_channel': 8,
        'k': 4,
        'train_size': 1024 * 8,
        'test_size': 1024 * 50,
        'batch_size': 1024,
        'num_epochs': 100,
        'learning_rate': 1e-3,
        'seed': None,
        'device': 'auto'
    }
    if model_parameters['n_user'] == 2:
        model_parameters['seed'] = 1
    if model_parameters['n_user'] == 4:
        model_parameters['seed'] = 1

model_parameters['m'] = 2 ** model_parameters['k']
model_parameters['r'] = model_parameters['k'] / model_parameters['n_channel']