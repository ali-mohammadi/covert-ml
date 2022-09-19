from shared_parameters import channel_parameters
'''
    Model Parameters for Rayleigh 
'''
if channel_parameters['channel_type'] == 'rayleigh':
    model_parameters = {
        'n_user': 2,
        'n_channel': 1,
        'k': 2,
        'train_size': 1024 * 8,
        'test_size': 1024 * 50,
        'batch_size': 2048,
        'num_epochs': 100,
        'learning_rate': 1e-3,
        'seed': 0,
        'device': 'auto'
    }

''' 
    Model parameters for AWGN
'''
if channel_parameters['channel_type'] == 'awgn':
    model_parameters = {
        'n_user': 4,
        'n_channel': 2,
        'k': 2,
        'train_size': 1024 * 8,
        'test_size': 1024 * 50,
        'batch_size': 2048,
        'num_epochs': 500,
        'learning_rate': 1e-3,
        'seed': 0,
        'device': 'auto'
    }


'''
    Model Parameters for Rician 
'''
if channel_parameters['channel_type'] == 'rician':
    model_parameters = {
        'n_user': 2,
        'n_channel': 8,
        'k': 4,
        'train_size': 1024 * 8,
        'test_size': 1024 * 50,
        'batch_size': 2048,
        'num_epochs': 100,
        'learning_rate': 1e-3,
        'seed': 0,
        'device': 'auto'
    }

model_parameters['m'] = 2 ** model_parameters['k']
model_parameters['r'] = model_parameters['k'] / model_parameters['n_channel']

