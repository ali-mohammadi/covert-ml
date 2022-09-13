system_parameters = {
    'system_type': 'mimo'
}

channel_parameters = {
    'channel_type': 'rayleigh',
    'channel_k': 8,  # Used only when channel type is set to Rician
    'channel_samples': None,
    'channel_taps': None,
}

'''
    Parameters for training autoencoder model
'''
# if channel_parameters['channel_type'] == 'awgn':
#     channel_parameters['ebno'] = 4
# if channel_parameters['channel_type'] == 'rayleigh':
#     channel_parameters['ebno'] = 16
# if channel_parameters['channel_type'] == 'rician':
#     channel_parameters['ebno'] = 30

'''
    Parameters for evaluating covert models on a fixed SNR during training
'''
if channel_parameters['channel_type'] == 'awgn':
    channel_parameters['ebno'] = 2
if channel_parameters['channel_type'] == 'rayleigh':
    channel_parameters['ebno'] = 15
if channel_parameters['channel_type'] == 'rician':
    channel_parameters['ebno'] = 8