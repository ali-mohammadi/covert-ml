channel_parameters = {
    'channel_type': 'rayleigh',
    'channel_samples': None,
    'channel_taps': None,
}

'''
    For training Autoencoder model
'''
# if channel_parameters['channel_type'] == 'awgn':
#     channel_parameters['ebno'] = 4
# if channel_parameters['channel_type'] == 'rayleigh':
#     channel_parameters['ebno'] = 16

'''
    For Evaluating Covert models on a fixed SNR during training
'''
if channel_parameters['channel_type'] == 'awgn':
    channel_parameters['ebno'] = 2
if channel_parameters['channel_type'] == 'rayleigh':
    channel_parameters['ebno'] = 15