import os

channel_parameters = {
    'channel_type': 'rayleigh',
    'channel_k': 1,  # Used only when channel type is Rician
    'channel_samples': None,
    'channel_taps': None,
}
system_parameters = {
    'system_type': ''
}

if "siso" in os.getcwd():
    system_parameters['system_type'] = 'siso'
else:
    system_parameters['system_type'] = 'mimo'

if 'wireless_autoencoder' in os.getcwd():
    '''
        Parameters for training autoencoder model
    '''
    if channel_parameters['channel_type'] == 'awgn':
        channel_parameters['ebno'] = 8 if system_parameters['system_type'] == 'mimo' else 4
    if channel_parameters['channel_type'] == 'rayleigh':
        channel_parameters['ebno'] = 16
    if channel_parameters['channel_type'] == 'rician':
        channel_parameters['ebno'] = 16

if 'wireless_covert' in os.getcwd():
    '''
        Parameters for evaluating covert models on a fixed SNR during training
    '''
    if channel_parameters['channel_type'] == 'awgn':
        channel_parameters['ebno'] = 6 if system_parameters['system_type'] == 'mimo' else 2
    if channel_parameters['channel_type'] == 'rayleigh':
        channel_parameters['ebno'] = 15
    if channel_parameters['channel_type'] == 'rician':
        channel_parameters['ebno'] = 15
