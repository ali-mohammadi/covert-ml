channel_parameters = {
    'channel_type': 'rayleigh',
    'channel_samples': None,
    'channel_taps': None
}

if channel_parameters['channel_type'] == 'awgn':
    channel_parameters['ebno'] = 4
if channel_parameters['channel_type'] == 'rayleigh':
    channel_parameters['ebno'] = 13
