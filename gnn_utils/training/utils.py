'''Training utils.'''

import torch


def _device(device=None):
    '''Determine device.'''

    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = device

    return device

