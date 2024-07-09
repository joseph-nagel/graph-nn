'''Models.'''

from numbers import Number

import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCNBlock(nn.Module):
    '''
    GCN model block.

    Parameters
    ----------
    num_channels : list
        Channel numbers of the GCN layers.
    activate_last : bool
        Determines whether an the output of the
        last layer gets nonlinearly activated.

    '''

    def __init__(self, num_channels, activate_last=True):

        super().__init__()

        # check number of channels
        if len(num_channels) < 2:
            raise ValueError('Number of channels needs at least two entries')

        # assemble GCN layers
        num_layers = len(num_channels) - 1

        gconv_layers = []
        activ_layers = []

        for idx, (in_channels, out_channels) in enumerate(zip(num_channels[:-1], num_channels[1:])):
            is_not_last = (idx < num_layers - 1)

            # create GCN layer
            gconv = GCNConv(in_channels, out_channels)
            gconv_layers.append(gconv)

            # create activation function
            if is_not_last or activate_last:
                activ = nn.LeakyReLU()
            else:
                activ = None # nn.ModuleList seems to accept None entries

            activ_layers.append(activ)

        self.gconv_layers = nn.ModuleList(gconv_layers)
        self.activ_layers = nn.ModuleList(activ_layers)

    def forward(self, x, edge_index):

        for gconv, activ in zip(self.gconv_layers, self.activ_layers):

            # apply graph convolution
            x = gconv(x, edge_index)

            # apply activation function
            if activ is not None:
                x = activ(x)

        return x


class DenseBlock(nn.Sequential):
    '''
    Dense model block.

    Parameters
    ----------
    num_features : list
        Feature numbers of the linear layers.
    activate_last : bool
        Determines whether an the output of the
        last layer gets nonlinearly activated.

    '''

    def __init__(self, num_features, activate_last=True):

        # check number of features
        if len(num_features) < 2:
            raise ValueError('Number of features needs at least two entries')

        # assemble dense layers
        num_layers = len(num_features) - 1

        layers = []
        for idx, (in_features, out_features) in enumerate(zip(num_features[:-1], num_features[1:])):
            is_not_last = (idx < num_layers - 1)

            # create linear layer
            linear = nn.Linear(in_features, out_features)
            layers.append(linear)

            # create activation function
            if is_not_last or activate_last:
                activ = nn.LeakyReLU()
                layers.append(activ)

        # initialize sequential model
        super().__init__(*layers)


class GCNModel(nn.Module):
    '''
    GCN-based model for node-level prediction.

    Parameters
    ----------
    num_channels : int or list
        Channel numbers for GCN layers.
    num_features : int or list
        Feature numbers for linear layers.

    '''

    def __init__(self,
                 num_channels=None,
                 num_features=None):

        super().__init__()

        # create GCN block
        if num_channels is None:
            self.gconv_layers = None

        elif isinstance(num_channels, Number):
            self.gconv_layers = None
            num_channels = [num_channels]

        elif isinstance(num_channels, (list, tuple)):
            if num_features is not None:
                activate_last = True
            else:
                activate_last = False

            self.gconv_layers = GCNBlock(
                num_channels=num_channels,
                activate_last=activate_last
            )

        else:
            raise TypeError(f'Invalid number of channels type: {type(num_channels)}')

        # create dense block
        if num_features is None:
            self.dense_layers = None

        elif isinstance(num_features, Number):
            if isinstance(num_channels, (list, tuple)):
                self.dense_layers = nn.Linear(num_channels[-1], num_features)
            else:
                raise TypeError(f'Incompatible number of channels type: {type(num_channels)}')

        elif isinstance(num_features, (list, tuple)):
            self.dense_layers = DenseBlock(
                num_features=[num_channels[-1]] + num_features,
                activate_last=False
            )

        else:
            raise TypeError(f'Invalid number of features type: {type(num_features)}')

    def forward(self, x, edge_index):

        # run GCN layers
        if self.gconv_layers is not None:
            x = self.gconv_layers(x, edge_index)

        # run dense layers
        if self.dense_layers is not None:
            x = self.dense_layers(x)

        return x

