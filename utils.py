'''Some utilities.'''

import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCNClassifier(nn.Module):
    '''
    Graph convolutional network for node-level classification.

    Parameters
    ----------
    num_channels : list
        Channel numbers for GCN layers.
    num_classes : int
        Number of classes.

    '''

    def __init__(self,
                 num_channels,
                 num_classes):

        super().__init__()

        # check number of channels
        if len(num_channels) < 2:
            raise ValueError('Number of channels needs at least two entries')

        # assemble GCN layers
        gconv_layers = []
        for in_channels, out_channels in zip(num_channels[:-1], num_channels[1:]):
            gconv = GCNConv(in_channels, out_channels)
            gconv_layers.append(gconv)

        self.gconv_layers = nn.ModuleList(gconv_layers)

        # create linear classifier
        self.linear = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x, edge_index):

        # run GCN layers
        for gconv in self.gconv_layers:
            x = gconv(x, edge_index)
            x = nn.functional.relu(x)

        # run linear classifier
        x = self.linear(x)

        return x

