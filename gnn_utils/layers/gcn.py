'''GCN layers.'''

from collections.abc import Sequence

import torch
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
        Determines whether the output of the
        last layer gets nonlinearly activated.

    '''

    def __init__(
        self,
        num_channels: Sequence[int],
        activate_last: bool = True
    ) -> None:

        super().__init__()

        # check number of channels
        if len(num_channels) < 2:
            raise ValueError('Number of channels needs at least two entries')

        # assemble GCN layers
        num_layers = len(num_channels) - 1

        gconv_layers = [] # type: list[nn.Module]
        activ_layers = [] # type: list[nn.Module]

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

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:

        for gconv, activ in zip(self.gconv_layers, self.activ_layers):

            # apply graph convolution
            x = gconv(x, edge_index)

            # apply activation function
            if activ is not None:
                x = activ(x)

        return x

