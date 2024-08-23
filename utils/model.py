'''Models.'''

from numbers import Number

import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from .layers import DenseBlock, GCNBlock


class GCNModel(nn.Module):
    '''
    GCN-based model for node-level or graph-level prediction.

    Parameters
    ----------
    num_channels : int or list
        Channel numbers for GCN layers.
    num_features : int or list
        Feature numbers for linear layers.
    graph_level : bool
        Determines whether the model predicts
        node-level or graph-level properties.

    Notes
    -----
    The main model input are (num_nodes, num_features)-shaped tensors.
    For a node-level prediction model, the output shape is (num_nodes, num_targets).
    The output of a graph-level model is usually (batch_size, num_targets)-shaped.

    In case of a graph-level model, the input is usually the node feature matrix
    of a larger graph that contains multiple isolated subgraphs (batch items).
    Averaging over nodes is employed in order to appropriately reduce the output.

    '''

    def __init__(self,
                 num_channels=None,
                 num_features=None,
                 graph_level=False):

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

        # set prediction level (node or graph)
        self.graph_level = graph_level

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

    def forward(self, x, edge_index, batch=None):

        # run GCN layers
        if self.gconv_layers is not None:
            x = self.gconv_layers(x, edge_index)

        # average over nodes (for graph-level predictions)
        if self.graph_level:
            if batch is not None:
                # A batch is a huge graph that consists of multiple isolated subgraphs.
                # In order to average over the nodes, one therefore needs to identify
                # all the nodes that belong to the same batched subgraph.
                # This is done by a "batch"-tensor that assigns each node to an item.
                x = global_mean_pool(x, batch=batch)
            else:
                raise TypeError('Nodes have to be assigned to batch items (indices missing)')
        elif batch is not None:
            raise TypeError('Nodes should not be assigned to batch items (indices passed)')

        # run dense layers
        if self.dense_layers is not None:
            x = self.dense_layers(x)

        return x

