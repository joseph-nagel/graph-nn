'''Dense layers.'''

from collections.abc import Sequence

import torch.nn as nn


class DenseBlock(nn.Sequential):
    '''
    Dense model block.

    Parameters
    ----------
    num_features : list
        Feature numbers of the linear layers.
    activate_last : bool
        Determines whether the output of the
        last layer gets nonlinearly activated.

    '''

    def __init__(
        self,
        num_features: Sequence[int],
        activate_last: bool = True
    ) -> None:

        # check number of features
        if len(num_features) < 2:
            raise ValueError('Number of features needs at least two entries')

        # assemble dense layers
        num_layers = len(num_features) - 1

        layers = [] # type: list[nn.Module]

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

