import torch
import torch.nn as nn

import numpy as np


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()

        # Convolution layer
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Compute output shape of layer to pass into fully connected layer
        conv_out_size = self._get_conv_out(input_shape)
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def _get_conv_out(self, shape):
        out = self.conv_layers(torch.zeros(1, *shape))
        return int(np.prod(out.size()))

    def forward(self, inputs):
        conv_out = self.conv_layers(inputs).view(inputs.size()[0], -1)
        return self.fc_layers(conv_out)
    
    def __call__(self, inputs):
        return self.forward(inputs)
