import numpy as np

from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        input_shape,
        output_shape,
        hidden_dim=256,
        num_classes=4,
        num_layers=1,
        expansion=4,
        use_layernorm=True,
        dropout_p=0.4,
    ):
        super(MLP, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.expansion = expansion

        if use_layernorm:
            norm_layer = nn.LayerNorm
        else:
            norm_layer = nn.BatchNorm1d

        input_size = int(np.prod(input_shape))
        output_size = int(np.prod(output_shape))

        last_size = input_size
        next_size = self.hidden_dim

        # layers = [nn.LayerNorm(last_size)]
        layers = []

        for layer in range(self.num_layers):
            if layer > 0:
                layers.append(nn.ReLU())
                layers.append(norm_layer(last_size))
                if dropout_p:
                    layers.append(nn.Dropout(p=dropout_p))
            if layer == num_layers - 1:
                next_size = output_size * self.num_classes
            layers.append(nn.Linear(last_size, next_size))
            last_size = next_size
            next_size *= self.expansion
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        x = x.view(x.size(0), self.num_classes, *self.output_shape)
        return x
