import torch
import torch.nn as nn
import torchcde

from models.Transformer_Layer import *


class CDEFunction(nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_function, hidden_function_layers, if_tanh=False):
        super(CDEFunction, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_function = hidden_function
        self.hidden_function_layers = hidden_function_layers
        self.if_tanh = if_tanh

        self.linear0 = torch.nn.Linear(self.hidden_channels, self.hidden_function)

        # Define middle layers
        layers = []
        for _ in range(self.hidden_function_layers):
            layers.append(torch.nn.Linear(self.hidden_function, self.hidden_function))
            layers.append(torch.nn.ReLU())
        self.middle_linears = nn.Sequential(*layers)

        self.linear4 = torch.nn.Linear(self.hidden_function, self.input_channels * self.hidden_channels)

    def forward(self, t, z):
        z = self.linear0(z)
        z = z.relu()
        z = self.middle_linears(z)
        z = self.linear4(z)

        if self.if_tanh:
            z = z.tanh()

        z = z.view(z.size(0), self.hidden_channels, self.input_channels)

        return z


class PIR_NCDE(nn.Module):
    def __init__(self, input_channels, window_size, hidden_function, output_channels, hidden_function_layers,
                 time_head_num, spatial_head_num, if_tanh, if_ts):
        super(PIR_NCDE, self).__init__()

        # Define hyper parameters
        self.input_channels = input_channels
        self.window_size = window_size
        self.hidden_function = hidden_function
        self.output_channels = output_channels
        self.hidden_function_layers = hidden_function_layers
        self.time_head_num = time_head_num
        self.spatial_head_num = spatial_head_num
        self.if_tanh = if_tanh
        self.if_ts = if_ts

        # Define function
        self.initialization = nn.Linear(in_features=self.input_channels, out_features=self.output_channels)
        self.function = CDEFunction(input_channels=self.input_channels, hidden_channels=self.output_channels,
                                    hidden_function=self.hidden_function,
                                    hidden_function_layers=self.hidden_function_layers, if_tanh=self.if_tanh)

        # Define modules to extract time features and spatial features
        self.time_extractor = get_torch_trans(heads=self.time_head_num, layers=1, channels=self.window_size)
        self.spatial_extractor = get_torch_trans(heads=self.spatial_head_num, layers=1, channels=self.output_channels)

        # Classification head
        self.classifier = nn.Linear(self.output_channels * 2, 3)

    def forward(self, x, coeffs):
        # Get features from defined NCDE
        X = torchcde.CubicSpline(coeffs)
        times = torch.arange(X.interval[-1].item() + 1).to(coeffs.device)
        X0 = X.evaluate(times)
        z0 = self.initialization(X0)
        z0 = z0.sum(dim=1)
        times = torch.arange(X.interval[-1].item() + 1).to(z0.device)
        zT = torchcde.cdeint(X=X, z0=z0, func=self.function, t=times)  # (batch_size, window_size, output_channels)

        # Time-spatial correlation
        # features_time: (batch_size, output_channels, window_size)
        features_time = self.time_extractor(zT.permute(0, 2, 1))
        if self.if_ts:
            # features_time_spatial: (batch_size, window_size, output_channels)
            features_time_spatial = self.spatial_extractor(features_time.permute(0, 2, 1))
        else:
            # features_time_spatial: (batch_size, window_size, output_channels)
            features_time_spatial = features_time.permute(0, 2, 1)

        # Spatial-time correlation
        # features_spatial: (batch_size, window_size, output_channels)
        features_spatial = self.spatial_extractor(zT)
        if self.if_ts:
            # features_spatial_time: (batch_size, output_channels, window_size)
            features_spatial_time = self.time_extractor(features_spatial.permute(0, 2, 1))
            # features_spatial_time: (batch_size, window_size, output_channels)
            features_spatial_time = features_spatial_time.permute(0, 2, 1)
        else:
            features_spatial_time = features_spatial

        # Concatenate features
        features_in_all = torch.concat([features_time_spatial, features_spatial_time],
                                       dim=-1)  # (batch_size, window_size, output_channels * 2)

        pooled_features = features_in_all.mean(dim=1)  # (batch_size, output_channels * 2)
        logits = self.classifier(pooled_features)  # (batch_size, 3)

        return logits