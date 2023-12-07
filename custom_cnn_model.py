import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class CustomCNN(nn.Module):
    def __init__(self, num_classes, filter_arch):
        super(CustomCNN, self).__init__()
        conv_layers = []
        in_channels = 3
        for i in filter_arch:
            out_channels = i
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_layers)

        # Calculate the input size for the fully connected layers
        input_size = int(out_channels * ((224 / (2 ** len(filter_arch))) ** 2))

        self.fc_layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Apply dropout with the specified probability
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

    def get_interested_layers(self, x, layer_indices):
        # Create a list to store intermediate feature maps
        intermediate_activations = []

        # Hook function to store intermediate feature maps
        def hook(module, input, output):
            intermediate_activations.append(output)

        # Register the hook for the specified layers
        hooks = []
        for layer_idx in layer_indices:
            layer = self.conv_layers[layer_idx * 4]  # Assuming every layer has Conv2d, BatchNorm2d, ReLU, MaxPool2d
            hook_handle = layer.register_forward_hook(hook)
            hooks.append(hook_handle)

        # Forward pass to compute the feature maps
        self.forward(x)

        # Remove the hooks
        for hook_handle in hooks:
            hook_handle.remove()

        # Return the intermediate feature maps
        return intermediate_activations