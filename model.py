import torch
from torch import nn


class ClassifyWifiBill(nn.Module):
    def __init__(self,
                 input_shape: int,
                 output_classes: int,
                 hidden_units: int = 10,
                 height: int = 64,
                 weight: int = 64) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      padding=0,
                      stride=1,
                      kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      padding=0,
                      stride=1,
                      kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      padding=0,
                      stride=1,
                      kernel_size=3,
                      ),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      padding=0,
                      stride=1,
                      kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.conv_output_size = self.get_conv_output_size(input_shape=input_shape,
                                                          height=height,
                                                          weight=weight)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.conv_output_size,
                      out_features=output_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))

    def get_conv_output_size(self,
                             input_shape: int,
                             height: int = 64,
                             weight: int = 64) -> int:

        with torch.inference_mode():
            x = torch.rand(1, input_shape, height, weight)
            x = self.conv_block_2(self.conv_block_1(x))
            return x.view(x.size(0), -1).size(1)
