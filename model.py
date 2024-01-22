import torch
from torch import nn


class ClassifyWifiBill(nn.Module):

    def __init__(self,
                 input_shape: int,
                 output_classes: int,
                 hidden_units: int = 10) -> None:
        super().__init__()
        """
        Todo: Need to add Conv Blocks 
        
        """
        self.conv_output_size = self.get_conv_output_size(input_shape=input_shape)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.conv_output_size,
                      out_features=output_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x:
        :return:
        """
        # //Todo
        return torch.Tensor()

    def get_conv_output_size(self,
                             input_shape: int,
                             height: int = 64,
                             weight: int = 64) -> int:

        with torch.inference_mode():
            x = torch.rand(1, input_shape, height, weight)
            x = self.conv_block_2(self.conv_block_1(x))
            return x.view(x.size(0), -1).size(1)
