import torch
from torch import nn
import logging
from transforms import TransformData
from load_dataset import main
from train import Train
from model import ClassifyWifiBill
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
from plot import plot_loss_curves
from handler import ModelHandler


class Main(TransformData, Train, ModelHandler):
    def __init__(self, directory, image_size):
        super(Train, self).__init__()
        super(TransformData, self).__init__()
        super(ModelHandler, self).__init__()
        self.BATCH_SIZE = None
        self.image_path: str | Path = ''
        self.directory = directory
        self.image_size: Tuple[int, int] = image_size

    def check_dataset(self, directory):
        """
        check the dataset .

        Returns:
            None
        """
        try:

            self.image_path = main(dir_name=directory)
            logging.info("Checking the dataset")

        except ValueError:
            logging.error("Invalid input. Please enter a numeric value (1 or 2).")

    def initialize_model(
            self,
            input_shape: int,
            output_shape: int,
            hidden_units: int = 10,
    ) -> nn.Module:
        """
        Initialize the neural network model.

        Args:
            input_shape (int): Number of input channels.
            output_shape (int): Number of output classes.
            hidden_units (int): Number of hidden units in the model.

        Returns:
            nn.Module: Initialized neural network model.
        """
        print(input_shape, output_shape, hidden_units)
        return ClassifyWifiBill(input_shape=input_shape,
                                hidden_units=hidden_units,
                                output_classes=output_shape,
                                height=self.image_size[0],
                                weight=self.image_size[1])

    def process(self,
                input_shape: Optional[int] = 3,
                hidden_units: Optional[int] = 10,
                epochs: int = 5,
                ):
        """
        Process the chosen action (train or predict) based on user input.

        Args:
            input_shape (Optional[int]): Number of input channels (default is 3 for RGB).
            hidden_units (Optional[int]): Number of hidden units in the model (default is 10).
            epochs (int): Number of training epochs (default is 5).

        Returns:
            None
        """

        self.BATCH_SIZE = 32
        self.check_dataset(directory=self.directory)
        train_data, test_data = self.load_data(image_path=self.image_path,
                                               image_size=self.image_size)
        train_data_loader, test_data_loader = self.create_dataloaders(train_data=train_data,
                                                                      test_data=test_data,
                                                                      batch_size=self.BATCH_SIZE)

        self.classes = train_data.classes
        # Initialize the model based on user-defined parameters
        self.model = self.initialize_model(input_shape=input_shape,
                                           hidden_units=hidden_units,
                                           output_shape=len(train_data.classes)
                                           ).to(self.device)
        # Train the model and store the results
        self.results = self.train(model=self.model,
                                  train_dataloader=train_data_loader,
                                  test_dataloader=test_data_loader,
                                  epochs=epochs,
                                  loss_fn=nn.CrossEntropyLoss(),
                                  optimizer=torch.optim.Adam(params=self.model.parameters(),
                                                             lr=0.001))

        print(pd.DataFrame(self.results)),
        plot_loss_curves(self.results)
        return self.results


if __name__ == "__main__":
    logging.info("Train Your Own Model or Predict the image")

    folder = r"wifi_bills"
    try:
        main_instance = Main(directory=folder,
                             image_size=(224, 224))
        main_instance.process(input_shape=3,
                              hidden_units=20,
                              epochs=10)

    except ValueError:
        logging.error("Invalid input. Please enter a numeric value (1 or 2).")
