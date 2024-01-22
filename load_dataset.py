import logging
from pathlib import Path


class CustomData:
    def __init__(self, data_path: str):
        """
        Initialize the CustomData object.

        Args:
            data_path (str): Path to the root directory where the dataset will be stored.
        """
        self.data_path = Path(data_path)
        self.image_path = None

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path '{self.data_path}' does not exist.")

    @staticmethod
    def remove_directory(path: Path):
        """
        Remove a directory if it exists.

        Args:
            path (Path): Path to the directory to be removed.
        """
        try:
            path.rmdir()
            logging.info(f"Directory '{path}' removed successfully.")
        except OSError as e:
            logging.debug(f"Error removing directory '{path}': {e}")

    @staticmethod
    def local_drive(path: Path) -> bool:
        """
        Check if the dataset is present on the local drive.

        Args:
            path (Path): Path to the root directory of the dataset.

        Returns:
            bool: True if the dataset is present, False otherwise.
        """
        expected_folders = ['train', 'test']
        if not all((path / folder).exists() for folder in expected_folders):
            return False

        for folder in expected_folders:
            sub_folders_path = path / folder
            sub_folders = [sub_folder for sub_folder in sub_folders_path.iterdir() if sub_folder.is_dir()]

            if not all(sub_folder.is_dir() for sub_folder in sub_folders):
                return False

        return True


def main(dir_name: str | Path = "wifi_bills") -> Path:
    """
    Main function for handling dataset download and local checks.

    Args:
        dir_name (str | Path, optional): Name of the subdirectory to create for the dataset.
            Defaults to "pizza_steak_sushi".

    Returns:
        Path: Path to the dataset directory.

    Raises:
        FileNotFoundError: If the dataset is not present locally.
    """
    logging.basicConfig(level=logging.INFO)

    custom_dataset = CustomData(data_path=r"C:\Users\CD138JR\PycharmProjects\DeepLearning\CNN\data")

    path = custom_dataset.data_path / dir_name
    present = custom_dataset.local_drive(path=path)

    if present:
        logging.info("Dataset is present locally.")
        return path
    else:
        logging.warning("Dataset is not present locally.")
        raise FileNotFoundError
