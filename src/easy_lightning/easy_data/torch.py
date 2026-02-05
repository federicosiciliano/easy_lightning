import numpy as np
import torchvision
import torch

def get_torchvision_data(name, root="../data/raw", download=True,
						transform=torchvision.transforms.Compose([
									torchvision.transforms.ToTensor(),
					    			#torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
									]),
						target_transform=None, **kwargs):
    """
    Load data from the torchvision library.

    :param name: Name of the dataset to load from torchvision (e.g., 'MNIST', 'CIFAR10').
    :param root: Root directory where the dataset will be stored (default is "../data/").
    :param download: Boolean, whether to download the dataset if it doesn't exist locally (default is True).
    :param transform: A composition of image transformations to apply to the dataset (default includes ToTensor()).
    :param target_transform: Optional transformation to apply to the target (labels).
    :param kwargs: Additional keyword arguments that can be passed to the torchvision dataset constructor.
    :return: A dictionary containing training and testing data as 'train_x', 'train_y', 'test_x', 'test_y'.
    """
    data = {}

    for split in ["train", "test"]:
        # Use getattr to dynamically fetch the dataset class based on the provided name (e.g., 'MNIST', 'CIFAR10').
        raw_data = getattr(torchvision.datasets, name)(
            root=root, train=split == "train", transform=transform, target_transform=target_transform, download=download
        )

        # Create keys for the data dictionary with split prefixes ('train' or 'test') and store data as tensors.
        data[split + "_x"] = torch.stack([app[0] for app in raw_data])
        data[split + "_y"] = torch.tensor([app[1] for app in raw_data])

    return data
