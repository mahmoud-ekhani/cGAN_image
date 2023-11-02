import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split

def get_dataloader(batch_size=64, train_size=0.8, seed=None):
    """
    Returns DataLoaders for the MNIST training and validation datasets.

    Parameters:
    - batch_size (int): The size of each batch of the training dataset. Default is 64.
    - train_size (float): The proportion of the dataset to use for training. Default is 0.8.
    - seed (int): The seed for random splitting of the dataset. Default is None.

    Returns:
    - (DataLoader, DataLoader): DataLoader instances for the MNIST training and validation datasets with specified batch size.

    Note:
    The function applies three transformations to the dataset:
    1. Upsamples the images to 64x64.
    2. Converts images to Tensors.
    3. Normalizes the images to have values in the range [-1, 1].

    The dataset is downloaded to 'data/MNIST_data/' if not already present.
    """
    transform = transforms.Compose([transforms.Resize((64, 64)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    
    dataset = MNIST('data/MNIST_data', download=True, train=True, transform=transform)

    # Splitting the dataset into training and validation sets
    n_train = int(len(dataset) * train_size)
    n_val = len(dataset) - n_train
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return trainloader, valloader
