from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

def get_dataloader(batch_size=64):
    """
    Returns a DataLoader for the MNIST dataset.

    Parameters:
    - batch_size (int): The size of each batch of the training dataset. Default is 64.

    Returns:
    - DataLoader: A DataLoader instance for the MNIST dataset with specified batch size.

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
    trainset = MNIST('data/MNIST_data', download=True, train=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    return trainloader
