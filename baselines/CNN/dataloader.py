"""
    Dataloader for MNIST/CIFAR data
    • Load the MNIST/CIFAR data
    • Remove a fixed sized patch from the image
    • Has graph connectivity as mentioned in:
        Processing of incomplete images by (graph) convolutional neural networks
        https://openreview.net/pdf?id=wxYPUnMSQCR
"""

# scipy
import numpy as np
# torch
import torch
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
# custom files
from utils import store_args
################### utils ###################
def train_val_split(dataset, train_val_split_ratio:float,  batch_size:int):
        """
            Split the data into train and validation sets
        """
        num_files = len(dataset)
        indices = list(range(num_files))
        split = int(np.floor(train_val_split_ratio * num_files))
        np.random.shuffle(indices)
        train_idx, val_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_dataloader = torch.utils.data.DataLoader(dataset, sampler=train_sampler, batch_size=batch_size)
        val_dataloader = torch.utils.data.DataLoader(dataset, sampler=val_sampler, batch_size=batch_size)
        return train_dataloader, val_dataloader

class MNISTDataloader():
    @store_args
    def __init__(self, batch_size:int, train_val_split_ratio:float, seed:int, patch_fill_type:str, testing:bool, download:bool=False):
        """
            Dataloader for training with MNIST dataset
            Parameters:
            –––––––––––
            batch_size: int
                Batch_size for sampling images
            train_val_split_ratio: float
                Ratio to split dataset in train and validation sets
                train_val_split_ratio in val and (1-train_val_split_ratio) in train
                NOTE: should be less than 1; preferably keep it (< 0.5)
            seed: int
                Random seed initialisation
            patch_fill_type:str
                Which method to use to substitute the pixel values
                Choices: 'mean', 'mask', 'knn'
            testing: bool
                If we also want to load the testloader
            download: bool
                If the dataset needs to be downloaded
        """
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # download MNIST and perform standard transforms
        # this is the training+validation set; have to split it into train-val set manually
        self.dataset = torchvision.datasets.MNIST('../../data/MNIST_data/', train=True, download=download, 
                                                transform=torchvision.transforms.Compose([ 
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(
                                                (0.1307,), (0.3081,)),
                                                ]))

        # split dataset into training and validation set and load into dataloaders
        self.train_dataloader,  self.val_dataloader = train_val_split(self.dataset, self.train_val_split_ratio, self.batch_size)

        if self.testing:
            testdata = torchvision.datasets.MNIST('../../data/MNIST_data/', train=False, download=download, 
                                                    transform=torchvision.transforms.Compose([ 
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(
                                                    (0.1307,), (0.3081,)),
                                                    ]))
            self.test_dataloader = torch.utils.data.DataLoader(testdata, batch_size=self.batch_size, shuffle=True)

    
class CIFARDataloader():
    @store_args
    def __init__(self, batch_size:int, train_val_split_ratio:float, seed:int, patch_fill_type:str, testing:bool, download:bool=False):
        """
            Dataloader for training with CIFAR dataset
            Parameters:
            –––––––––––
            batch_size: int
                Batch_size for sampling images
            train_val_split_ratio: float
                Ratio to split dataset in train and validation sets
                train_val_split_ratio in val and (1-train_val_split_ratio) in train
                NOTE: should be less than 1; preferably keep it (< 0.5)
            seed: int
                Random seed initialisation
            patch_fill_type:str
                Which method to use to substitute the pixel values
                Choices: 'mean', 'mask', 'knn'
            testing: bool
                If we also want to load the testloader
            download: bool
                If the dataset needs to be downloaded
        """
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # download CIFAR and perform standard normalisation transforms
        self.classes = ('plane', 'car', 'bird', 'cat', 'truck'
                        'deer', 'dog', 'frog', 'horse', 'ship')
        # this is the training+validation set; have to split it into train-val set manually
        self.dataset = torchvision.datasets.CIFAR10('../../data/CIFAR_data/', train=True, download=download, 
                                                transform=torchvision.transforms.Compose([ 
                                                torchvision.transforms.ToTensor(), 
                                                torchvision.transforms.Normalize(
                                                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                ]))

        # split dataset into training and validation set and load into dataloaders
        self.train_dataloader, self.val_dataloader = train_val_split(self.dataset, self.train_val_split_ratio, self.batch_size)

        if self.testing:
            testdata = torchvision.datasets.CIFAR10('../../data/CIFAR_data/', train=False, download=download, 
                                                    transform=torchvision.transforms.Compose([ 
                                                    torchvision.transforms.ToTensor(), 
                                                    torchvision.transforms.Normalize(
                                                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                    ]))
            self.test_dataloader = torch.utils.data.DataLoader(testdata, batch_size=self.batch_size, shuffle=True)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    print('_'*50); print('MNIST'); print('_'*50)
    for method in ['mean', 'mask']:
        print('_'*50); print(f'Showing images with {method} imputation'); print('_'*50)
        loader = MNISTDataloader(32, 0.5, 0, method, False)
        trainloader = loader.train_dataloader
        for i, (data, target) in enumerate(trainloader):
            x = data[0,0].unsqueeze(0)
            x = x.permute(1, 2, 0).numpy()
            plt.imshow(x)
            plt.show()
            if i==2:
                break
    print('_'*50); print('CIFAR'); print('_'*50)
    for method in ['mean', 'mask']:
        print('_'*50); print(f'Showing images with {method} imputation'); print('_'*50)
        loader = CIFARDataloader(32, 0.5, 0, method, False)
        trainloader = loader.train_dataloader
        for i, (data, target) in enumerate(trainloader):
            x = data[0,:3]
            x = x.permute(1, 2, 0).numpy()
            x = ((x * 0.5) + 0.5)
            plt.imshow(x)
            plt.show()
            if i==2:
                break