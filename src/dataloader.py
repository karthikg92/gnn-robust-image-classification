"""
    Dataloader for MNIST data
    * Load the MNIST data
    * Sample random pixels
    * Create a graph out of those pixels
        * Make adjacency matrix
        * Make node feature matrix
    * Pass it out as a batch
"""
import numpy as np
import torch
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler

from utils import store_args


class MNISTDataloader():
    @store_args
    def __init__(self, num_nodes:int, batch_size:int, train_val_split_ratio:float, seed:int, testing:bool):
        """
            Dataloader for training with MNIST dataset
            Parameters:
            -----------
            num_nodes: int
                Number of pixels to sample from the image
            batch_size: int
                Batch_size for sampling images
            train_val_split_ratio: float
                Ratio to split dataset in train and validation sets
                train_val_split_ratio in val and (1-train_val_split_ratio) in train
                NOTE: should be less than 1; preferably keep it (< 0.5)
            seed: int
                Random seed initialisation
            test: bool
                If we also want to load the testloader
        """
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # download MNIST and perform standard transforms
        # this is the training+validation set; have to split it into train-val set manually
        self.dataset = torchvision.datasets.MNIST('MNIST_data/', train=True, download=True, 
                                                transform=torchvision.transforms.Compose([ 
                                                torchvision.transforms.ToTensor(), 
                                                torchvision.transforms.Normalize( 
                                                (0.1307,), (0.3081,)) 
                                                ]))

        self.train_val_split()

        if self.testing:
            testdata = torchvision.datasets.MNIST('MNIST_data/', train=False, download=True, 
                                                    transform=torchvision.transforms.Compose([ 
                                                    torchvision.transforms.ToTensor(), 
                                                    torchvision.transforms.Normalize( 
                                                    (0.1307,), (0.3081,)) 
                                                    ]))
            self.test_dataloader = torch.utils.data.Dataloader(testdata, batch_size=self.batch_size, shuffle=True)

    def train_val_split(self):
        """
            Split the data into train and validation sets
        """
        num_files = len(self.dataset)
        indices = list(range(num_files))
        split = int(np.floor(self.train_val_split_ratio * num_files))
        np.random.shuffle(indices)
        train_idx, val_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        self.train_dataloader = torch.utils.data.DataLoader(self.dataset, sampler=train_sampler, batch_size=self.batch_size)
        self.val_dataloader = torch.utils.data.DataLoader(self.dataset, sampler=val_sampler, batch_size=self.batch_size)


    def process(self, data:torch.tensor, num_samples:int, euclid:bool):
        """
            Sample random points from the images and 
            give adjacency matrix and the feature matrix
            Parameters:
            -----------
            data: torch.tensor
                The batch of data to sample points from
                Should be of shape (batch_size, C, H, W)
            num_samples: int
                Number of pixels to sample
            euclid: bool
                True if we want a single adj matrix 
                containing the euclid dist between pixels
                False if we want two adj matrices, del_x and del_y
            Returns:
            --------
            X_feat: torch.tensor
                The pixel values at the nodes sampled randomly
                Shape: (batch_size, C, num_samples)
            A_x: torch.tensor
                The x_coord differences between the sampled points
            A_y: torch.tensor
                The y_coord differences between the sampled points
            Usage:
            ------
            for batch_idx, (data, target) in enumerate(self.train_dataloader):
                X_feat, (A_x, A_y) = self.process(data, num_samples=10)
                ...
        """
        batch_size, channels, H, W = data.shape
        x = torch.randint(0, H, (num_samples,))
        y = torch.randint(0, W, (num_samples,))
        # NOTE: sample from same random points in a given batch
        X_feat = data[:,:,x,y]      # shape [batch_size, C, num_samples]
        X_feat= X_feat.transpose(1,2) # shape [batch_size, num_samples,C,]
        A = self.get_dist_adjacency(x, y, H, W, euclid=euclid)
        # A = A.expand(batch_size, num_samples, num_samples)
        return X_feat, A
        # NOTE confirm with @karthikg92 if we should be using multiple adj matrices
        # A_x, A_y = self.get_dist_adjacency(x, y, num_samples, H, W)
        # return X_feat, (A_x, A_y)

    def get_dist_adjacency(self, x:torch.tensor, y:torch.tensor, H:int, W:int, euclid:bool):
        """
            Given the indices, get the adjacency matrices of delta_x and delta_y
            Parameters:
            -----------
            x: torch.tensor
                The x-coords of the sampled points
                Shape: (num_samples)
            y: torch.tensor
                The y-coords of the sampled points
                Shape: (num_samples)
            H: int
                height of the image
            W: int
                width of the image
            euclid: bool
                True if we want a single adj matrix 
                containing the euclid dist between pixels
                False if we want two adj matrices, del_x and del_y
        """
        x_dist = torch.abs(x - x.reshape(-1,1))    # this will evaluate |x[i]-x[j]| for all combinations of i,j
        y_dist = torch.abs(y - y.reshape(-1,1))    # this will evaluate |y[i]-y[j]| for all combinations of i,j
        # NOTE @karthikg92 do we have to threshold this?
        if euclid:
            A = torch.sqrt(torch.square(x_dist).float() + torch.square(y_dist).float())
            A /= np.sqrt(H**2 + W**2)               # normalise with diagonal length
            return A
        else: 
            x_dist /= H
            y_dist /= W
            return (x_dist, y_dist)

    def get_correlation_adjacency(self, X_feat, x, y):
        """
            Get the correlation between pixel values
            NOTE: ask @karthikg92 about this
        """
        pass

if __name__ == "__main__":
    loader = MNISTDataloader(10, 32, 0.5, 0, False)
    trainloader = loader.train_dataloader
    for i, (data, target) in enumerate(trainloader):
        print(f'Data shape: {data.shape}')
        print(f'Target shape: {target.shape}')
        X, A = loader.process(data, 10, True)
        print(f'Node feat shape: {X.shape}')
        print(f'Adj shape: {A.shape}')
        break
    



