"""
    Dataloader for MNIST/CIFAR data
    • Load the MNIST/CIFAR data
    • Remove a fixed sized patch from the image
    • Has baselines as mentioned in:
        Processing of incomplete images by (graph) convolutional neural networks
        https://openreview.net/pdf?id=wxYPUnMSQCR
    • Use this with CNN with three options:
        • Mean: absent attributes are replaced by mean values for a given coordinate
        • Mask: zero imputation with an additional binary channel indicating unknown pixels
        • k-nearest neighbours: substitute missing features with mean values of those 
                                features computed from the k-nearest training samples
"""
# scipy
import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
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

class removePatch(object):
    def __init__(self, dataset_name:str, patch_fill_type:str, mean_coords):
        """
            Remove a patch from the image of size 13x13(for MNIST) or 15x15(for CIFAR/SVHN)
            from a uniformly chosen position
            Parameters:
            –––––––––––
            dataset_name: str
                The size of patch to remove
            patch_fill_type: str
                Which method to use to substitute the pixel values
                Choices: 'mean', 'mask', 'knn'
            mean_coords: torch.tensor
                Shape [C,H,W] containing the mean value of pixels at each coordinate (c,h,w)
        """
        self.patch_fill_type = patch_fill_type 
        self.mean_coords = mean_coords
        if dataset_name == 'MNIST':
            self.patch_size = 6 # (13-1)/2
            self.img_size = 28
        elif dataset_name == 'CIFAR' or 'SVHN':
            self.patch_size = 7 # (15-1)/2
            self.img_size = 32

    def __call__(self, img):
        """
            Parameters:
            –––––––––––
            img: torch.tensor
            Returns:
            ––––––––
            Tensor: image with patch removed
        """
        return self.replace_patch(img, self.patch_size)

    def replace_patch(self, img:torch.tensor, ps:int):
        """
            Choose a random position to remove the patch and then
            replace those pixel values according to the one of the
            methods as defined above:
            • Mean
            • Mask
            • k-NN
            Tested 'mask' and 'mean' by visually inspecting the images 
        """
        # randomly choose the centre of the patch to be removed
        cx = torch.randint(ps, self.img_size-ps, (1,)).item()
        cy = torch.randint(ps, self.img_size-ps, (1,)).item()
        if self.patch_fill_type == 'mask':
            # replace with zeros
            img[:, cx-ps:cx+ps+1, cy-ps:cy+ps+1] = 0
            mask = torch.zeros_like(img)
            # fill mask with ones where pixels are missing
            mask[:, cx-ps:cx+ps+1, cy-ps:cy+ps+1] = 1
            return torch.cat((img,mask), 0)  # shape [2*C, H, W]
        if self.patch_fill_type == 'mean':
            # replace with the means at coords
            img[:, cx-ps:cx+ps+1, cy-ps:cy+ps+1] = self.mean_coords[:, cx-ps:cx+ps+1, cy-ps:cy+ps+1]
            return img   # shape [C, H, W]
        if self.patch_fill_type == 'knn':
            assert self.patch_fill_type!='knn', "KNN Method not implemented yet"
            return img

    def __repr__(self):
        return self.__class__.__name__ + f'({self.dataset_name}, {self.patch_fill_type}, {self.img_size}, {self.patch_size*2+1})'


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
        mean_coords = self.get_mean_coord()
        # this is the training+validation set; have to split it into train-val set manually
        self.dataset = torchvision.datasets.MNIST('../../data/MNIST_data/', train=True, download=download, 
                                                transform=torchvision.transforms.Compose([ 
                                                torchvision.transforms.ToTensor(),
                                                # removePatch('MNIST', patch_fill_type, mean_coords),
                                                torchvision.transforms.Normalize( 
                                                (0.1307,), (0.3081,)),
                                                removePatch('MNIST', patch_fill_type, mean_coords),
                                                ]))

        # split dataset into training and validation set and load into dataloaders
        self.train_dataloader,  self.val_dataloader = train_val_split(self.dataset, self.train_val_split_ratio, self.batch_size)

        if self.testing:
            testdata = torchvision.datasets.MNIST('../../data/MNIST_data/', train=False, download=download, 
                                                    transform=torchvision.transforms.Compose([ 
                                                    torchvision.transforms.ToTensor(),
                                                    # removePatch('MNIST', patch_fill_type, mean_coords),
                                                    torchvision.transforms.Normalize( 
                                                    (0.1307,), (0.3081,)),
                                                    removePatch('MNIST', patch_fill_type, mean_coords),
                                                    ]))
            self.test_dataloader = torch.utils.data.DataLoader(testdata, batch_size=self.batch_size, shuffle=True)
    
    def get_mean_coord(self):
        """
            Get the mean pixel values across 
            the dataset for all co-ordinates
            Will return a [C,H,W] shaped torch.tensor
        """
        # load dataset in a dummy manner
        dataset = torchvision.datasets.MNIST('../../data/MNIST_data/', train=True, download=False)
        mean = (dataset.data.float().mean(0)/255).unsqueeze(0) # [1,28,28]
        return mean

    
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
        mean_coords = self.get_mean_coord()
        # this is the training+validation set; have to split it into train-val set manually
        self.dataset = torchvision.datasets.CIFAR10('../../data/CIFAR_data/', train=True, download=download, 
                                                transform=torchvision.transforms.Compose([ 
                                                torchvision.transforms.ToTensor(), 
                                                # removePatch('CIFAR', patch_fill_type, mean_coords),
                                                torchvision.transforms.Normalize( 
                                                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                removePatch('CIFAR', patch_fill_type, mean_coords),
                                                ]))

        # split dataset into training and validation set and load into dataloaders
        self.train_dataloader, self.val_dataloader = train_val_split(self.dataset, self.train_val_split_ratio, self.batch_size)

        if self.testing:
            testdata = torchvision.datasets.CIFAR10('../../data/CIFAR_data/', train=False, download=download, 
                                                    transform=torchvision.transforms.Compose([ 
                                                    torchvision.transforms.ToTensor(), 
                                                    # removePatch('CIFAR', patch_fill_type, mean_coords),
                                                    torchvision.transforms.Normalize( 
                                                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                    removePatch('CIFAR', patch_fill_type, mean_coords),
                                                    ]))
            self.test_dataloader = torch.utils.data.DataLoader(testdata, batch_size=self.batch_size, shuffle=True)

    def get_mean_coord(self):
        """
            Get the mean pixel values across 
            the dataset for all co-ordinates
            Will return a [C,H,W] shaped torch.tensor
        """
        # load dataset in a dummy manner
        dataset = torchvision.datasets.CIFAR10('../../data/CIFAR_data/', train=True, download=False)
        data = torch.FloatTensor(dataset.data).permute(0,3,1,2) # shape [num_img, 3, 32, 32]
        mean = data.mean(0)/255 # [3,32,32]
        return mean

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