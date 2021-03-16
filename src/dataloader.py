"""
    Dataloader for MNIST data
    * Load the MNIST data
    * Sample random pixels
    * Create a graph out of those pixels
        * Make adjacency matrix
        * Make node feature matrix
    * Pass it out as a batch
"""
# scipy
import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
# torch
import torch
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
# torch_geometric
import torch_geometric
from torch_geometric.data import Data
# custom files
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
    
    def process_torch_geometric(self, data:torch.tensor, num_samples:int, k:int, polar:bool):
        """
            Sample random points from the images and 
            give adjacency matrix and the feature matrix

            Will convert this into pytorch geometric compatible input
            The data should have the following attributes:
                data.x: 
                    Node feature matrix with shape [num_nodes, num_node_features]
                data.edge_index: 
                    Graph connectivity in COO format with shape [2, num_edges] and type torch.long
                data.edge_attr: 
                    Edge feature matrix with shape [num_edges, num_edge_features]

            Parameters:
            -----------
            data: torch.tensor
                The batch of data to sample points from
                Should be of shape (batch_size, C, H, W)
            num_samples: int
                Number of pixels to sample
            k: int
                Number of neighbouring pixels to connect to
            polar: bool
                What features to return as edge_weight matrix
                If True then will return (r,θ)
                where r \in [0,1] and θ \in [-1,1];
                NOTE: r is normalised by (H^2 + W^2) and θ is normalised by π
                If False then will return (Δx,Δy)
                where Δx \in [-1,1] and Δy \in [-1,1]
                NOTE: Δx is normalised by H and Δy is normalised by W
            Returns:
            --------
            torch_geometric.data.DataLoader():
                with node feature matrix, edge_index, edge_attr loaded
            Usage:
            ------
            for batch_idx, (data, target) in enumerate(self.train_dataloader):
                geom_loader = self.process_torch_geometric(data, num_samples=10, k=5)
                # NOTE here that there is only one element in geom_loader with batch_size same as the train_dataloader
                for batch in geom_loader:
                    out = net(batch)
                    ...
        """
        batch_size, channels, H, W = data.shape
        x = torch.randint(0, H, (num_samples,))
        y = torch.randint(0, W, (num_samples,))
        # NOTE: sample from same random points in a given batch
        X_feat = data[:,:,x,y]          # shape [batch_size, node_feat, num_samples]
        X_feat= X_feat.transpose(1,2)   # shape [batch_size, num_samples, node_feat]
        A, edge_weights = self.get_adjacency(x, y, k, H, W, symmetric=True, polar=polar)
        sA = sparse.csr_matrix(A)   # convert to sparse format
        edge_index, edge_weight = self._toCOO(sA, edge_weights)
        data_list = []
        for i in range(batch_size):
            geom_data = Data(x=X_feat[i], edge_index=edge_index, edge_attr=edge_weight)
            data_list.append(geom_data)
        return torch_geometric.data.DataLoader(data_list, batch_size=batch_size)

    def _toCOO(self, A, edge_weight:torch.tensor):
        """
            Converts a scipy sparse matrix to edge indices and edge attributes 
            in the COO format compatible with torch_geometric
            Parameters:
            -----------
            A: scipy.sparse
                A sparse matrix representing the adjacency matrix
            edge_weight: torch.tensor
                This contains the edge weights, can be multi-dimensional edge-weights
                NOTE: This is not same as A as A just contains binary representing whether 
                an edge exists. Shape [num_nodes, num_nodes, edge_feat_dims]
        """
        A = A.tocoo()
        row = torch.from_numpy(A.row).to(torch.long)
        col = torch.from_numpy(A.col).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)
        edge_weight = edge_weight[row,col]  # shape [num_edges, num_edge_feat]
        return edge_index, edge_weight

    def get_adjacency(self, x:torch.tensor, y:torch.tensor, k:int, H:int, W:int, symmetric:bool, polar:bool):
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
            k: int
                Number of neighbouring points to connect to
            H: int
                height of the image
            W: int
                width of the image
            symmetric: bool
                The k-nearest neighbours matrix is not symmetric
                If 'symmetric' then it will be made symmetric
            polar: bool
                What features to return as edge_weight matrix
                If True then will return (r,θ)
                where r \in [0,1] and θ \in [-1,1];
                NOTE: r is normalised by (H^2 + W^2) and θ is normalised by π
                If False then will return (Δx,Δy)
                where Δx \in [-1,1] and Δy \in [-1,1]
                NOTE: Δx is normalised by H and Δy is normalised by W
        """
        point_coords = torch.stack((x,y),1)
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(point_coords)
        A = nbrs.kneighbors_graph(point_coords).toarray()   # this is not symmetric
        if symmetric:
            A = (A+A.T)
            A = (A>0).astype(int)   # make it a binary matrix
        A = torch.FloatTensor(A)
        # this will evaluate (x[i]-x[j]) and (y[i]-y[j]) for all combinations of i,j
        x_dist = (x - x.reshape(-1,1)).float()
        y_dist = (y - y.reshape(-1,1)).float()
        # make (r,θ) as edge_weights
        if polar:
            # this doesn't have self loops; diagonal elements are zero; [num_samples,num_samples]
            r = A*torch.sqrt(x_dist**2 + y_dist**2)/np.sqrt(H**2 + W**2)
            theta = A*torch.atan2(y_dist,x_dist)/np.pi
            edge_weights = torch.stack((r,theta),2)     # shape [num_samples,num_samples,num_edge_feat]
        # make (Δx,Δy) as edge_weights
        else:
            # this doesn't have self loops; diagonal elements are zero; [num_samples,num_samples]
            delX = A*x_dist/H
            delY = A*y_dist/W
            edge_weights = torch.stack((delX,delY),2)   # shape [num_samples,num_samples,num_edge_feat]
        return A, edge_weights

if __name__ == "__main__":
    loader = MNISTDataloader(10, 32, 0.5, 0, False)
    trainloader = loader.train_dataloader
    for i, (data, target) in enumerate(trainloader):
        geom_loader = loader.process_torch_geometric(data, num_samples=10, k=5, polar=True)
        for batch in geom_loader:
            print(f'Torch Geometric Batch: {batch}')
        break
    



