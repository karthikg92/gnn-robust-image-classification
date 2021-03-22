import argparse
from distutils.util import strtobool

parser = argparse.ArgumentParser(description='GNN robust image classification')

##################### Experimental values #####################
parser.add_argument('--num_nodes', type=int, default=50, help='Number of pixels to sample from the image')
parser.add_argument('--epoch', type=int, default=1000, help='Number of epochs for training')
##################################################################

##################### data #####################
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--train_val_split_ratio', type=float, default=0.2, 
                    help='Ratio to split dataset in train and validation sets;\
                    train_val_split_ratio in val and (1-train_val_split_ratio) in train')
parser.add_argument('--num_neighbours', type=int, default=20, help='Number of neighbouring pixels to connect to')
parser.add_argument('--polar', type=lambda x:bool(strtobool(x)), default=False, help='Whether we want polar coords for edge_weights')
parser.add_argument('--remove_9', type=lambda x:bool(strtobool(x)), default=True, help='If True, remove images with digit 9 from the dataset')
##################################################################

##################### network architecture #####################
parser.add_argument('--in_feat', type=int, default=1, help='Dimensionality of node features')
parser.add_argument('--edge_feat', type=int, default=2, help='Dimensionality of edge features')
parser.add_argument('--num_class', type=int, default=10, help='Number of classes for classification')
parser.add_argument('--K', type=int, default=1, help='Number of K hops in SGConv')
##################################################################

##################### optimizer #####################
parser.add_argument('--opt', type=str, default='Adam', help='Optimizer choice.\
                    Has to be equal to pytorch class name for optimizer in torch.optim')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for optimizer')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 penalty) on weights')
##################################################################

##################### book keeping #####################
parser.add_argument('--save_freq', type=int, default=1000, help='Number of steps after which to save the weights of the network')
parser.add_argument('--seed', type=int, default=0, help='Seed for all randomness')
parser.add_argument('--dryrun', type=lambda x:bool(strtobool(x)), default=False, help='If just testing the code')
##################################################################

def config_check(args:argparse.Namespace):
    """
        Check if the arguments are compatible
    """
    # if dryrun the don't run for default number epochs
    if args.dryrun:
        args.epoch = 5
    # if 9 is removed from the dataset, then reduce num_class
    if args.remove_9:
        args.num_class = 9
    return args

args = parser.parse_args()
args = config_check(args)