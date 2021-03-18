import os
import time
from tqdm import trange
import argparse
import numpy as np
import torch
import torch.nn as nn

from dataloader import MNISTDataloader
from model import GNNConvNet
from utils import print_args, print_box

class Trainer():
    def __init__(self, args:argparse.Namespace, device:str, logger=None):
        self.args = args
        self.logger = logger

        self.device = device
        ### seeds ###
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        ####################

        ### dataloader ###
        self.mnist_dataloader = MNISTDataloader(num_nodes=self.args.num_nodes, batch_size=self.args.batch_size,\
                                                train_val_split_ratio=self.args.train_val_split_ratio,\
                                                seed=self.args.seed, remove9=self.args.remove_9, testing=False)
        self.trainLoader = self.mnist_dataloader.train_dataloader
        self.valLoader   = self.mnist_dataloader.val_dataloader
        ####################

        ### network init ###
        self.network = GNNConvNet(in_feat=self.args.in_feat, edge_feat=self.args.edge_feat, num_class=self.args.num_class).to(self.device)
        print_box(self.network, num_dash=80)
        print_box(f'Is the Network on CUDA?: {next(self.network.parameters()).is_cuda}')
        ####################

        self.optimizer = getattr(torch.optim, self.args.opt)(self.network.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.loss_function = nn.CrossEntropyLoss()

    def train(self):
        train_step = 0
        prog_bar = trange(self.args.epoch)
        for epoch in range(self.args.epoch):

            correct = 0; total = 0; epoch_loss = 0
            for batch_idx, (data, target) in enumerate(self.trainLoader):
                # if dryrun then only run for 100 batches
                if self.args.dryrun and batch_idx == 100:
                    break
                ############
                geom_loader = self.mnist_dataloader.process_torch_geometric(data, self.args.num_nodes, k=self.args.num_neighbours, polar=self.args.polar)
                for batch in geom_loader:
                    batch = batch.to(self.device)
                    target = target.to(self.device)

                    self.optimizer.zero_grad()
                    # forward pass
                    output = self.network(batch)
                    loss = self.loss_function(output, target)
                    ##############

                # accuracy prediction
                _, predicted = torch.max(output.data,1)
                correct += (predicted == target).sum().item()
                total   += predicted.shape[0]
                epoch_loss += loss.item()
                ##############

                # backprop
                loss.backward()
                self.optimizer.step()
                ##############

                if train_step % self.args.save_freq and self.logger:
                    self.save_checkpoint(self.logger.weight_save_path, train_step)

                train_step += 1
                prog_bar.set_description(f'Epoch={epoch} Loss (loss={loss.item():.3f})')
            
            # validation
            self.network.eval()
            val_correct = 0; val_total = 0
            for j, (val_data, val_target) in enumerate(self.valLoader):
                if self.args.dryrun and j == 100:
                    break
                geom_loader_val = self.mnist_dataloader.process_torch_geometric(val_data, self.args.num_nodes, k=self.args.num_neighbours, polar=self.args.polar)
                for batch in geom_loader_val:
                    batch = batch.to(self.device)
                    val_target = val_target.to(self.device)
                    
                    with torch.no_grad():
                        val_output = self.network(batch)
                    _, val_predicted = torch.max(val_output.data,1)
                    val_correct += (val_predicted == val_target).sum().item()
                    val_total   += val_predicted.shape[0]
            # important to set network back to training mode
            self.network.train()

            print_box(f'Epoch: {epoch}, Epoch Loss:{epoch_loss/len(self.trainLoader):.3f}, Train Accuracy:{correct/total:.3f} Val Accuracy:{val_correct/val_total:.3f}')
            if self.logger:
                self.logger.writer.add_scalar('Epoch loss', epoch_loss/len(self.trainLoader), epoch)
                self.logger.writer.add_scalar('Train Accuracy', correct/total, epoch)
                self.logger.writer.add_scalar('Val Accuracy', val_correct/val_total, epoch)
            prog_bar.update(1)

    def save_checkpoint(self, path:str, train_step_num: int):
        """
            Saves the model in the wandb experiment run directory
            This will store the 
                * model state_dict
                * args:
                    Will save this as a Dict as well as argparse.Namespace
            param:
                path: str
                    path to the wandb run directory
                    Example: wandb.run.dir
                train_step_num: int
                    The train step number at which model is getting saved
        """
        checkpoint = {}
        checkpoint['args'] = self.args
        checkpoint['args_dict'] = vars(self.args)
        checkpoint['state_dict'] = self.network.state_dict()
        checkpoint['train_step_num'] = train_step_num
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path:str):
        """
            Load the trained model weights
            param:
                path: str
                    path to the saved weights file
        """
        use_cuda = torch.cuda.is_available()
        device   = torch.device("cuda" if use_cuda else "cpu")
        checkpoint_dict = torch.load(path, map_location=device)
        self.network.load_state_dict(checkpoint_dict['state_dict']) 

if __name__ == "__main__":
    from config import args
    from logger import WandbLogger

    print_args(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_box(f'Device:{device}') 

    if args.dryrun:
        logger = None
    else:
        logger = WandbLogger(experiment_name='gnnMNIST', save_folder='gnn', project='robustGNN', entity='robust_gnn', args=args)

    trainer = Trainer(args, device, logger)
    trainer.train()