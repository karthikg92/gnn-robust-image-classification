import os
import argparse
import time
import wandb
from torch.utils.tensorboard import SummaryWriter
from utils import connected_to_internet

class WandbLogger:
    def __init__(self, experiment_name:str, save_folder:str, project:str, entity:str, args:argparse.Namespace):
        """
            Wandb Logger Wrapper
            Parameters:
            -----------
            experiment_name: str
                Name for logging the experiment on Wandboard
            save_folder: str
                Name of the folder to store wandb run files
            project: str
                Project name for wandboard
                Example: 'My Repo Name'
            entity: str
                Entity/username for wandboard
                Example: 'nsidn98'
            args: argparse.Namespace
                Experiment arguments to save
        """
        # check if internet is available; if not then change wandb mode to dryrun
        if not connected_to_internet():
            import json
            # save a json file with your wandb api key in your home folder as {'my_wandb_api_key': 'INSERT API HERE'}
            # NOTE this is only for running on systems without internet access
            # have to run `wandb sync wandb/run_name` to sync logs to wandboard
            with open(os.path.expanduser('~')+'/keys.json') as json_file: 
                key = json.load(json_file)
                my_wandb_api_key = key['my_wandb_api_key'] # NOTE change here as well
            os.environ["WANDB_API_KEY"] = my_wandb_api_key # my Wandb api key
            os.environ["WANDB_MODE"] = "dryrun"

        start_time = time.strftime("%H_%M_%S-%d_%m_%Y", time.localtime())
        experiment_name = f"{experiment_name}_{start_time}"
        
        print('_'*50)
        print('Creating wandboard...')
        print('_'*50)
        wandb_save_dir = os.path.join(os.path.abspath(os.getcwd()),f"wandb_{save_folder}")
        if not os.path.exists(wandb_save_dir):
            os.makedirs(wandb_save_dir)
        wandb.init(project=project, entity=entity, sync_tensorboard=True,\
                    config=vars(args), name=experiment_name,\
                    save_code=True, dir=wandb_save_dir)
        self.writer = SummaryWriter(f"{wandb.run.dir}/{experiment_name}")
        self.weight_save_path = os.path.join(wandb.run.dir, "model.ckpt")