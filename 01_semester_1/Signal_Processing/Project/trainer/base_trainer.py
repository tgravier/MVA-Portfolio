import time
from pathlib import Path

import json5
import numpy as np
import torch
import os
from torch.optim.lr_scheduler import StepLR
from util.utils import prepare_empty_dir, ExecutionTime


class BaseTrainer:
    def __init__(self,
                 config,
                 resume:bool,
                 model,
                 loss_function,
                 optimizer):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.optimizer = optimizer
        self.loss = loss_function

        self.model = model.to(self.device)

        # For the trainer :

        self.epochs = config["trainer"]["epochs"]
        self.save_checkpoint_interval = config["trainer"]["save_checkpoint_interval"]
        self.validation_config = config["trainer"]["validation"]
        self.find_max = self.validation_config["find_max"]
        self.validation_interval = self.validation_config["interval"]
        self.validation_custom_config = self.validation_config["custom"]

         # The following args is not in the config file. We will update it if the resume is True in later.
        self.start_epoch = 1
        self.best_score = -np.inf if self.find_max else np.inf
        self.root_dir = Path(__file__).resolve().parent
        self.checkpoints_dir = self.root_dir / "checkpoints"
        self.logs_dir = self.root_dir / "logs"
        prepare_empty_dir([self.checkpoints_dir, self.logs_dir], resume=resume)

        if resume:
            self._resume_checkpoint()
        
        #print("Configurations are as follows:")
        #print(json5.dumps(config, indent=2, sort_keys=False))


    def _resume_checkpoint(self):
        """Resume experiment from the latest checkpoint.
        """
        latest_model_path = self.checkpoints_dir.expanduser().absolute() / "latest_model.tar"
        assert latest_model_path.exists(), f"{latest_model_path} does not exist, can not load latest checkpoint."

        checkpoint = torch.load(latest_model_path.as_posix(), map_location=self.device)

        self.start_epoch = checkpoint["epoch"] + 1
        self.best_score = checkpoint["best_score"]
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(checkpoint["model"])
        else:
            self.model.load_state_dict(checkpoint["model"])

        print(f"Model checkpoint loaded. Training will begin in {self.start_epoch} epoch.")

    def _save_checkpoint(self, epoch, is_best=False):
        """Save checkpoint to <root_dir>/checkpoints directory.
        It contains:
            - current epoch
            - best score in history
            - optimizer parameters
            - model parameters
        Args:
            is_best(bool): if current checkpoint got the best score, it also will be saved in <root_dir>/checkpoints/best_model.tar.
        """
        print(f"\t Saving {epoch} epoch model checkpoint...")

        # construct checkpoint tar package

        state_dict = {

            "epoch": epoch,
            "best_score": self.best_score,
            "optimizer": self.optimizer.state_dict(),
        }

        if isinstance(self.model, torch.nn.DataParallel):
            state_dict["model"] = self.model.module.state_dict()
        else:
            state_dict["model"] = self.model.cpu().state_dict()
        """
        Notes:
            - latest_model.tar:
                Contains all checkpoint information, including optimizer parameters, model parameters, etc. New checkpoint will overwrite old one.
            - model_<epoch>.pth: 
                The parameters of the model. Follow-up we can specify epoch to inference.
            - best_model.tar:
                Like latest_model, but only saved when <is_best> is True.
        """

        torch.save(state_dict, (self.checkpoints_dir / "latest_model.tar").as_posix())
        torch.save(state_dict["model"], (self.checkpoints_dir / f"model_{str(epoch).zfill(4)}.pth").as_posix())
        if is_best:
            print(f"\t Found best score in {epoch} epoch, saving...")
            torch.save(state_dict, (self.checkpoints_dir / "best_model.tar").as_posix())

        # Use model.cpu() or model.to("cpu") will migrate the model to CPU, at which point we need remigrate model back.
        # No matter tensor.cuda() or tensor.to("cuda"), if tensor in CPU, the tensor will not be migrated to GPU, but the model will.
        self.model.to(self.device)

    def _is_best(self, score, find_max=True):
        """Check if the current model is the best model.
        Args:
            score(float): current score
            find_max(bool): if True, the greater the score, the better the model.
        """
        if find_max and score > self.best_score:
            self.best_score = score
            return True
        if not find_max and score < self.best_score:
            self.best_score = score
            return True
        return False
    
    @staticmethod
    def _transform_pesq_range(pesq_score):

        """ transform PESQ range from [-0.5, 4.5] to [0, 1]"""
        return (pesq_score + 0.5) / 5.0
    
    def _set_models_to_train_mode(self):
        self.model.train()
    
    def _set_models_to_eval_mode(self):
        self.model.eval()
    
    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            print(f"============== {epoch} epoch ==============")
            print("[0 seconds] Begin training...")
            timer = ExecutionTime()

            self._set_models_to_train_mode()
            self._train_epoch(epoch)


            if self.save_checkpoint_interval != 0 and (epoch % self.save_checkpoint_interval == 0):
                self._save_checkpoint(epoch)

            if self.validation_interval != 0 and epoch % self.validation_interval == 0:
                print(f"[{timer.duration()} seconds] Training is over. Validation is in progress...")

                self._set_models_to_eval_mode()
                score = self._validation_epoch(epoch)
                print(score)

                if self._is_best(score, find_max=self.find_max):
                    self._save_checkpoint(epoch, is_best=True)

            print(f"[{timer.duration()} seconds] End this epoch.")

def _train_epoch(self, epoch):
    raise NotImplementedError

def _validation_epoch(self, epoch):
    raise NotImplementedError
