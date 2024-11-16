# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 21:48:57 2024

@author: zhoushus
"""

# Import Packages
import os
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score
import numpy as np
import random
import warnings

#Setting
num_cpus = os.cpu_count()
torch.set_num_threads(num_cpus-1)
warnings.filterwarnings("ignore")

# Custom dataset for handling features and labels
class SimpleDataset(Dataset):
    
    # Initialize the dataset
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    # Get the length of the dataset
    def __len__(self):
        return len(self.X)

    # Get a specific item from the dataset
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
# Data module for managing training, validation, and test data
class SimpleNNDataModule(pl.LightningDataModule):
    
    # Initialize the data module
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, batch_size):
        super(SimpleNNDataModule, self).__init__()
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test
        self.batch_size = batch_size

    # Set up the datasets
    def setup(self, stage=None):
        self.train_dataset = SimpleDataset(self.X_train, self.y_train)
        self.val_dataset = SimpleDataset(self.X_val, self.y_val)
        self.test_dataset = SimpleDataset(self.X_test, self.y_test)

    # Get the training data loader
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    # Get the validation data loader
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    # Get the test data loader
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
    
# Base Lightning module for training and evaluation
class SimpleNNTrainerLightning(pl.LightningModule):
    
# Initialize the trainer module
    def __init__(self, model, optimizer, scheduler, criterion, seed):
        super(SimpleNNTrainerLightning, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.train_outputs = []
        self.val_outputs = []
        self.test_outputs = []
        self.__set_seed()
        
    # Sets a random seed for reproducibility
    def __set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Define the forward pass
    def forward(self, x):
        return self.model(x)
    
    # Compute AUC metric for an output
    @staticmethod
    def compute_metrics(outputs):
        y_true = torch.cat([o["y_true"] for o in outputs], dim=0).cpu().numpy()
        y_pred = torch.cat([o["y_pred"] for o in outputs], dim=0).sigmoid().detach().cpu().numpy()
        auc = roc_auc_score(y_true, y_pred)
        return auc
    
    # Compute loss and outputs for a batch data
    def compute_loss_and_outputs(self, batch):
        batch_X, batch_y = batch
        outputs = self(batch_X)
        loss = self.criterion(outputs, batch_y)
        return loss, outputs

    # Executes a single training step
    def training_step(self, batch, batch_idx):
        loss, outputs = self.compute_loss_and_outputs(batch)
        self.train_outputs.append({"y_true": batch[1], "y_pred": outputs})
        return {"loss": loss}
    
    # Called at the start of the training epoch
    def on_train_epoch_start(self):
        self.train_outputs = []
    
    # Called at the end of the training epoch to calculate and log AUC
    def on_train_epoch_end(self):
        train_auc = self.compute_metrics(self.train_outputs)
        print(f"Epoch [{self.current_epoch + 1}] Train AUC: {train_auc:.4f}")
        self.log("train_auc", train_auc, prog_bar=True)
    
    # Executes a single validation step
    def validation_step(self, batch, batch_idx):
        loss, outputs = self.compute_loss_and_outputs(batch)
        self.val_outputs.append({"y_true": batch[1], "y_pred": outputs})
        return {"val_loss": loss}
    
    # Called at the start of the validation epoch
    def on_validation_epoch_start(self):
        self.val_outputs = []
    
    # Called at the end of the validation epoch to calculate and log AUC
    def on_validation_epoch_end(self):
        val_auc = self.compute_metrics(self.val_outputs)
        print(f"Epoch [{self.current_epoch + 1}] Val AUC: {val_auc:.4f}")
        self.log("val_auc", val_auc, prog_bar=True)
        
    # Executes a single test step
    def test_step(self, batch, batch_idx):
        loss, outputs = self.compute_loss_and_outputs(batch)
        y_pred = torch.sigmoid(outputs)
        self.test_outputs.append({"y_true": batch[1], "y_pred": y_pred})
        
        return {"test_loss": loss}
    
    # Called at the start of the test epoch
    def on_test_epoch_start(self):
        self.test_outputs = []
    
    # Called at the end of the test epoch
    def on_test_epoch_end(self):
        test_auc = self.compute_metrics(self.test_outputs)
        print(f"Test AUC: {test_auc:.4f}")
        self.log("test_auc", test_auc, prog_bar=True)

    # Configure the optimizer and scheduler
    def configure_optimizers(self):
        
        if self.scheduler:
            return {"optimizer": self.optimizer, "lr_scheduler": {"scheduler": self.scheduler, "monitor": "val_auc"}}
        
        return self.optimizer
    
