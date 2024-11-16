# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 16:01:25 2024

@author: zhoushus
"""

# Import Packages
import os
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt
import random
import warnings
import pickle

#Setting
num_cpus = os.cpu_count()
torch.set_num_threads(num_cpus-1)
warnings.filterwarnings("ignore")

# Define the SimpleNNTrainer
class SimpleNNTrainer:
    
    # Initializes the SimpleNNTrainer class
    def __init__(self, model, optimizer, scheduler, criterion, device, model_name,\
                 batch_size=128, num_epochs=1000, patience=5, seed=42, grad_scaler=None):
        self.device = device
        print(f"Using device: {self.device}")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion.to(self.device)
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience
        self.seed = seed
        self.best_auc = 0.0
        self.best_epoch = 0
        self.early_stop_count = 0
        self.grad_scaler = grad_scaler
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

    # Creates data loaders for training and validation datasets
    def create_loaders(self, X_train, y_train, X_val, y_val):
        
        # Create a data loader
        def create_loader(X, y, shuffle):
            dataset = TensorDataset(torch.tensor(X, dtype=torch.float32).to(self.device),
                                    torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(self.device))
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        
        return create_loader(X_train, y_train, True), create_loader(X_val, y_val, False)
    
    # Converts data to tensors and moves them to the device
    def __to_tensor(self, X, y):
        return (torch.tensor(X, dtype=torch.float32).to(self.device),
                torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(self.device))

    # Trains the model on the training dataset and validates on the validation dataset at each epoch
    def train(self, X_train, y_train, X_val, y_val):
        train_loader, val_loader = self.create_loaders(X_train, y_train, X_val, y_val)
        
        for epoch in range(self.best_epoch, self.num_epochs):
            self.__run_epoch(train_loader, training=True)
            train_auc = self.evaluate_model(train_loader)
            val_auc = self.evaluate_model(val_loader)
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")
            self.scheduler.step(val_auc)

            if val_auc > self.best_auc:
                self.best_auc, self.best_epoch, self.early_stop_count = val_auc, epoch + 1, 0
                self.save_checkpoint(epoch)
            else:
                self.early_stop_count += 1
                
                if self.early_stop_count >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch+1}.")
                    break

    # Tests the model on a given test dataset and calculates the AUC score
    def test(self, X_test, y_test):
        self.load_checkpoint(f"./{self.model_name}/{self.model_name}.pt")
        test_loader = DataLoader(TensorDataset(*self.__to_tensor(X_test, y_test)), batch_size=self.batch_size, shuffle=False)
        auc_score = self.evaluate_model(test_loader, draw_roc=True)
        print(f"AUC Score on test set: {auc_score:.4f}")
        
    # Executes a training step for a single batch or epoch
    def __run_epoch(self, loader, training=True):
        self.model.train(training)
        
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            self.optimizer.zero_grad()
            
            if self.grad_scaler:
                
                with autocast():
                    loss = self.criterion(self.model(batch_X, None), batch_y)
                    
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                loss = self.criterion(self.model(batch_X, None), batch_y)
                loss.backward()
                self.optimizer.step()
    
    # Predict probabilities for the given input X, handling data in batches
    def predict_proba(self, X):
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return self.predict(loader, return_true_labels=False)
    
    # Evaluates the model performance on a given dataset loader
    def evaluate_model(self, loader, draw_roc=False):
        y_true, y_pred = self.predict(loader, return_true_labels=True)
        auc_score = roc_auc_score(y_true, y_pred)
        
        if draw_roc:
            self.__plot_roc_curve(y_true, y_pred, auc_score)
            
        return auc_score
    
    # Predict probabilities for a given DataLoader in a binary classification setting
    def predict(self, loader, return_true_labels=False):
        self.model.eval()
        y_true, y_pred = [], []
        
        with torch.no_grad():
            
            for batch in loader:
                batch_X = batch[0].to(self.device)
                outputs = torch.sigmoid(self.model(batch_X))
                y_pred.extend(outputs.cpu().numpy())
                
                if return_true_labels:
                    y_true.extend(batch[1].cpu().numpy())
        
        y_pred = np.array(y_pred)
        
        if return_true_labels:
            return np.array(y_true), y_pred
        else:
            return np.column_stack([1 - y_pred, y_pred])

    # Plots the ROC curve for model evaluation
    def __plot_roc_curve(self, y_true, y_pred, auc_score):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        ks = tpr[abs(tpr - fpr).argmax()] - fpr[abs(tpr - fpr).argmax()]
        plt.plot(fpr, fpr, label="Random Guess")
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.3f})")
        plt.plot([fpr[abs(tpr - fpr).argmax()]] * len(fpr), np.linspace(fpr[abs(tpr - fpr).argmax()], tpr[abs(tpr - fpr).argmax()], len(fpr)), "--")
        plt.title(f"ROC Curve of {self.model_name}")
        plt.legend()
        print(f"KS = {ks:.3f}\nAUC = {auc_score:.3f}")
        plt.show()
        
    # Saves the model data as a file
    def save_checkpoint(self, epoch, filename=None):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_auc": self.best_auc,
            "best_epoch": self.best_epoch,
            "early_stop_count": self.early_stop_count,
        }
    
        if self.grad_scaler:
            checkpoint["grad_scaler_state_dict"] = self.grad_scaler.state_dict()
    
        filename = filename or f"./{self.model_name}/{self.model_name}.pt"
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved to {filename}")
    
    # Loads the checkpoint from a specified path, optionally including optimizer, scheduler, and scaler states
    def load_checkpoint(self, path, load_optimizer=False, load_scheduler=False, load_grad_scaler=False):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.best_auc = checkpoint.get("best_auc", 0.0)
        self.best_epoch = checkpoint.get("best_epoch", 0)

        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("Optimizer state loaded from checkpoint.")
            
        if load_scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print("Scheduler state loaded from checkpoint.")
            
        if load_grad_scaler and "grad_scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
            print("Grad_scaler state loaded from checkpoint.")
            
        self.model.to(self.device)    
        print(f"Loaded model state from '{path}' with best AUC {self.best_auc:.4f} at epoch {self.best_epoch} to device {self.device}.")

    #Saves the SimpleNNTrainer object (including model) to a .pkl file
    def save_trainer_to_pkl(self, filename=None, extra_objects=None):
        filename = filename or f"{self.model_name}_full_trainer.pkl"
        objects_to_save = (extra_objects or []) + [self]
        self.model.to("cpu")
        
        with open(filename, "wb") as f:
            pickle.dump(objects_to_save, f)
            
        print(f"Trainer (including model) saved to {filename}")
        self.model.to(self.device)
    
    #Load the SimpleNNTrainer object from a .pkl file and move model to device
    @staticmethod
    def load_trainer_from_pkl(filename, device):
        
        with open(filename, "rb") as f:
            objects = pickle.load(f)
            
        trainer = None
        extra_objects = []
        
        for obj in objects:
            
            if isinstance(obj, SimpleNNTrainer):
                trainer = obj
            else:
                extra_objects.append(obj)
        
        if trainer is None:
            raise ValueError("The file does not contain a SimpleNNTrainer instance.")
        
        trainer.device = device
        trainer.model.to(device)
        print(f"Trainer (including model) loaded from {filename} and moved to {device}")
        return trainer, extra_objects

