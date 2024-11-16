# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 15:19:17 2024

@author: zhoushus
"""

# Import Packages
import sys
import os
import torch
from torch import nn
from torch import optim
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import warnings
import pickle
script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../script'))
sys.path.insert(0, script_dir)
script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../examples'))
sys.path.insert(0, script_dir)
from simple_nn_modules import GeneralNN
from simple_nn_trainer_lightningai import SimpleNNDataModule, SimpleNNTrainerLightning
from cal_ranking_by_freq import calRankingByFreq2

#Setting
num_cpus = os.cpu_count()
torch.set_num_threads(num_cpus-1)
warnings.filterwarnings("ignore")

# Lightning module for the GeneralNN architecture
class GeneralNNLightning(SimpleNNTrainerLightning):
    
    # Initialize the GeneralNN module
    def __init__(self, input_dim, hidden_sizes, dropout_rate, learning_rate,
                 pos_weight, patience, factor, seed):
        model = GeneralNN(input_dim, hidden_sizes, dropout_rate)
        self.learning_rate = learning_rate
        self.pos_weight = pos_weight
        self.patience = patience
        self.factor = factor
        self.seed = seed
        super(GeneralNNLightning, self).__init__(
            model=model,
            optimizer=None,
            scheduler=None,
            criterion=nn.BCEWithLogitsLoss(pos_weight=pos_weight),
            seed = self.seed
        )

    # Configure the optimizer and scheduler
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=self.patience, factor=self.factor, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_auc"},
        }
    


#Setting
num_cpus = os.cpu_count()
torch.set_num_threads(num_cpus-1)
warnings.filterwarnings("ignore")

# Define model name
model_name="GeneralNN_for_breast_cancer"
standardscaler_filename = f"./{model_name}/{model_name}_standardscaler.pkl"

# Load data
data = load_breast_cancer()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Preprocess data
standardscaler = StandardScaler()
X_train = standardscaler.fit_transform(X_train)
with open(standardscaler_filename, "wb") as f:
    pickle.dump(standardscaler, f)
X_val = standardscaler.transform(X_val)
X_test = standardscaler.transform(X_test)

# Define model parameters
input_dim = X.shape[1]
hidden_sizes = [128, 64, 32]
dropout_rate = 0.2
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pos_weight = torch.tensor([1 / y_train.mean() - 1], device=device)
patience = 3
factor = 0.1
seed = 42

# Initialize the model
trainer_lightning = GeneralNNLightning(
    input_dim=input_dim,
    hidden_sizes=hidden_sizes,
    dropout_rate=dropout_rate,
    learning_rate=learning_rate,
    pos_weight=pos_weight,
    patience=patience,
    factor=factor,
    seed=seed
)

# Initialize the data module
batch_size = 64
datamodule = SimpleNNDataModule(
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    X_test=X_test, y_test=y_test,
    batch_size=batch_size
)

# Callbacks for model checkpointing and early stopping
monitor = "val_auc"
mode = "max"
patience = 5
dirpath=f"./{model_name}/"
checkpoint_callback = ModelCheckpoint(
    monitor=monitor,
    mode=mode,
    save_top_k=1,
    dirpath=dirpath,
    filename=model_name,
    save_last=True
)
early_stopping_callback = EarlyStopping(
    monitor=monitor,
    mode=mode,
    patience=patience,
    verbose=True
)

# PyTorch Lightning Trainer
max_epochs = 8964
log_every_n_steps = 10
trainer = Trainer(
    max_epochs=max_epochs,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    callbacks=[checkpoint_callback, early_stopping_callback],
    log_every_n_steps=log_every_n_steps,
    precision="16-mixed"
)

# Train the model
trainer.fit(trainer_lightning, datamodule)

# Evaluate the model on the test set
trainer.test(trainer_lightning, datamodule)

#-------------------------

# Define model name
model_name = "GeneralNN_for_breast_cancer"
standardscaler_filename = f"./{model_name}/{model_name}_standardscaler.pkl"
ckpt_filename = f"./{model_name}/{model_name}.ckpt"

# Preprocess new data
data = load_breast_cancer()
X, y = data.data, data.target

# Load the saved StandardScaler for data preprocessing
with open(standardscaler_filename, "rb") as f:
    standardscaler = pickle.load(f)
X = standardscaler.transform(X)

# Define model parameters
input_dim = X.shape[1]
hidden_sizes = [128, 64, 32]
dropout_rate = 0.2
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pos_weight = torch.tensor([1 / y_train.mean() - 1], device=device)
patience = 3
factor = 0.1
seed = 42

# Initialize the model
model = GeneralNNLightning(
    input_dim=input_dim,
    hidden_sizes=hidden_sizes,
    dropout_rate=dropout_rate,
    learning_rate=learning_rate,
    pos_weight=pos_weight,
    patience=patience,
    factor=factor,
    seed=seed
).to(device)

# Load the state_dict into the model
checkpoint = torch.load(ckpt_filename, map_location=device)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# Convert data to PyTorch tensor
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

# Make predictions using the inner model
with torch.no_grad():
    outputs = model.model(X_tensor)
    y_pred = torch.sigmoid(outputs).cpu().numpy().ravel()
    
df2 = pd.DataFrame({"score":y_pred,
                   "y":y})

# Evaluate model
tmp = calRankingByFreq2(df2, label="y", score="score", bins=10)

