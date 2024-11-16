# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 14:44:41 2024

@author: zhoushus
"""

# Import Packages
import sys
import os
import torch
from torch import nn
from torch import optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import warnings
script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../script'))
sys.path.insert(0, script_dir)
script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../examples'))
sys.path.insert(0, script_dir)
from simple_nn_modules import GeneralNN
from simple_nn_trainer_pytorch import SimpleNNTrainer
from cal_ranking_by_freq import calRankingByFreq2

#Setting
num_cpus = os.cpu_count()
torch.set_num_threads(num_cpus-1)
warnings.filterwarnings("ignore")

# Define model name
model_name="GeneralNN_for_breast_cancer"

# Load data
data = load_breast_cancer()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Preprocess data
standardscaler = StandardScaler()
X_train = standardscaler.fit_transform(X_train)
X_val = standardscaler.fit_transform(X_val)
X_test = standardscaler.transform(X_test)

# Define model parameters
input_dim = X.shape[1]
hidden_sizes = [128, 64, 32]
dropout_rate = 0.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model
model = GeneralNN(input_dim=input_dim,
                  hidden_sizes=hidden_sizes,
                  dropout_rate=dropout_rate).to(device)

# Initialize optimizer, scheduler, and criterion
learning_rate = 0.001 
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler_patience = 3
factor = 0.1
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                 mode="max",
                                                 patience=scheduler_patience,
                                                 factor=factor, verbose=True)
pos_weight = torch.tensor([1 / y.mean() - 1], device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
grad_scaler = GradScaler()
batch_size = 64

# Initialize trainer
num_epochs = 8964
trainer = SimpleNNTrainer(model=model,
                          optimizer=optimizer,
                          scheduler=scheduler,
                          criterion=criterion,
                          device=device,
                          #grad_scaler = grad_scaler,
                          batch_size = batch_size,
                          model_name = model_name,
                          num_epochs=num_epochs)

# Train model
trainer.train(X_train, y_train, X_val, y_val)

# Evaluate model
trainer.test(X_test, y_test)
trainer.save_trainer_to_pkl(filename=f"./{model_name}/{model_name}.pkl",
                            extra_objects=[standardscaler])

#-------------------------

# Define model name
model_name="GeneralNN_for_breast_cancer"

# Load new data
data = load_breast_cancer()
X, y = data.data, data.target

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainer, extra_objects = SimpleNNTrainer.load_trainer_from_pkl(filename=f"./{model_name}/{model_name}.pkl",
                                                               device=device)

# Preprocess new data
standardscaler = extra_objects[0]
X = standardscaler.fit_transform(X)

#Make loader
batch_size = 128
X_tensor, y_tensor = torch.tensor(X, dtype=torch.float32).to(device), torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)
loader = DataLoader(TensorDataset(X_tensor, y_tensor),
                    batch_size=batch_size,
                    shuffle=True)

#Set the model environment
trainer.device = device
trainer.model.to(device)
trainer.model.eval()

# Make predictions using the inner model
y_pred = trainer.predict_proba(X)[:,1]
df2 = pd.DataFrame({"score":y_pred,
                   "y":y})

# Evaluate model
auc = trainer.evaluate_model(loader, draw_roc=True)
tmp = calRankingByFreq2(df2, label="y", score="score", bins=10)

