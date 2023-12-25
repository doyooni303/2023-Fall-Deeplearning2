import pandas as pd
import os

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Function to load data from CSV and process it
def load_and_process_data(file_name):
    df = pd.read_csv(file_name)
    features = df.iloc[:, 1:-1].to_numpy()
    labels = df.iloc[:, -1].to_numpy()

    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).view(-1, 1)

    return features_tensor, labels_tensor


# Load the datasets
script_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = 'data_reconstructed'

X_train, y_train = load_and_process_data(os.path.join(script_dir, data_folder, 'train_False_df2.csv'))
X_val, y_val = load_and_process_data(os.path.join(script_dir, data_folder, 'valid_False_df2.csv'))
X_test, y_test = load_and_process_data(os.path.join(script_dir, data_folder, 'test_df2.csv'))

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=False)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

n_features = X_train.shape[1] 
n_targets = 1


class MLPRegressor(nn.Module):
    def __init__(self, n_features, n_targets, num_layers, hidden_size, dropout_rate=0):
        super(MLPRegressor, self).__init__()
        layers = [nn.Linear(n_features, hidden_size), nn.ReLU()]
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(hidden_size, n_targets))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# Example of model instantiation with different layer sizes
num_layers = 3
hidden_size = 128
model = MLPRegressor(n_features, n_targets, num_layers, hidden_size, dropout_rate=0)


# Define loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

# Training loop
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

# Validation function
def validate(dataloader, model, loss_fn):
    model.eval()
    total_val_loss = 0
    total_val_mape = 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            loss = loss_fn(pred, y)
            total_val_loss += loss.item()
            total_val_mape += mean_absolute_percentage_error(y.numpy(), pred.numpy())
            
    avg_val_loss = total_val_loss / len(dataloader)
    avg_val_mape = total_val_mape / len(dataloader)
    return avg_val_loss, avg_val_mape

# Train and validate the model
epochs = 201
for epoch in range(epochs):
    train_loss = train(train_loader, model, loss_fn, optimizer)
    val_loss, val_mape = validate(val_loader, model, loss_fn)
    if epoch % 1 == 0:  # Print every 10 epochs
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val MAPE: {val_mape:.4f}")

# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    test_loss = loss_fn(y_pred, y_test).item()
    test_mape = mean_absolute_percentage_error(y_test.numpy(), y_pred.numpy())

print(f"Test Loss: {test_loss:.4f}")
print(f"Test MAPE: {test_mape:.4f}")
