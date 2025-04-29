import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from seaborn import histplot

torch.manual_seed(1) # for reproducibility

# Step 3: Preprocess Data
class WrightFisherDataset(Dataset):
    def __init__(self, data, params):
        self.data = data  # Shape: (num_samples, num_generations)
        self.params = params  # Shape: (num_samples, num_params)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].unsqueeze(0), self.params[idx]  # Data shape: (1, num_generations)

# Step 4: Define RNN Model
class RNNRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(RNNRegressor, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),

            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        # x: (batch_size, 1, num_generations)
        x = x.squeeze(1)  # Shape: (batch_size, num_generations)
        x = x.unsqueeze(-1)  # Shape: (batch_size, num_generations, 1)
        rnn_out, _ = self.rnn(x)  # Shape: (batch_size, num_generations, hidden_dim)
        x = rnn_out[:, -1, :]  # Use the last time step: (batch_size, hidden_dim)
        x = self.fc(x)  # Shape: (batch_size, output_dim)
        return x

# Step 5: Training and Evaluation
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, device):
    model.eval()
    predictions = []
    true_values = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            predictions.append(outputs.cpu())
            true_values.append(targets.cpu())

    predictions = torch.cat(predictions)
    true_values = torch.cat(true_values)
    
    return predictions, true_values
    
    
if __name__ == "__main__":
    # Hyperparameters
    batch_size = 50
    num_epochs = 100
    lr = 1e-4
    TRAIN_SPLIT = 0.9

    # Data
    x, theta = torch.load('test_sims/x.pt'), torch.load('test_sims/theta.pt')
    n_samples = x.shape[0]
    train_size = int(n_samples * TRAIN_SPLIT)
    train_x = x[:train_size]
    train_theta = theta[:train_size]
    val_x = x[train_size:]
    val_theta = theta[train_size:]

    train_dataset = WrightFisherDataset(train_x, train_theta)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_dataset = WrightFisherDataset(val_x, val_theta)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model and Training Setup
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = RNNRegressor(
        input_dim=1,  # Each time step has one value (allele frequency)
        hidden_dim=50,
        num_layers=6,
        output_dim=theta.shape[1]
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Train Model
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        val_outputs, val_targets = evaluate_model(model, val_loader, device)
        val_loss = criterion(val_outputs, val_targets)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        
    # Test model
    for sim in ['WF', 'WF_DFE', 'WF_bottleneck', 'combined_mis', 'combined_miss']:
        data, params = torch.load(f'test_sims/test_x_{sim}.pt'), torch.load(f'test_sims/test_theta_{sim}.pt')
        test_dataset = WrightFisherDataset(data, params)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # This shuffle was one hell of a bug :(
        
        ests, true_vals = evaluate_model(model, test_loader, device)
        print(sim)
        print(r2_score(ests[:,0].numpy(),true_vals[:,0].numpy()))
        torch.save(ests, f'test_sims/predictions_{sim}_rnn.pt')