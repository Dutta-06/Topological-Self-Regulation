import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
import json

from data.etth1 import get_etth1_loaders
from tsr.layers.tsr_lstm import TSRLSTM

class LSTMModel(nn.Module):
    def __init__(self, input_size=7, hidden_size=16, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
class GRUModel(nn.Module):
    def __init__(self, input_size=7, hidden_size=16, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

class TCNModel(nn.Module):
    def __init__(self, input_size=7, hidden_channels=16, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_channels, kernel_size, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=2, dilation=2)
        self.fc = nn.Linear(hidden_channels, 1)
        
    def forward(self, x):
        # x is (B, L, C). Conv1d expects (B, C, L)
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        # Take the last time step
        return self.fc(x[:, :, -1])

class DenseModel(nn.Module):
    def __init__(self, seq_len=96, input_size=7, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(seq_len * input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        
    def forward(self, x):
        # x is (batch, seq_len, input_size)
        x = x.reshape(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class TSRSeqModel(nn.Module):
    def __init__(self, input_size=7, hidden_size=16):
        super().__init__()
        self.lstm = TSRLSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def train_eval_loop(model_name, model, train_loader, val_loader, test_loader, device, epochs=10, is_tsr=False):
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    
    print(f"\nTraining {model_name}...")
    best_val_loss = float('inf')
    test_mse = float('inf')
    
    history = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for step, (x, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)):
            if step > 20: break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred.squeeze(), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for step, (x, y) in enumerate(val_loader):
                if step > 10: break
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += F.mse_loss(pred.squeeze(), y).item()
        val_loss /= len(val_loader)
        
        # TSR specific structural plasticity simulation (grow if we are not improving much)
        # For simplicity, we grow 4 neurons at epoch 5 to see if it escapes local minima
        if is_tsr and epoch == 4:
            print("--- Triggering Growth in TSRLSTM ---")
            model.lstm.cell.grow_neurons(4, init_scale=0.01)
            fc_weight = model.fc.weight.data
            new_cols = torch.zeros(model.fc.out_features, 4, device=device)
            model.fc.weight = nn.Parameter(torch.cat([fc_weight, new_cols], dim=1))
            model.fc.in_features += 4
            optimizer = Adam(model.parameters(), lr=0.001) # Reinit optimizer for new params
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Eval on test set
            test_loss = 0
            test_steps = 0
            with torch.no_grad():
                for step, (x, y) in enumerate(test_loader):
                    if step > 10: break
                    x, y = x.to(device), y.to(device)
                    pred = model(x)
                    test_loss += F.mse_loss(pred.squeeze(), y).item()
                    test_steps += 1
            test_mse = test_loss / test_steps
            
        print(f"Epoch {epoch+1} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f} | Test MSE: {test_mse:.4f}")
        
        stat = {
            "epoch": epoch + 1,
            "train_mse": train_loss,
            "val_mse": val_loss,
            "test_mse": test_mse,
        }
        if is_tsr:
            stat["hidden_size"] = model.lstm.cell.hidden_size
            stat["effective_neurons"] = model.lstm.cell.effective_neurons()
            stat["dominant_activation"] = model.lstm.cell.dominant_activation()
        history.append(stat)
        
    return history

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader, val_loader, test_loader = get_etth1_loaders(batch_size=128, seq_len=96, pred_len=1)
    
    # 1. Baseline standard LSTM (Static)
    baseline_model = LSTMModel(input_size=7, hidden_size=16)
    baseline_history = train_eval_loop("Baseline LSTM", baseline_model, train_loader, val_loader, test_loader, device, epochs=6, is_tsr=False)
    
    # 2. TSRLSTM (Dynamic) starts smaller, grows
    tsr_model = TSRSeqModel(input_size=7, hidden_size=12)
    tsr_history = train_eval_loop("TSR LSTM", tsr_model, train_loader, val_loader, test_loader, device, epochs=6, is_tsr=True)
    
    # 3. GRU (Static)
    gru_model = GRUModel(input_size=7, hidden_size=16)
    gru_history = train_eval_loop("Baseline GRU", gru_model, train_loader, val_loader, test_loader, device, epochs=6, is_tsr=False)
    
    # 4. TCN (Static)
    tcn_model = TCNModel(input_size=7, hidden_channels=16)
    tcn_history = train_eval_loop("Baseline TCN", tcn_model, train_loader, val_loader, test_loader, device, epochs=6, is_tsr=False)
    
    # 5. Dense/MLP (Static)
    dense_model = DenseModel(seq_len=96, input_size=7, hidden_size=64)
    dense_history = train_eval_loop("Baseline Dense", dense_model, train_loader, val_loader, test_loader, device, epochs=6, is_tsr=False)
    
    os.makedirs("results", exist_ok=True)
    with open("results/etth1_evaluation.json", "w") as f:
        json.dump({
            "lstm": baseline_history,
            "tsr_lstm": tsr_history,
            "gru": gru_history,
            "tcn": tcn_history,
            "dense": dense_history
        }, f, indent=2)
        
    print("\nBenchmark complete! Results saved to results/etth1_evaluation.json")

if __name__ == "__main__":
    main()
