import os
import math
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

def tsr_lstm_growth_signal(cell, val_loss_history, growth_rate=0.25,
                           saturation_frac=0.75, plateau_tol=0.01, max_hidden=128):
    """Decide how many hidden units to add to a TSRLSTMCell from its own state.

    Mirrors the main TSR engine's bottleneck+plateau logic, adapted to the
    recurrent cell, so growth is driven by intrinsic signals rather than a
    hardcoded schedule:

      - Saturation: a large fraction of gates are near their ceiling (> 0.9),
        i.e. the cell is using essentially all of its current capacity.
      - Plateau: validation loss has stopped improving, so more capacity
        (rather than more training of the current capacity) is warranted.

    Growth fires only when both hold, and is proportional to current width.

    Args:
        cell: The TSRLSTMCell to inspect.
        val_loss_history: List of validation losses, most recent last.
        growth_rate: Fraction of current hidden size to add when growing.
        saturation_frac: Fraction of gates that must be > 0.9 to count as
            "at capacity".
        plateau_tol: Relative improvement below this counts as a plateau.
        max_hidden: Hard cap on hidden size.

    Returns:
        Number of units to add (0 if no growth is warranted).
    """
    if cell.hidden_size >= max_hidden:
        return 0

    # ── Saturation signal ──
    gate_vals = cell.gate_values()
    saturation = (gate_vals > 0.9).float().mean().item()
    if saturation < saturation_frac:
        return 0

    # ── Plateau signal ── need at least a few epochs to judge a trend.
    if len(val_loss_history) < 4:
        return 0
    prev = sum(val_loss_history[-4:-2]) / 2
    recent = sum(val_loss_history[-2:]) / 2
    if prev <= 0:
        return 0
    rel_improvement = (prev - recent) / abs(prev)
    if rel_improvement >= plateau_tol:
        return 0  # still improving — don't grow yet

    n = max(1, math.ceil(cell.hidden_size * growth_rate))
    return min(n, max_hidden - cell.hidden_size)


def train_eval_loop(model_name, model, train_loader, val_loader, test_loader, device, epochs=10, is_tsr=False):
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=0.001)

    print(f"\nTraining {model_name}...")
    best_val_loss = float('inf')
    test_mse = float('inf')

    history = []
    val_loss_history = []

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
        val_loss_history.append(val_loss)

        # TSR structural plasticity: grow the hidden state only when the cell's
        # OWN signals call for it (gates saturated + validation loss plateaued),
        # not on a fixed schedule. This makes the LSTM genuinely self-regulating.
        if is_tsr:
            n_grow = tsr_lstm_growth_signal(model.lstm.cell, val_loss_history)
            if n_grow > 0:
                print(f"--- TSRLSTM growth signal fired: +{n_grow} hidden units "
                      f"({model.lstm.cell.hidden_size} → {model.lstm.cell.hidden_size + n_grow}) ---")
                model.lstm.cell.grow_neurons(n_grow, init_scale=0.01)
                # Pair the downstream FC layer: new hidden units feed in as new
                # input columns (zero-init so output is unchanged at insertion).
                fc_weight = model.fc.weight.data
                new_cols = torch.zeros(model.fc.out_features, n_grow, device=device)
                model.fc.weight = nn.Parameter(torch.cat([fc_weight, new_cols], dim=1))
                model.fc.in_features += n_grow
                optimizer = Adam(model.parameters(), lr=0.001)  # re-init for new params

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
