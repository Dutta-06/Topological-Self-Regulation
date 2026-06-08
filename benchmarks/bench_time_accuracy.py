import os
import json
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.cifar import get_cifar10_loaders
from baselines.fixed_arch import FixedVGG
from tsr.model import TSRNetwork
from tsr.flops import compute_model_flops
from training.trainer import TSRTrainer
from tsr.utils import count_parameters

def train_baseline_with_tracking(model_name: str, model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, epochs: int = 10, eval_interval: int = 200):
    """Training loop for static baseline that tracks accuracy vs. cumulative FLOPs over time."""
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    
    print(f"\nTraining baseline for Time-to-Accuracy: {model_name}")
    
    history = []
    cumulative_flops = 0
    step = 0
    
    # 1 Forward + 1 Backward FLOPs per sample approx = 3 * forward_flops
    forward_flops_per_sample = compute_model_flops(model, (3, 32, 32))
    flops_per_step = forward_flops_per_sample * 3 * train_loader.batch_size
    
    for epoch in range(epochs):
        model.train()
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()
            
            cumulative_flops += flops_per_step
            step += 1
            
            if step % eval_interval == 0:
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for vx, vy in val_loader:
                        vx, vy = vx.to(device), vy.to(device)
                        out = model(vx)
                        pred = out.argmax(dim=1)
                        correct += pred.eq(vy).sum().item()
                        total += vy.size(0)
                
                acc = correct / max(total, 1)
                history.append({
                    "step": step,
                    "cumulative_flops": cumulative_flops,
                    "accuracy": acc
                })
                model.train()
                
    return history

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader, val_loader = get_cifar10_loaders(batch_size=128)
    epochs = 5  # Short for demonstration
    eval_interval = 100
    
    # Target Architecture (what TSR will hopefully grow to match or beat)
    target_model = FixedVGG([32, 32, 64], num_classes=10)
    baseline_history = train_baseline_with_tracking(
        "vgg_medium", target_model, train_loader, val_loader, device, epochs, eval_interval
    )
    
    # TSR Architecture (starts tiny)
    print("\nTraining Adaptive TSR Model")
    tsr_model = TSRNetwork(in_channels=3, seed_channels=[8, 8], num_classes=10).to(device)
    
    config = {
        "training": {
            "max_epochs": epochs,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "eval_interval": eval_interval,
        },
        "regulation": {
            "monitor_window": 50,
            "update_interval": 100,
            "bottleneck_threshold": 0.1,
            "growth_rate": 0.2,
        }
    }
    
    trainer = TSRTrainer(tsr_model, train_loader, val_loader, config, device)
    trainer.train()
    
    tsr_history = []
    for h in trainer.metrics_history:
        tsr_history.append({
            "step": h["step"],
            "cumulative_flops": h["cumulative_flops"],
            "accuracy": h["val_accuracy"]
        })
        
    results = {
        "baseline_vgg_medium": baseline_history,
        "tsr_adaptive": tsr_history
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/time_accuracy_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\nBenchmark Complete! Results saved to results/time_accuracy_benchmark.json")

if __name__ == "__main__":
    main()
