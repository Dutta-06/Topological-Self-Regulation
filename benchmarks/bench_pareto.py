import os
import json
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.cifar import get_cifar10_loaders
from baselines.fixed_arch import get_baseline_models
from tsr.model import TSRNetwork
from tsr.flops import compute_model_flops
from training.trainer import TSRTrainer
from tsr.utils import count_parameters

def train_baseline(model_name: str, model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, epochs: int = 10):
    """Simple training loop for static baselines."""
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    
    print(f"\nTraining baseline: {model_name} (Params: {count_parameters(model):,})")
    
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Eval
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Eval]", leave=False):
                x, y = x.to(device), y.to(device)
                out = model(x)
                pred = out.argmax(dim=1)
                correct += pred.eq(y).sum().item()
                total += y.size(0)
                
        acc = correct / max(total, 1)
        if acc > best_acc:
            best_acc = acc
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f} - Acc: {acc:.4f} (Best: {best_acc:.4f})")
        
    return best_acc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Data
    train_loader, val_loader = get_cifar10_loaders(batch_size=128)
    
    results = {}
    epochs = 2  # Short for demonstration; use 100+ for paper
    
    # 2. Load Existing Baselines (to save time)
    if os.path.exists("results/pareto_benchmark.json"):
        with open("results/pareto_benchmark.json", "r") as f:
            old_results = json.load(f)
            for k, v in old_results.items():
                if k != "tsr_adaptive":
                    results[k] = v
        
    # 3. Train TSR (Adaptive)
    print("\nTraining Adaptive TSR Model")
    tsr_model = TSRNetwork(in_channels=3, seed_channels=[8, 8], num_classes=10).to(device)
    
    config = {
        "training": {
            "max_epochs": epochs,
            "learning_rate": 0.001,
            "optimizer": "adam",
        },
        "regulation": {
            "monitor_window": 100,
            "update_interval": 200,
            "bottleneck_threshold": 0.1,  # use the fixed threshold
            "growth_rate": 0.2,
        }
    }
    
    trainer = TSRTrainer(tsr_model, train_loader, val_loader, config, device)
    tsr_metrics = trainer.train()
    
    results["tsr_adaptive"] = {
        "accuracy": tsr_metrics["best_val_accuracy"],
        "inference_flops": tsr_metrics["inference_flops"],
        "params": tsr_metrics["effective_params"],
        "type": "adaptive",
        "final_topology": tsr_metrics["final_topology"]
    }
    
    # 4. Save Results
    os.makedirs("results", exist_ok=True)
    with open("results/pareto_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\nBenchmark Complete! Results saved to results/pareto_benchmark.json")
    for k, v in results.items():
        print(f"{k}: Acc={v['accuracy']:.4f}, FLOPs={v['inference_flops']:,}")

if __name__ == "__main__":
    main()
