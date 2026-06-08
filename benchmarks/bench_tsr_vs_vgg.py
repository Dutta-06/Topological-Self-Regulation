import os
import json
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.cifar import get_cifar10_loaders
from baselines.fixed_arch import get_baseline_models
from tsr.model import TSRNetwork
from training.trainer import TSRTrainer
from tsr.utils import count_parameters

def train_baseline(model_name: str, model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, epochs: int = 50):
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    
    print(f"\nTraining baseline: {model_name} (Params: {count_parameters(model):,})")
    
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()
            
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                pred = out.argmax(dim=1)
                correct += pred.eq(y).sum().item()
                total += y.size(0)
                
        acc = correct / max(total, 1)
        if acc > best_acc:
            best_acc = acc
        print(f"Epoch {epoch+1}/{epochs} - Acc: {acc:.4f} (Best: {best_acc:.4f})")
        
    return best_acc

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vgg-epochs", type=int, default=50)
    parser.add_argument("--tsr-epochs", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader, val_loader = get_cifar10_loaders(batch_size=128)
    
    results = {
        "baselines": {},
        "tsr": {
            "trajectory": [],
            "validation_results": {}
        }
    }
    
    # 1. Train Static VGG Baselines
    baselines = get_baseline_models(num_classes=10)
    for name, model in baselines.items():
        params = count_parameters(model)
        # Check if we already have it in a previous run to save immense time, but user requested full run
        best_acc = train_baseline(name, model, train_loader, val_loader, device, epochs=args.vgg_epochs)
        results["baselines"][name] = {
            "params": params,
            "accuracy": best_acc
        }
    
    # 2. Train TSR Adaptive
    print("\nTraining Adaptive TSR Model")
    tsr_model = TSRNetwork(in_channels=3, seed_channels=[8, 8], num_classes=10).to(device)
    
    config = {
        "training": {
            "max_epochs": args.tsr_epochs,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "eval_interval": len(train_loader) # eval every epoch
        },
        "regulation": {
            "monitor_window": 100,
            "update_interval": 200,
            "bottleneck_threshold": 0.1,
            "growth_rate": 0.2,
        }
    }
    
    trainer = TSRTrainer(tsr_model, train_loader, val_loader, config, device)
    tsr_metrics = trainer.train()
    
    # Extract trajectory
    for point in trainer.metrics_history:
        results["tsr"]["trajectory"].append({
            "step": point["step"],
            "params": point["params"],
            "accuracy": point["val_accuracy"]
        })
        
    # 3. Validation Logic
    print("\n" + "="*50)
    print("VALIDATION: Does TSR make smarter simpler architectures?")
    print("="*50)
    
    for vgg_name, vgg_data in results["baselines"].items():
        vgg_acc = vgg_data["accuracy"]
        vgg_params = vgg_data["params"]
        
        # Find first epoch where TSR beats VGG accuracy
        crossover_point = next((p for p in results["tsr"]["trajectory"] if p["accuracy"] > vgg_acc), None)
        
        if crossover_point:
            tsr_params = crossover_point["params"]
            success = tsr_params < vgg_params
            print(f"[{vgg_name}] VGG: {vgg_acc:.4f} acc, {vgg_params:,} params")
            print(f"  -> TSR beat accuracy ({crossover_point['accuracy']:.4f}) with {tsr_params:,} params")
            print(f"  -> Hypothesis Validated? {'YES' if success else 'NO'}")
            
            results["tsr"]["validation_results"][vgg_name] = {
                "tsr_beat_acc": True,
                "tsr_params_at_crossover": tsr_params,
                "vgg_params": vgg_params,
                "hypothesis_validated": success
            }
        else:
            print(f"[{vgg_name}] VGG: {vgg_acc:.4f} acc, {vgg_params:,} params")
            print(f"  -> TSR did not beat this accuracy within {args.tsr_epochs} epochs.")
            results["tsr"]["validation_results"][vgg_name] = {
                "tsr_beat_acc": False,
                "hypothesis_validated": False
            }
    
    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/tsr_vs_vgg_comprehensive.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # Optional: Plotting
    try:
        plt.figure(figsize=(10, 6))
        # Plot TSR trajectory
        tsr_params = [p["params"] for p in results["tsr"]["trajectory"]]
        tsr_acc = [p["accuracy"] for p in results["tsr"]["trajectory"]]
        plt.plot(tsr_params, tsr_acc, 'bo-', label="TSR Trajectory", alpha=0.7)
        
        # Plot VGG baselines
        vgg_p = []
        vgg_a = []
        for name, data in results["baselines"].items():
            plt.scatter(data["params"], data["accuracy"], marker='x', s=100, label=name)
            
        plt.xlabel("Parameters")
        plt.ylabel("Validation Accuracy")
        plt.xscale("log")
        plt.title("TSR vs Static VGG Benchmark")
        plt.legend()
        plt.grid(True)
        plt.savefig("results/tsr_vs_vgg_plot.png")
        print("\nSaved plot to results/tsr_vs_vgg_plot.png")
    except Exception as e:
        print(f"Could not generate plot: {e}")

if __name__ == "__main__":
    main()
