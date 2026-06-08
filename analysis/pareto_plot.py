import json
import matplotlib.pyplot as plt
import numpy as np

def plot_pareto_frontier(results_file: str, output_file: str):
    with open(results_file, 'r') as f:
        results = json.load(f)
        
    static_flops = []
    static_accs = []
    static_labels = []
    
    tsr_flops = None
    tsr_acc = None
    
    for name, data in results.items():
        if data['type'] == 'static':
            static_flops.append(data['inference_flops'] / 1e6)  # MegaFLOPs
            static_accs.append(data['accuracy'] * 100)
            static_labels.append(name)
        elif data['type'] == 'adaptive':
            tsr_flops = data['inference_flops'] / 1e6
            tsr_acc = data['accuracy'] * 100
            
    plt.figure(figsize=(10, 6))
    
    # Plot static baselines
    plt.scatter(static_flops, static_accs, c='gray', marker='o', s=100, label='Fixed Architecture (VGG)')
    for i, label in enumerate(static_labels):
        plt.annotate(label, (static_flops[i], static_accs[i]), xytext=(5, 5), textcoords='offset points')
        
    # Draw Pareto frontier line for static
    # Sort by FLOPs
    sorted_indices = np.argsort(static_flops)
    sf_sorted = np.array(static_flops)[sorted_indices]
    sa_sorted = np.array(static_accs)[sorted_indices]
    
    pareto_front_flops = [sf_sorted[0]]
    pareto_front_acc = [sa_sorted[0]]
    
    for i in range(1, len(sf_sorted)):
        if sa_sorted[i] >= pareto_front_acc[-1]:
            pareto_front_flops.append(sf_sorted[i])
            pareto_front_acc.append(sa_sorted[i])
            
    plt.plot(pareto_front_flops, pareto_front_acc, 'k--', alpha=0.5, label='Static Pareto Frontier')
    
    # Plot TSR
    if tsr_flops is not None:
        plt.scatter(tsr_flops, tsr_acc, c='blue', marker='*', s=250, label='TSR (Adaptive)')
        plt.annotate("TSR", (tsr_flops, tsr_acc), xytext=(5, 5), textcoords='offset points', color='blue', fontweight='bold')
        
    plt.xlabel('Inference Compute (MegaFLOPs)')
    plt.ylabel('Accuracy (%)')
    plt.title('Benchmark 1: Accuracy vs. Inference Cost (Pareto Frontier)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Pareto plot saved to {output_file}")

if __name__ == "__main__":
    plot_pareto_frontier("results/pareto_benchmark.json", "results/pareto_frontier.png")
