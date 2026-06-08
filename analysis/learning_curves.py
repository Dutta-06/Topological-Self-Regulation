import json
import matplotlib.pyplot as plt

def plot_time_to_accuracy(results_file: str, output_file: str):
    with open(results_file, 'r') as f:
        results = json.load(f)
        
    plt.figure(figsize=(10, 6))
    
    # Plot baseline
    baseline = results.get('baseline_vgg_medium', [])
    if baseline:
        b_flops = [step['cumulative_flops'] / 1e9 for step in baseline]  # GigaFLOPs
        b_acc = [step['accuracy'] * 100 for step in baseline]
        plt.plot(b_flops, b_acc, 'gray', linestyle='--', linewidth=2, label='Fixed Baseline (VGG Medium)')
        
    # Plot TSR
    tsr = results.get('tsr_adaptive', [])
    if tsr:
        t_flops = [step['cumulative_flops'] / 1e9 for step in tsr]
        t_acc = [step['accuracy'] * 100 for step in tsr]
        plt.plot(t_flops, t_acc, 'blue', linewidth=3, label='TSR (Adaptive Growth)')
        
    plt.xlabel('Cumulative Training Compute (GigaFLOPs)')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Benchmark 5: Time-to-Accuracy (Convergence Efficiency)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Time-to-Accuracy plot saved to {output_file}")

if __name__ == "__main__":
    plot_time_to_accuracy("results/time_accuracy_benchmark.json", "results/time_accuracy_curve.png")
