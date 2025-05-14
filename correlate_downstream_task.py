# correlate_downstream_task.py

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau
import os

def main(csv_file):
    # Load CSV
    df = pd.read_csv(csv_file)
    
    # Validate columns
    if not {'distance', 'task_performance'}.issubset(df.columns):
        raise ValueError("CSV file must contain 'distance' and 'task_performance' columns.")
    
    distances = df['distance'].values
    performances = df['task_performance'].values

    # Compute correlation coefficients and p-values
    pearson_r, pearson_p = pearsonr(distances, performances)
    spearman_r, spearman_p = spearmanr(distances, performances)
    kendall_r, kendall_p = kendalltau(distances, performances)

    print("Correlation Results:")
    print(f"Pearson  r = {pearson_r:.4f}, p = {pearson_p:.4e}")
    print(f"Spearman r = {spearman_r:.4f}, p = {spearman_p:.4e}")
    print(f"Kendall  r = {kendall_r:.4f}, p = {kendall_p:.4e}")

    # Plot
    plt.figure(figsize=(6, 4))
    plt.scatter(distances, performances, alpha=0.7)
    plt.xlabel("Distance Metric (e.g., FRD)")
    plt.ylabel("Downstream Task Performance (e.g., Dice)")
    plt.title("Distance vs. Task Performance")
    plt.grid(True)

    # Save plot
    output_dir = os.path.dirname(os.path.abspath(csv_file))
    plot_path = os.path.join(output_dir, "correlation_plot.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute correlation between distance metrics and downstream task performance.")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file containing 'distance' and 'task_performance' columns.")
    args = parser.parse_args()
    main(args.csv_file)
