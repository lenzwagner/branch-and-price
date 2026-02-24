import os
import glob
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def plot_latest_crossover_grid():
    """
    Finds the latest crossover result pickle file and plots the Pareto front
    for k_learn vs. crossover_theta.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    
    if not os.path.exists(results_dir):
        print(f"Directory not found: {results_dir}")
        return

    pickle_files = glob.glob(os.path.join(results_dir, '*.pkl'))
    if not pickle_files:
        print(f"Error: No pickle files found in '{results_dir}'.")
        return

    # Find the newest file
    newest_pkl = max(pickle_files, key=os.path.getmtime)
    print(f"Loading newest file: {newest_pkl}")

    with open(newest_pkl, 'rb') as f:
        data = pickle.load(f)

    if 'grid_results' not in data:
        print("Error: The latest result file does not contain a 'grid_results' 2D analysis.")
        return

    grid_results = data['grid_results']
    
    # Extract data for plotting
    k_learns = []
    thetas = []

    for res in grid_results:
        if res.get('crossover_theta') is not None:
            k_learns.append(res['k_learn'])
            thetas.append(res['crossover_theta'])

    if not k_learns:
        print("Error: No successful crossovers found in the grid results.")
        return

    # Create the plot
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(k_learns, thetas, marker='o', linestyle='-', linewidth=2, markersize=8, color="#0C7BDC")

    # Add labels and title
    ax.set_title("Pareto Front: Required Start-Efficiency (theta_base) vs. Learning Rate (k_learn)", fontsize=14, pad=15)
    ax.set_xlabel("Learning Speed (k_learn)", fontsize=12)
    ax.set_ylabel("Min. Required Crossover Threshold (theta_base)", fontsize=12)

    # Add text annotations for each point
    for k, t in zip(k_learns, thetas):
        ax.annotate(f"{t:.3f}", 
                    (k, t), 
                    textcoords="offset points", 
                    xytext=(0, 10), 
                    ha='center', 
                    fontsize=10,
                    weight='bold')

    # Highlight interpretation areas
    ax.grid(True, linestyle="--", alpha=0.7)
    
    # Save the plot alongside the pickle
    base_name = os.path.splitext(newest_pkl)[0]
    plot_path = f"{base_name}_grid_plot.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"✓ Plot saved to: {plot_path}")
    
    # Show it interactively (useful when run manually in PyCharm)
    plt.show()

if __name__ == "__main__":
    plot_latest_crossover_grid()

