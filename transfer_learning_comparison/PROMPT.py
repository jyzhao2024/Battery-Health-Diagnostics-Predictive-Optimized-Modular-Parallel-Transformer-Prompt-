import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

plt.rcParams['font.family'] = 'Times New Roman'

def load_res_dict(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def plot_scatter_all_batteries(res_dict, font_size=30, tick_size=30, stride=25):
    actual_vals = []
    estimated_vals = []
    cycle_numbers = []

    for battery_name, battery_data in res_dict.items():
        try:
            y_true = np.array(battery_data['soh']['true']).flatten()
            y_pred = np.array(battery_data['soh']['transfer']).flatten()
            cycles = np.arange(len(y_true))

            actual_vals.extend(y_true[::stride])
            estimated_vals.extend(y_pred[::stride])
            cycle_numbers.extend(cycles[::stride])

        except Exception as e:
            print(f" {battery_name} error: {e}")

    actual_vals = np.array(actual_vals)
    estimated_vals = np.array(estimated_vals)
    cycle_numbers = np.array(cycle_numbers)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(actual_vals, estimated_vals, c=cycle_numbers, cmap='viridis', s=100, edgecolor='none')

    lims = [min(actual_vals.min(), estimated_vals.min()) - 0.02,
            max(actual_vals.max(), estimated_vals.max()) + 0.02]
    ax.plot(lims, lims, '--', color='black', linewidth=2)

    ax.set_xlabel('Actual capacity (Ah)', fontsize=font_size)
    ax.set_ylabel('Estimated capacity (Ah)', fontsize=font_size)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.tick_params(axis='both', labelsize=tick_size)

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Cycle number', fontsize=font_size)
    cbar.ax.tick_params(labelsize=tick_size)

    plt.grid(False)
    plt.tight_layout()
    plt.show()

def main():
    res_dict = load_res_dict('./result/PROMPT_results.pkl')
    plot_scatter_all_batteries(res_dict, font_size=22, tick_size=20, stride=25)

main()
