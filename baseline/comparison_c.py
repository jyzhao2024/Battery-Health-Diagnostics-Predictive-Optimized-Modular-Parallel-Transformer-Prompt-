import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import cm

def load_res_dict(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def plot_comparison(true_vals_dicts, transfer_vals_dicts, battery_names, max_cycle_life, model_names, interval=35,
                    title_size=26, label_size=35, tick_size=30, legend_size=22):
    plt.rcParams['font.family'] = 'Times New Roman'

    fig, ax = plt.subplots(figsize=(10, 7))

    markers = ['o', 'o', 'o', 'o', 'o']
    colors = ['#529DC2', '#FDA279', '#97D1B0', '#7F8BB7', '#DE8488']
    edge_colors = ['#115E85','#ED6E35', '#5EB083', '#424E7C', '#C44046']

    for model_idx, model_name in enumerate(model_names):
        true_vals_dict = true_vals_dicts[model_idx]
        transfer_vals_dict = transfer_vals_dicts[model_idx]

        label_shown = False

        for battery_name in battery_names:
            if battery_name in true_vals_dict:
                soh_true = true_vals_dict[battery_name]
                soh_transfer = transfer_vals_dict[battery_name]

                indices = np.arange(0, len(soh_true), interval)
                soh_true_sampled = np.array(soh_true)[indices]
                soh_transfer_sampled = np.array(soh_transfer)[indices]

                ax.scatter(soh_true_sampled, soh_transfer_sampled,
                           s=100, alpha=1,
                           label=model_name if not label_shown else None,
                           marker=markers[model_idx % len(markers)], color=colors[model_idx % len(colors)],
                           edgecolor='None')
                label_shown = True

    ax.plot([0.89, 1.12], [0.89, 1.12], color='black', linestyle='--', linewidth=2)

    ax.set_xlabel('Actual capacity (Ah)', fontsize=label_size)
    ax.set_ylabel('Estimated capacity (Ah)', fontsize=label_size)
    ax.set_xlim(0.89, 1.12)
    ax.set_ylim(0.89, 1.12)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    ax.legend(fontsize=legend_size, loc='upper left', )

    plt.tight_layout()
    plt.show()

def main():
    model_files = ['./result-2/CNN.pkl', './result-2/SVR.pkl', './result-2/LSTM.pkl',
                   './result-2/T.pkl', './result-2/prompt.pkl']
    model_names = ['CNN', 'SVR', 'LSTM', 'Transformer', 'Proposed model']

    selected_batteries = ['a21', 'a23', 'a24', 'a25', 'a26', 'a27']

    true_vals_dicts = []
    transfer_vals_dicts = []
    max_cycle_life = 0

    for model_idx, file_path in enumerate(model_files):
        res_dict = load_res_dict(file_path)

        true_vals_dict = {}
        transfer_vals_dict = {}

        for battery_name in selected_batteries:
            try:
                soh_true = res_dict[battery_name]['soh']['true']

                if model_names[model_idx] == 'Proposed model':
                    soh_transfer = res_dict[battery_name]['soh']['transfer']
                elif model_names[model_idx] == 'SVR':
                    soh_transfer = res_dict[battery_name]['soh']['pred']
                else:
                    soh_transfer = res_dict[battery_name]['soh']['base']

                true_vals_dict[battery_name] = soh_true
                transfer_vals_dict[battery_name] = soh_transfer

                max_cycle_life = max(max_cycle_life, len(soh_true))

            except Exception as e:
                print(f" {battery_name} error: {e}")

        true_vals_dicts.append(true_vals_dict)
        transfer_vals_dicts.append(transfer_vals_dict)


    plot_comparison(true_vals_dicts, transfer_vals_dicts, selected_batteries, max_cycle_life, model_names, interval=35)

main()
