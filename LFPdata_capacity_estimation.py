import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import t
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import norm
import seaborn as sns

def load_res_dict(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def calculate_confidence_interval_z(y_pred, y_true, confidence=0.95):
    residuals = y_true - y_pred
    n = len(y_pred)
    if n < 2:
        return y_pred, y_pred
    mean_se = np.std(residuals, ddof=1) / np.sqrt(n)
    z_value = norm.ppf((1 + confidence) / 2)
    margin_error = z_value * mean_se
    ci_upper = y_pred + margin_error
    ci_lower = y_pred - margin_error
    return ci_lower, ci_upper

def plot_soh_estimation(true_vals, transfer_vals, battery_name, title_size=26, label_size=35, tick_size=35,
                        legend_size=30):
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, ax = plt.subplots(figsize=(10, 8))
    x = np.arange(len(true_vals))
    x_new = np.linspace(x.min(), x.max(), 800)
    spl_true = make_interp_spline(x, true_vals, k=3)
    spl_transfer = make_interp_spline(x, transfer_vals, k=3)
    smoothed_true_vals = spl_true(x_new)
    smoothed_transfer_vals = spl_transfer(x_new)
    ax.plot(x_new, smoothed_true_vals, label="Actual", color="#c381a8", linewidth=3.5)
    ax.plot(x_new, smoothed_transfer_vals, label="Estimated", color="#407bae", linewidth=3.5)
    residuals = transfer_vals - true_vals
    ci_lower, ci_upper = calculate_confidence_interval_z(transfer_vals, residuals)
    ax.fill_between(np.arange(len(transfer_vals)), ci_lower, ci_upper, color="#7CBEAE", alpha=0.6, label="Uncertainty")
    ax.set_xlabel("Cycle number", fontsize=label_size)
    ax.set_ylabel("Discharge capacity (Ah)", fontsize=label_size)
    ax.text(0.02, 0.98, f'{battery_name}', transform=ax.transAxes, fontsize=35, verticalalignment='top', horizontalalignment='left')
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
    ax.set_ylim(0.88, 1.12)
    ax.set_yticks(np.arange(0.9, 1.15, 0.05))
    legend = ax.legend(loc='upper right', fontsize=legend_size)
    for text in legend.get_texts():
        text.set_fontname('Times New Roman')
    add_density_plot(ax, true_vals, transfer_vals)
    plt.show()

def add_density_plot(ax, true_vals, pred_vals):
    inset_ax = ax.inset_axes([0.14, 0.16, 0.45, 0.45])
    sns.kdeplot(true_vals, ax=inset_ax, label="Actual", color='#c381a8', shade=True, alpha=0.7, linewidth=1.5)
    sns.kdeplot(pred_vals, ax=inset_ax, label="Estimated", color='#407bae', shade=True, alpha=0.5, linewidth=1.5)
    inset_ax.set_xlabel("Discharge capacity (Ah)", fontsize=28)
    inset_ax.set_ylabel("Density", fontsize=28)
    inset_ax.tick_params(axis='both', which='major', labelsize=26)
    inset_ax.legend(fontsize=23, loc='upper left')

def main():
    res_dict = load_res_dict('.F:/result/LFP.pkl')
    error_batteries = []
    for battery_name, battery_data in res_dict.items():
        try:
            soh_true = battery_data['soh']['true']
            soh_transfer = battery_data['soh']['transfer']
            plot_soh_estimation(soh_true, soh_transfer, battery_name, title_size=24, label_size=35, tick_size=35,
                                legend_size=26)
        except Exception as e:
            print(f" {battery_name} error: {e}")
            error_batteries.append(battery_name)
    if error_batteries:
        print(f"error: {error_batteries}")

main()
