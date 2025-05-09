import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.font_manager import FontProperties
from matplotlib.cm import ScalarMappable
from sklearn.ensemble import IsolationForest

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

result = load_obj(f'./result/70%')
test_1 = list(result.keys())

fig, ax = plt.subplots(figsize=(12, 7))
fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)

cmap = plt.cm.viridis
norm = mcolors.Normalize(vmin=0, vmax=600)

font_prop = FontProperties(family='Times New Roman', size=30)

for i, name in enumerate(test_1):
    A = load_obj(f'D:EV/data/{name}')[name]
    rul_true = result[name]['soh']['true']
    rul_pred = result[name]['soh']['transfer']
    b = A['rul']
    cycle_life = np.array(list(A['rul'].values()))
    colors = cycle_life[:len(rul_true)]
    interval = 1

    ax.scatter(rul_true[::interval], rul_pred[::interval], c=colors[::interval],
               cmap=cmap, norm=norm, s=100, edgecolors='none')

    error = rul_true - rul_pred
    ax.scatter(rul_true[::interval] / 1000, error[::interval], c=colors[::interval],
               cmap=cmap, norm=norm, s=100, edgecolors='none')

ax.plot([113, 155], [113, 155], 'k--', label='y = x', linewidth=2)
ax.set_xlim(113, 155)
ax.set_ylim(113, 155)

sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Charging segments', fontsize=35, family='Times New Roman')
cbar.ax.tick_params(labelsize=30)
for label in cbar.ax.get_yticklabels():
    label.set_fontproperties(FontProperties(family='Times New Roman', size=30))

plt.xlabel('Actual capacity (Ah)', fontsize=30, family='Times New Roman')
plt.ylabel('Estimated capacity (Ah)', fontsize=30, family='Times New Roman')
plt.xticks(fontsize=30, family='Times New Roman')
plt.yticks(fontsize=30, family='Times New Roman')
ax.grid(False)
plt.show()

simplified_vehicle_names = [name.split('_')[0] for name in test_1]

fig, ax = plt.subplots(figsize=(20, 7))
font_prop = FontProperties(family='Times New Roman', size=35)

errors_per_vehicle = []
outlier_detector = IsolationForest(contamination=0.2)

for i, name in enumerate(test_1):
    A = load_obj(f'D:EV/data/{name}')[name]
    rul_true = result[name]['soh']['true']
    rul_pred = result[name]['soh']['transfer']
    error = rul_true - rul_pred
    errors_per_vehicle.append(error)

ax.boxplot(errors_per_vehicle, vert=True, patch_artist=True)

ax.set_xlabel('Vehicles', fontsize=30, family='Times New Roman')
ax.set_ylabel('Error (Ah)', fontsize=30, family='Times New Roman')
ax.set_xticklabels(simplified_vehicle_names, fontsize=30, family='Times New Roman', rotation=90)
plt.xticks(fontsize=30, family='Times New Roman')
plt.yticks(fontsize=30, family='Times New Roman')
ax.grid(False)
plt.tight_layout()
plt.show()
