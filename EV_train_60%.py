import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

result = load_obj('./result/60%')
test_1 = list(result.keys())

fig, ax = plt.subplots(figsize=(8, 6))
fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
font_prop = FontProperties(family='Times New Roman', size=22)

min_size = 50
max_size = 150

for name in test_1:
    A = load_obj(f'D:EV/data/{name}')[name]
    rul_true = result[name]['soh']['true']
    rul_pred = result[name]['soh']['transfer']

    rul_dict = A['rul']
    sorted_keys = sorted(rul_dict.keys())
    cycle_life = np.array([rul_dict[k] for k in sorted_keys])[:len(rul_true)]
    cycle_life_scaled = (cycle_life - cycle_life.min()) / (cycle_life.max() - cycle_life.min() + 1e-8)
    sizes = min_size + cycle_life_scaled * (max_size - min_size)

    interval = 1

    ax.scatter(rul_true[::interval], rul_pred[::interval],
               s=sizes[::interval], color='#24A6BB', alpha=0.9, edgecolors='none')

ax.plot([113, 155], [113, 155], color='#DA7479', linestyle='--', linewidth=2.5)

ax.set_xlim(113, 155)
ax.set_ylim(113, 155)
plt.xlabel('Actual capacity (Ah)', fontsize=22, family='Times New Roman')
plt.ylabel('Estimated capacity (Ah)', fontsize=22, family='Times New Roman')
plt.xticks(fontsize=22, family='Times New Roman')
plt.yticks(fontsize=22, family='Times New Roman')
ax.grid(False)

plt.show()
