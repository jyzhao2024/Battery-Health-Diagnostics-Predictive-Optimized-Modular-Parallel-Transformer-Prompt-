import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.font_manager import FontProperties
from matplotlib.cm import ScalarMappable  # 导入 ScalarMappable
from sklearn.ensemble import IsolationForest

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)



result = load_obj(f'./NCA')

test_1= ['SNL_18650_NCA_25C_0-100_0.5-2C_a', 'SNL_18650_NCA_25C_0-100_0.5-2C_b', 'SNL_18650_NCA_35C_0-100_0.5-1C_a', 'SNL_18650_NCA_35C_0-100_0.5-1C_b',
         'SNL_18650_NCA_35C_0-100_0.5-1C_c', 'SNL_18650_NCA_35C_0-100_0.5-1C_d', 'SNL_18650_NCA_35C_0-100_0.5-2C_a', 'SNL_18650_NCA_35C_0-100_0.5-2C_b']
channel_1 = ['25#6', '25#6', '35#1', '35#2', '35#3','35#4', '35#5', '35#6']

fig, ax = plt.subplots(figsize=(10, 7))
fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)


cmap = plt.cm.viridis
norm = mcolors.Normalize(vmin=0, vmax=800)


font_prop = FontProperties(family='Times New Roman', size=35)


for i, name in enumerate(test_1):
    A = load_obj(f'E:/Python/SNL(times)/{name}')[name]
    rul_true = result[name]['soh']['true']
    rul_pred = result[name]['soh']['transfer']
    b = A['rul'][1]

    cycle_life = np.array(list(A['rul'].values()))

    colors = cycle_life[:len(rul_true)]

    interval = 1

    ax.scatter(rul_true[::interval], rul_pred[::interval], c=colors[::interval],
               cmap=cmap, norm=norm, s=100, edgecolors='none')


ax.plot([2.45, 3.1], [2.45, 3.1], 'k--', label='y = x', linewidth=2)  # y=x 参考线

ax.set_xlim(2.45, 3.1)
ax.set_ylim(2.45, 3.1)

sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Cycle number', fontsize=35, family='Times New Roman')
cbar.ax.tick_params(labelsize=30)
for label in cbar.ax.get_yticklabels():
    label.set_fontproperties(FontProperties(family='Times New Roman', size=35))

plt.xlabel('Actual capacity (Ah)', fontsize=35, family='Times New Roman')
plt.ylabel('Estimated capacity (Ah)', fontsize=35, family='Times New Roman')

plt.xticks(fontsize=35, family='Times New Roman')
plt.yticks(fontsize=35, family='Times New Roman')

ax.grid(False)

plt.show()

fig2, ax2 = plt.subplots(figsize=(10, 7))
fig2.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)

colors = ['#2F4D92', '#C19796', '#C5B2CF', '#3E6534', '#81B333', '#E59646',
          '#47B9E5', '#BB232C']

error_list = []

outlier_detector = IsolationForest(contamination=0.5)

for i, name in enumerate(test_1):
    rul_true = result[name]['soh']['true']
    rul_pred = result[name]['soh']['transfer']
    error = (rul_true - rul_pred)


    error = error[(error >= -0.05) & (error <= 0.05)]


    error_reshaped = np.array(error).reshape(-1, 1)
    outliers = outlier_detector.fit_predict(error_reshaped)

    error_cleaned = [e for e, o in zip(error, outliers) if o == 1]

    error_list.append(error_cleaned)

boxplot = ax2.boxplot(error_list, labels=channel_1, vert=True, patch_artist=True)

print(f"Number of boxplots: {len(boxplot['boxes'])}")

for i, box in enumerate(boxplot['boxes']):
    if i < len(colors):
        color = colors[i % len(colors)]
        box.set_facecolor(color)
        box.set_edgecolor('black')
        box.set_linewidth(2)

for i, whisker in enumerate(boxplot['whiskers']):
    whisker.set_color('black')
    whisker.set_linewidth(2)

for i, cap in enumerate(boxplot['caps']):
    cap.set_color('black')
    cap.set_linewidth(2)

for i, median in enumerate(boxplot['medians']):
    median.set_color('black')
    median.set_linewidth(2)

for i, flier in enumerate(boxplot['fliers']):
    flier.set(markerfacecolor='#646464', marker='o', markersize=8, markeredgewidth=0.5)

ax2.set_xlabel('Sample', fontsize=35, family='Times New Roman')
ax2.set_ylabel('Error (Ah)', fontsize=35, family='Times New Roman')

ax2.tick_params(axis='x', labelsize=35, labelcolor='black', width=1)
ax2.tick_params(axis='y', labelsize=35, labelcolor='black', width=1)

for label in ax2.get_xticklabels():
    label.set_fontproperties(FontProperties(family='Times New Roman', size=34))
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)

ax2.set_yticks(np.arange(-0.04, 0.06, 0.02))
ax2.set_yticklabels([f'{x:.2f}' for x in np.arange(-0.04, 0.06, 0.02)], fontsize=35, family='Times New Roman')

plt.show()