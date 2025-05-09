import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd
import pickle
import warnings

warnings.filterwarnings("ignore")

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

result = load_obj(f'./result/70%')
test_1 = list(result.keys())
simplified_vehicle_names = [name.split('_')[0] for name in test_1]

third = len(simplified_vehicle_names) // 3
first_third = simplified_vehicle_names[:third]
second_third = simplified_vehicle_names[third:2*third]
third_third = simplified_vehicle_names[2*third:]

fig, ax = plt.subplots(figsize=(10, 7))
font_prop = FontProperties(family='Times New Roman', size=35)

errors_per_vehicle_first_third = []
for i, name in enumerate(test_1):
    A = load_obj(f'D:EV/data/{name}')[name]
    rul_true = result[name]['soh']['true']
    rul_pred = result[name]['soh']['transfer']
    error = rul_true - rul_pred
    errors_per_vehicle_first_third.append(error)

error_data_first_third = []
for i, name in enumerate(first_third):
    for error in errors_per_vehicle_first_third[i]:
        error_data_first_third.append([simplified_vehicle_names[i], error])

error_df_first_third = pd.DataFrame(error_data_first_third, columns=['Vehicle', 'Error'])
sns.violinplot(x='Vehicle', y='Error', data=error_df_first_third, inner="box", palette="muted", ax=ax)

ax.set_xlabel('Vehicles', fontsize=30, family='Times New Roman')
ax.set_ylabel('Error (Ah)', fontsize=30, family='Times New Roman')
ax.set_xticklabels(first_third, fontsize=20, family='Times New Roman', rotation=90)
plt.xticks(fontsize=30, family='Times New Roman')
plt.yticks(fontsize=30, family='Times New Roman')
ax.grid(False)
sns.set_palette("Set2")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 7))

errors_per_vehicle_second_third = []
for i, name in enumerate(test_1):
    A = load_obj(f'D:EV/data/{name}')[name]
    rul_true = result[name]['soh']['true']
    rul_pred = result[name]['soh']['transfer']
    error = rul_true - rul_pred
    errors_per_vehicle_second_third.append(error)

error_data_second_third = []
for i, name in enumerate(second_third):
    for error in errors_per_vehicle_second_third[i]:
        error_data_second_third.append([simplified_vehicle_names[i + third], error])

error_df_second_third = pd.DataFrame(error_data_second_third, columns=['Vehicle', 'Error'])
sns.violinplot(x='Vehicle', y='Error', data=error_df_second_third, inner="box", palette="muted", ax=ax)

ax.set_xlabel('Vehicles', fontsize=30, family='Times New Roman')
ax.set_ylabel('Error (Ah)', fontsize=30, family='Times New Roman')
ax.set_xticklabels(second_third, fontsize=20, family='Times New Roman', rotation=90)
plt.xticks(fontsize=30, family='Times New Roman')
plt.yticks(fontsize=30, family='Times New Roman')
ax.grid(False)
sns.set_palette("Set2")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 7))

errors_per_vehicle_third_third = []
for i, name in enumerate(test_1):
    A = load_obj(f'D:EV/data/{name}')[name]
    rul_true = result[name]['soh']['true']
    rul_pred = result[name]['soh']['transfer']
    error = rul_true - rul_pred
    errors_per_vehicle_third_third.append(error)

error_data_third_third = []
for i, name in enumerate(third_third):
    for error in errors_per_vehicle_third_third[i]:
        error_data_third_third.append([simplified_vehicle_names[i + 2*third], error])

error_df_third_third = pd.DataFrame(error_data_third_third, columns=['Vehicle', 'Error'])
sns.violinplot(x='Vehicle', y='Error', data=error_df_third_third, inner="box", palette="muted", ax=ax)

ax.set_xlabel('Vehicles', fontsize=30, family='Times New Roman')
ax.set_ylabel('Error (Ah)', fontsize=30, family='Times New Roman')
ax.set_xticklabels(third_third, fontsize=20, family='Times New Roman', rotation=90)
plt.xticks(fontsize=30, family='Times New Roman')
plt.yticks(fontsize=30, family='Times New Roman')
ax.grid(False)
sns.set_palette("Set2")
plt.tight_layout()
plt.show()
