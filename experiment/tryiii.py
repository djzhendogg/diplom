import pandas as pd

from experiment.fgw import FusedGromovWassersteinComputer
from experiment.preset_tools_dc import data_processing

df = pd.read_csv('./antidiabetic.csv')
max_length = df['sequence'].str.len().max()
df_0 = df[df['label'] == 0]
df_1 = df[df['label'] == 1]
df_1['label'] = 0

def return_X(df, max_size = None):
    data = list(df['sequence'])
    narray = data_processing(data, max_size)
    return narray

X0 = return_X(df_0, max_length)
X1 = return_X(df_1, max_length)

fgw_computer = FusedGromovWassersteinComputer()
dist_fgw, T, log = fgw_computer.compute_fgw_distance(
    X0, X1,
    alpha=0.5,  # Равный вес признаков и структуры
    loss_fun='square_loss',
    verbose=True
)

# res = fgw_computer.compute_fgw_with_structure_search(
#     X0, X1
# )

print(f"FGW расстояние: {dist_fgw:.4f}")
print(f"Форма транспортной матрицы: {T.shape}")

