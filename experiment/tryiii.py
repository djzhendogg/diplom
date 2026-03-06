import pandas as pd

from experiment.fgw import FusedGromovWassersteinComputer
from experiment.preset_tools_difflen import encode_seqs

df = pd.read_csv('antibacterial.csv')
max_length = df['sequence'].str.len().max()
df_0 = df[df['label'] == 0]
df_1 = df[df['label'] == 1]
df_1['label'] = 0

X0 = encode_seqs(df_0['sequence'].to_list())
X1 = encode_seqs(df_1['sequence'].to_list())

fgw_computer = FusedGromovWassersteinComputer()
dist_fgw, log = fgw_computer.compute_fgw_distance(
    X0, X1,
    alpha=0.5,
    loss_fun='square_loss',
    verbose=True
)

# res = fgw_computer.compute_fgw_with_structure_search(
#     X0, X1
# )

print(f"FGW расстояние: {dist_fgw:.4f}")
