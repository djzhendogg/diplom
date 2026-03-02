from experiment.dataloader_torch import create_dataset_from_batches
from experiment.preset_tools import oversampling
# from otdd.pytorch.datasets import load_torchvision_data
from otdd.pytorch.distance import DatasetDistance
import pandas as pd
import gc

monomer_dict = {
    'A': 'CC(N)C(=O)O', 'R': 'NC(N)=NCCCC(N)C(=O)O', 'N': 'NC(=O)CC(N)C(=O)O',
    'D': 'NC(CC(=O)O)C(=O)O', 'C': 'NC(CS)C(=O)O', 'Q': 'NC(=O)CCC(N)C(=O)O',
    'E': 'NC(CCC(=O)O)C(=O)O', 'G': 'NCC(=O)O', 'H': 'NC(Cc1cnc[nH]1)C(=O)O',
    'I': 'CCC(C)C(N)C(=O)O', 'L': 'CC(C)CC(N)C(=O)O', 'K': 'NCCCCC(N)C(=O)O',
    'M': 'CSCCC(N)C(=O)O', 'F': 'NC(Cc1ccccc1)C(=O)O', 'P': 'O=C(O)C1CCCN1',
    'S': 'NC(CO)C(=O)O', 'T': 'CC(O)C(N)C(=O)O', 'W': 'NC(Cc1c[nH]c2ccccc12)C(=O)O',
    'Y': 'NC(Cc1ccc(O)cc1)C(=O)O', 'V': 'CC(C)C(N)C(=O)O', 'O': 'CC1CC=NC1C(=O)NCCCCC(N)C(=O)O',
    'U': 'NC(C[Se])C(=O)O'
}

batch_size = 32
max_len = 100

df = pd.read_csv('./antibacterial.csv')
df_0 = df[df['label'] == 0][:4991]
df_1 = df[df['label'] == 1][:4991]
df_1['label'] = 0

def return_dataloader(df):
    data = list(zip(df['sequence'], df['label']))
    data = oversampling(sequences_with_labels=data, target_divisor=batch_size)
    batches_0 = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    dataloader = create_dataset_from_batches(batches=batches_0, monomer_dict=monomer_dict, max_len=max_len)
    return dataloader


dataloader_0 = return_dataloader(df_0)

dataloader_1 = return_dataloader(df_1)


gc.collect()
# Load data
# loaders_src  = load_torchvision_data('MNIST', valid_size=0, resize = 28, maxsize=2000)[0]
# loaders_tgt  = load_torchvision_data('FashionMNIST',  valid_size=0, resize = 28, maxsize=2000)[0]

# Instantiate distance
dist = DatasetDistance(dataloader_0, dataloader_1,
                        # method='augmentation',
# ignore_source_labels=True,
# ignore_target_labels=True,
                          inner_ot_method = 'gaussian_approx',
                          debiased_loss = True,
                          p = 2, entreg = 1e-1,
                          device='cpu')

d = dist.distance(maxsamples = 10000)
print(f'OTDD(MNIST,FashionMNIST)={d:8.2f}')
