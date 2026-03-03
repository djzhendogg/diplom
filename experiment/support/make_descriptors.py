import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.preprocessing import StandardScaler


def make_monomer_descriptors(monomer_dict: dict[str, str]) -> pd.DataFrame:
    descriptor_names = list(rdMolDescriptors.Properties.GetAvailableProperties())
    get_descriptors = rdMolDescriptors.Properties(descriptor_names)
    num_descriptors = len(descriptor_names)

    descriptors_set = np.empty((0, num_descriptors), float)

    for _, value in monomer_dict.items():
        molecule = Chem.MolFromSmiles(value)
        descriptors = np.array(get_descriptors.ComputeProperties(molecule)).reshape((-1, num_descriptors))
        descriptors_set = np.append(descriptors_set, descriptors, axis=0)

    sc = StandardScaler()
    scaled_array = sc.fit_transform(descriptors_set)
    descriptors_set = pd.DataFrame(scaled_array, columns=descriptor_names, index=monomer_dict.keys())

    energy_data = pd.read_csv('energy_data.csv')
    energy_set = energy_data.set_index("Aminoacid").iloc[:, :]

    energy_names = energy_set.columns

    scaled_energy = sc.fit_transform(energy_set)
    scaled_energy_set = pd.DataFrame(scaled_energy, columns=energy_names, index=monomer_dict.keys())

    all_descriptors = pd.concat([descriptors_set, scaled_energy_set], axis=1)
    return all_descriptors


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

a = make_monomer_descriptors(monomer_dict)
a.to_csv('aa_descriptors_scaled.csv')
print(a)
