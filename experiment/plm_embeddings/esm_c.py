import os

import numpy as np
import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

from utils import setup_torch_device, process_data_files


def esm_c_encoding(sequences, model_name="esmc_600m"):
    device = setup_torch_device()
    model = ESMC.from_pretrained(model_name).to(device)
    model.eval()

    config = LogitsConfig(
        sequence=True,
        return_embeddings=True,
    )

    embeddings = []

    with torch.no_grad():
        for sequence in sequences:
            protein = ESMProtein(sequence=sequence)
            protein_tensor = model.encode(protein)

            logits_output = model.logits(
                protein_tensor,
                config
            )

            sequence_embeddings = logits_output.embeddings.squeeze(0)
            pooled_embedding = sequence_embeddings.mean(dim=0).cpu().numpy()
            embeddings.append(pooled_embedding)

    return np.vstack(embeddings)


def main():
    embeddings_type = 'esm_c'

    data_path = '../data'
    files = [f.split('.')[0] for f in os.listdir(data_path)
             if os.path.isfile(os.path.join(data_path, f)) and f.endswith('.csv')]
    process_data_files(files, data_path, embeddings_type, esm_c_encoding)


if __name__ == "__main__":
    main()
