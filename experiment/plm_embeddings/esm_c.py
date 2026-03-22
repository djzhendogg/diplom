import os

import numpy as np
import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

from utils import setup_torch_device, process_data_files


def esm_c_encoding(sequences, model_name="esmc_600m"):
    """
    Encode protein sequences using ESMC model with mean pooling.

    Args:
        sequences: List of protein sequences to encode
        model_name: Name of the pre-trained ESMC model

    Returns:
        numpy.ndarray: Array of sequence embeddings (n_sequences, embedding_dim)
    """
    # device = setup_torch_device()
    model = ESMC.from_pretrained(model_name).to('cpu')
    model.eval()

    config_mean = LogitsConfig(
        sequence=True,
        return_embeddings=True,
        return_mean_embedding=True,
    )

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
            sequence_embeddings = logits_output.embeddings.cpu().numpy()
            np.save(f"sequence_embeddings_{sequence}.npy", sequence_embeddings)

            logits_output_mean = model.logits(
                protein_tensor,
                config_mean
            )

            sequence_embeddings_mean = logits_output_mean.embeddings.cpu().numpy()
            np.save(f"sequence_embeddings_mean_{sequence}.npy", sequence_embeddings_mean)

            # sequence_embeddings = logits_output.embeddings.squeeze(0)
            # pooled_embedding = sequence_embeddings.mean(dim=0).cpu().numpy()
            # embeddings.append(pooled_embedding)

    return sequence_embeddings_mean


# def main():
#     embeddings_type = 'esm_c'
#
#     data_path = '../data'
#     files = [os.path.join(data_path, f) for f in ['umami.csv', 'amp_gonzales.csv']]
#     # files = [os.path.join(data_path, f) for f in os.listdir(data_path)
#     #          if os.path.isfile(os.path.join(data_path, f)) and f.endswith('.csv')]
#     process_data_files(files, embeddings_type, esm_c_encoding)


if __name__ == "__main__":
    sequences = ['VVYPWTQRF', 'LDL']
    esm_c_encoding(sequences)
