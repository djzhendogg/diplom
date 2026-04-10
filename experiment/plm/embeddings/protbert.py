import os

import numpy as np
import torch
from transformers import BertModel, BertTokenizer

from utils import setup_torch_device, process_data_files


def protbert_encoding(sequences, model_name=None, batch_size=32):
    device = setup_torch_device()

    if model_name is None:
        model_name = "Rostlab/prot_bert"

    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
    model = BertModel.from_pretrained(model_name).to(device)
    model.eval()

    # requirement
    processed_sequences = [" ".join(list(seq)) for seq in sequences]

    embeddings = []

    with torch.no_grad():
        for i in range(0, len(processed_sequences), batch_size):
            batch_sequences = processed_sequences[i:i + batch_size]

            encoded = tokenizer(
                batch_sequences, return_tensors="pt", padding=True
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}

            # Forward pass
            outputs = model(**encoded)
            last_hidden = outputs.last_hidden_state.cpu()
            attention_mask = encoded["attention_mask"].cpu()

            mean_embeddings = []
            for seq_num in range(len(last_hidden)):
                seq_len = (attention_mask[seq_num] == 1).sum()
                seq_emd = last_hidden[seq_num][1:seq_len - 1]
                seq_emd_mean = seq_emd.mean(dim=0)
                mean_embeddings.append(seq_emd_mean)

            embeddings.append(mean_embeddings)

    return np.vstack(embeddings)


def main():
    embeddings_type = 'protbert'
    data_path = '../../data'
    files = [f.split('.')[0] for f in os.listdir(data_path)
             if os.path.isfile(os.path.join(data_path, f)) and f.endswith('.csv')]
    process_data_files(files, data_path, embeddings_type, protbert_encoding)


if __name__ == "__main__":
    main()
