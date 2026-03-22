import os
import re

import numpy as np
import torch
from transformers import T5Tokenizer, T5EncoderModel

from utils import setup_torch_device, process_data_files


def prott5_encoding(sequences, model_name=None, batch_size=16):
    device = setup_torch_device()

    if model_name is None:
        model_name = "Rostlab/prot_t5_xl_half_uniref50-enc"

    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_name).to(device)
    model.eval()

    # requirement
    processed_sequences = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in sequences]

    embeddings = []

    with torch.no_grad():
        for i in range(0, len(processed_sequences), batch_size):
            batch_sequences = processed_sequences[i:i + batch_size]

            encoded = tokenizer(batch_sequences, add_special_tokens=True, return_tensors="pt", padding="longest")
            encoded = {k: v.to(device) for k, v in encoded.items()}

            outputs = model(**encoded)
            last_hidden = outputs.last_hidden_state.cpu()  # shape (batch_size, seq_len, hidden_dim)
            attention_mask = encoded["attention_mask"].cpu()

            mean_embeddings = []
            for seq_num in range(len(last_hidden)):
                seq_len = (attention_mask[seq_num] == 1).sum()
                seq_emd = last_hidden[seq_num][:seq_len - 1]
                seq_emd_mean = seq_emd.mean(dim=0)
                mean_embeddings.append(seq_emd_mean)

            embeddings.append(mean_embeddings)

    return np.vstack(embeddings)


def main():
    embeddings_type = 'prott5'
    data_path = '../data'
    files = [f.split('.')[0] for f in os.listdir(data_path)
             if os.path.isfile(os.path.join(data_path, f)) and f.endswith('.csv')]
    process_data_files(files, data_path, embeddings_type, prott5_encoding)


if __name__ == "__main__":
    main()
