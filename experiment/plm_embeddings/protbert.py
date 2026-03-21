import os

import numpy as np
import torch
from transformers import BertModel, BertTokenizer

from utils import setup_torch_device, process_data_files


def protbert_encoding(sequences, model_name=None, batch_size=32):
    # Setup device
    # device = setup_torch_device()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name is None:
        model_name = "Rostlab/prot_bert"

    # Load model and tokenizer
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
            attention_mask = encoded["attention_mask"].to(device)
            np.save('attention_mask.npy', attention_mask)

            # Forward pass
            outputs = model(**encoded)
            last_hidden = outputs.last_hidden_state.cpu()  # (batch_size, seq_len, hidden_dim)
            np.save('last_hidden.npy', last_hidden)
            # Mean pooling: skip special tokens [CLS] and [SEP] (positions 1:-1)
            # Note: This assumes [CLS] is first token and [SEP] is last token
            batch_embeddings = last_hidden[:, 1:-1, :].mean(dim=1).cpu().numpy()
            embeddings.append(batch_embeddings)

    return np.vstack(embeddings)


def main():
    embeddings_type = 'protbert'
    data_path = '../data'
    files = [f.split('.')[0] for f in os.listdir(data_path)
             if os.path.isfile(os.path.join(data_path, f)) and f.endswith('.csv')]
    process_data_files(files, data_path, embeddings_type, protbert_encoding)


if __name__ == "__main__":
    main()
