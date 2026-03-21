import os
import re

import numpy as np
import torch
from transformers import T5Tokenizer, T5EncoderModel

from utils import setup_torch_device, process_data_files


def prott5_encoding(sequences, model_name=None, batch_size=16):
    # device = setup_torch_device()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            batch_seqs = processed_sequences[i:i + batch_size]

            # Токенизация с добавлением паддинга до длины самой длинной последовательности в батче
            encoded = tokenizer(batch_seqs, add_special_tokens=True, return_tensors="pt", padding="longest")
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            print("="*30)
            print("ATTENTION MASK")
            print(attention_mask)
            print("=" * 30)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state  # shape (batch_size, seq_len, hidden_dim)
            print("=" * 30)
            print("lAST HIDDEN")
            print(last_hidden)
            print("=" * 30)
            print("=" * 30)
            print("JUST MEAN lAST HIDDEN")
            print(last_hidden.mean(dim=1))
            print("=" * 30)
            # Усредняем по длине последовательности для каждого белка
            attention_mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = (last_hidden * attention_mask_expanded).sum(dim=1)
            mean_embeddings = sum_embeddings / attention_mask_expanded.sum(dim=1)
            print("=" * 30)
            print("MASKED MEAN lAST HIDDEN")
            print(mean_embeddings)
            print("=" * 30)
            embeddings.append(mean_embeddings)

    return np.vstack(embeddings)


def main():
    embeddings_type = 'prott5'

    data_path = '../data'
    files = [os.path.join(data_path, f) for f in ['umami.csv', 'amp_gonzales.csv']]
    # files = [os.path.join(data_path, f) for f in os.listdir(data_path)
    #          if os.path.isfile(os.path.join(data_path, f)) and f.endswith('.csv')]
    process_data_files(files, embeddings_type, prott5_encoding)


if __name__ == "__main__":
    main()
