import ankh
import numpy as np
import torch
from utils import setup_torch_device, process_data_files

device = setup_torch_device()
print(device)
model, tokenizer = ankh.load_large_model()
model.eval()
model.to(device=device)


def ankh_encoding_with_attention_mask(sequences, batch_size=32):
    processed_sequences = [list(seq) for seq in sequences]
    result_embeddings = []
    with torch.no_grad():
        for i in range(0, len(processed_sequences), batch_size):
            batch_sequences = processed_sequences[i:i + batch_size]
            encoded = tokenizer(
                batch_sequences,
                add_special_tokens=True,
                padding=True,
                is_split_into_words=True,
                return_tensors="pt",
            )
            # print(encoded['attention_mask'])

            embeddings = model(input_ids=encoded['input_ids'].to(device), attention_mask=encoded['attention_mask'].to(device))[0].cpu()
            # print('out emb shape')
            # print(embeddings.shape)
            attention_mask = encoded["attention_mask"].cpu()

            mean_embeddings = []
            for seq_num in range(len(embeddings)):
                seq_len = (attention_mask[seq_num] == 1).sum()
                seq_emd = embeddings[seq_num][:seq_len - 1]
                seq_emd_mean = seq_emd.mean(dim=0)
                mean_embeddings.append(seq_emd_mean)

            result_embeddings.append(mean_embeddings)
    result_embeddings = np.vstack(result_embeddings)
    np.save('embeddings_mask.npy', result_embeddings)


def ankh_encoding_no_attention_mask(sequences, batch_size=32):
    processed_sequences = [list(seq) for seq in sequences]
    result_embeddings = []
    with torch.no_grad():
        for i in range(0, len(processed_sequences), batch_size):
            batch_sequences = processed_sequences[i:i + batch_size]
            encoded = tokenizer(
                batch_sequences,
                add_special_tokens=True,
                padding=True,
                is_split_into_words=True,
                return_tensors="pt",
            )
            # print(encoded['attention_mask'])

            embeddings = model(input_ids=encoded['input_ids'].to(device), attention_mask=encoded['attention_mask'].to(device))[0].cpu()
            # print('out emb shape')
            # print(embeddings.shape)

            mean_embeddings = []
            for seq_num in range(len(embeddings)):
                seq_emd = embeddings[seq_num]
                seq_emd_mean = seq_emd.mean(dim=0)
                mean_embeddings.append(seq_emd_mean)

            result_embeddings.append(mean_embeddings)
    result_embeddings = np.vstack(result_embeddings)
    np.save('embeddings_no_mask.npy', result_embeddings)


def main():
    protein_sequences = [
        'MKALCLLL',
        'GSHMS',
    ]
    ankh_encoding_with_attention_mask(protein_sequences)
    ankh_encoding_no_attention_mask(protein_sequences)
if __name__ == "__main__":
    main()
