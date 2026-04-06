import ankh
import torch
from utils import setup_torch_device, process_data_files

device = setup_torch_device()
print(device)
model, tokenizer = ankh.load_large_model()
model.eval()
model.to(device=device)

def embed_dataset(sequences, shift_left = 0, shift_right = -1):
    # inputs_embedding = []
    with torch.no_grad():
        for sample in sequences:
            ids = tokenizer.batch_encode_plus([sample], add_special_tokens=True,
                                              padding=True, is_split_into_words=True,
                                              return_tensors="pt")
            embedding = model(input_ids=ids['input_ids'].to(device))[0]
            print(embedding.shape)
            embedding = embedding[0].detach().cpu().numpy()[shift_left:shift_right]
            print(embedding.shape)

            # inputs_embedding.append(embedding)
    #         inputs_embedding.append(embedding)
    # return inputs_embedding
def ankh_encoding(sequences, batch_size=32):
    protein_sequences = [list(seq) for seq in sequences]

    encoded = tokenizer(
        protein_sequences,
        add_special_tokens=True,
        padding=True,
        is_split_into_words=True,
        return_tensors="pt",
    )
    print(encoded['attention_mask'])
    with torch.no_grad():
        embeddings = model(input_ids=encoded['input_ids'].to(device), attention_mask=encoded['attention_mask'].to(device))[0].cpu()
        print('out emb shape')
        print(embeddings.shape)
        attention_mask = encoded["attention_mask"].cpu()
        for seq_num in range(len(embeddings)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            print("seq len")
            print(seq_len)
            seq_emd = embeddings[seq_num][:seq_len - 1]
            seq_emd_mean = seq_emd.mean(dim=0)
            print("mean:")
            print(seq_emd_mean.shape)


def main():
    protein_sequences = [
        'MKALCLLL',
        'GSHMS',
    ]
    print(len(protein_sequences[0]))
    print(len(protein_sequences[1]))
    print('batch')
    ankh_encoding(protein_sequences)
    print('by one')
    embed_dataset(protein_sequences)
