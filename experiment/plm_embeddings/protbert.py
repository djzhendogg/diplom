import numpy as np
import torch
from transformers import BertModel, BertTokenizer

from utils import setup_torch_device


def encode_sequences_with_protbert(sequences, model_name=None, batch_size=32):
    """
    Encode protein sequences using ProtBERT model with mean pooling.

    Args:
        sequences: List of protein sequences to encode
        model_name: Name of the pre-trained ProtBERT model
        batch_size: Number of sequences to process in each batch

    Returns:
        numpy.ndarray: Array of sequence embeddings (n_sequences, embedding_dim)
    """

    # Setup device
    device = setup_torch_device()

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

            inputs = tokenizer(
                batch_sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )

            # Move inputs to device
            inputs = {key: value.to(device) for key, value in inputs.items()}

            # Forward pass
            outputs = model(**inputs)
            last_hidden = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)

            # Mean pooling: skip special tokens [CLS] and [SEP] (positions 1:-1)
            # Note: This assumes [CLS] is first token and [SEP] is last token
            batch_embeddings = last_hidden[:, 1:-1, :].mean(dim=1).cpu().numpy()
            embeddings.append(batch_embeddings)

    return np.vstack(embeddings)
