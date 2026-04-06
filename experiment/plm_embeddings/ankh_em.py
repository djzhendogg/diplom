import ankh
import torch

# Load Ankh base.
print(dir(ankh))
model, tokenizer = ankh.load_base_model()
model.eval()

binary_classification_model = ankh.ConvBertForBinaryClassification(
    input_dim=768,
    nhead=4,
    hidden_dim=384,
    num_hidden_layers=1,
    num_layers=1,
    kernel_size=7,
    dropout=0.2,
    pooling='max',
)
binary_classification_model.eval()

protein_sequences = [
    'MKALCLLLLPVLGLLVSSKTLCSMEEAINERIQEVAGSLIFRAISSIGLECQSVTSRGDLATCPRGFAVTGCTCGSACGSWDVRAETTCHCQCAGMDWTGARCCRVQPLEHHHHHH',
    'GSHMSLFDFFKNKGSAATATDRLKLILAKERTLNLPYMEEMRKEIIAVIQKYTKSSDIHFKTLDSNQSVETIEVEIILPR',
]

protein_sequences = [list(seq) for seq in protein_sequences]

outputs = tokenizer(
    protein_sequences,
    add_special_tokens=True,
    padding=True,
    is_split_into_words=True,
    return_tensors="pt",
)
with torch.no_grad():
    embeddings = model(input_ids=outputs['input_ids'], attention_mask=outputs['attention_mask'])
    print(dir(embeddings))
    print(embeddings[0].shape)

with torch.no_grad():
    embeddings = binary_classification_model(input_ids=outputs['input_ids'], attention_mask=outputs['attention_mask'])
    print(embeddings.shape)

