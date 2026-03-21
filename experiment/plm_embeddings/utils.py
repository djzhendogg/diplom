import json
import os

import pandas as pd
import torch


def setup_torch_device():
    """Initialize and validate Torch CUDA device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cpu":
        raise RuntimeError("CUDA device is required for this script")

    return device


def save_embeddings(
        sequences_df,
        encoding_func,
        sequence_column,
        output_path
):
    """
    Generate embeddings for sequences.
    """
    sequences = sequences_df[sequence_column].tolist()

    embeddings = encoding_func(sequences)

    embeddings_df = pd.DataFrame(embeddings)

    result_df = pd.concat([sequences_df, embeddings_df], axis=1)

    result_df.to_pickle(output_path)


def process_data_files(files, embeddings_type, encoding_func):
    save_path = 'results'
    out_path = os.path.join(save_path, embeddings_type)

    errors = []
    for file in files:
        try:
            data_df = pd.read_csv(file)[:10]
            output_file = os.path.join(out_path, file + '.pkl')
            save_embeddings(
                data_df,
                encoding_func,
                'sequence',
                output_file
            )

            print("Processing completed successfully!")

        except FileNotFoundError as e:
            error = {
                'file': file,
                'error': e,
                'comment':'Could not find input file'
            }
            errors.append(error)
        except Exception as e:
            error = {
                'file': file,
                'error': e,
                'comment':'Error during processing'
            }
            errors.append(error)
    if errors:
        errors_path = os.path.join(save_path, f'errors_{embeddings_type}.json')
        with open(errors_path, 'w', encoding='utf-8') as f:
            json.dump({
                'total_errors': len(errors),
                'errors': errors
            }, f, ensure_ascii=False, indent=4)
