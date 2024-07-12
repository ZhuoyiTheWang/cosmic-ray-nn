import numpy as np
import tensorflow as tf
import os
from matplotlib import pyplot as plt
from keras.models import load_model, Model
from keras.layers import Layer, Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Masking, Flatten
from keras.optimizers import Adam


# Specify which GPU it trains on
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Get the processed data
preprocessed_data = 'DataFast/zwang/data_prod_0_to_20_zen_below_60.npz'

# Load data into a multi-array object
f = np.load(preprocessed_data, allow_pickle=True)

# Creates training folder to record information if not already exist
directory = 'home/zwang/cosmic-ray-nn/training/training_details/'
os.makedirs(directory, exist_ok=True)

# Extract variables from file
mass = f['mass']
zen = f['zenith']
X = f['x']
dEdX = f['dEdX']

# Format data
zen = np.repeat(zen[:, np.newaxis], X.shape[1], axis=1)
sequential_features = np.stack([X, dEdX, zen], axis=-1)

# Split the data into training and test sets
indicesFile = 'DataFast/zwang/train_indices_prod_0_to_20_zen_below_60.npz'
indices = np.load(indicesFile)
indices_train = indices['indices_train']
indices_test = indices['indices_test']

# Split the non-array data into train and test
x_train_sequential = sequential_features[indices_train]
x_test_sequential = sequential_features[indices_test]
y_train = mass[indices_train]
y_test = mass[indices_test]

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout, activation):
    # Normalization and Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x, attn_weights = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x, x, return_attention_scores=True)
    x = Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation=activation)(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    output = x + res
    return output, attn_weights

def build_model(sequence_len, feature_size, head_size, num_heads, ff_dim, num_encoder_layers, dropout, activation):
    sequence_input = Input(shape=(sequence_len, feature_size), batch_size=None)
    all_encoder_attn_weights = []

    encoder_output = sequence_input
    for _ in range(num_encoder_layers):
        encoder_output, attn_weights = transformer_encoder(encoder_output, head_size, num_heads, ff_dim, dropout, activation)
        all_encoder_attn_weights.append(attn_weights)

    x = LayerNormalization(epsilon=1e-6)(encoder_output)
    x = Flatten()(x)
    x = Dense(512, activation=activation)(x)
    x = Dense(256, activation=activation)(x)
    x = Dense(1)(x)  # Assuming a single output value for each time step

    return Model(inputs = sequence_input, outputs=[x, all_encoder_attn_weights])

hp = {
    'ff_dim': 16, # Hidden layer size in feed forward network inside transformer
    'dropout': 0.1, # Dropout rate
    # 'batch_size': 32, # Batch size
    'activation': 'elu', # Activation function
    'num_encoder_layers': 16, # Number of transformer encoder layers
    'head_size': 64, # Size of each attention head
    'num_heads': 8 # Number of attention heads
}

sequence_len = sequential_features.shape[1]  # Number of events in the sequence
sequential_feature_size = 3  # Number of features per time step (X, dEdX, zen)

model_location = 'home/zwang/cosmic-ray-nn/training/training_details/best_model.h5'

model = build_model(sequence_len, sequential_feature_size, hp['head_size'], hp['num_heads'], hp['ff_dim'], hp['num_encoder_layers'], hp['dropout'], hp['activation'])
model.load_weights(model_location, by_name=True, skip_mismatch=True)

proton_entries = x_test_sequential[y_test == 0.0]
iron_entries = x_test_sequential[y_test == 4.02535169073515]

print('X test length ', len(x_test_sequential))
print('Proton length ', len(proton_entries))
print('Iron length ', len(iron_entries))

proton_X_values = proton_entries[:10, ..., 0]  # First 10 entries, X coordinate
proton_dEdX_values = proton_entries[:10, ..., 1]  # First 10 entries, dEdX coordinate

# Loop through the first 10 entries and save each plot as a PNG
for i in range(10):
    sequence_length = len(proton_dEdX_values[i])

    # Generate sequence indices
    sequence_indices = np.arange(sequence_length)

    plt.figure(figsize=(8, 6))
    plt.scatter(sequence_indices, proton_dEdX_values[i], alpha=0.5)
    plt.title(f'Entry {i+1}: Scatter Plot of X vs dEdX')
    plt.xlabel('X')
    plt.ylabel('dEdX')
    plt.savefig(f'home/zwang/cosmic-ray-nn/entry_graphs/proton_entry_{i+1}.png')  # Save the figure to a file
    plt.close()

iron_X_values = iron_entries[:10, ..., 0]  # First 10 entries, X coordinate
iron_dEdX_values = iron_entries[:10, ..., 1]  # First 10 entries, dEdX coordinate

# Loop through the first 10 entries and save each plot as a PNG
for i in range(10):
    sequence_length = len(iron_dEdX_values[i])

    # Generate sequence indices
    sequence_indices = np.arange(sequence_length)

    plt.figure(figsize=(8, 6))
    plt.scatter(sequence_indices, iron_dEdX_values[i], alpha=0.5)
    plt.title(f'Entry {i+1}: Scatter Plot of X vs dEdX')
    plt.xlabel('X')
    plt.ylabel('dEdX')
    plt.savefig(f'home/zwang/cosmic-ray-nn/entry_graphs/iron_entry_{i+1}.png')  # Save the figure to a file
    plt.close()

proton_first_10 = proton_entries[:10]
iron_first_10 = iron_entries[:10]
combined_entries = np.concatenate([proton_first_10, iron_first_10], axis=0)

outputs, attention_weights = model.predict(combined_entries)

attention_array = np.array(attention_weights)  # shape [num_layers, num_samples, num_heads, seq_len, seq_len]

num_samples = attention_array.shape[1]
seq_len = attention_array.shape[-1]

# Loop through each sample to compute aggregated sum and mean attention
for sample_idx in range(num_samples):
    # Sum and mean across layers and heads for this sample
    sum_attention = np.sum(attention_array[:, sample_idx, :, :, :], axis=(0, 1))
    mean_attention = np.mean(attention_array[:, sample_idx, :, :, :], axis=(0, 1))

    sum_attention_flipped = np.flipud(sum_attention)
    mean_attention_flipped = np.flipud(mean_attention)

    # Function to add row and column indices to the matrix
    def add_indices(matrix):
        rows, cols = matrix.shape
        row_indices = np.arange(rows)[:, None]  # Create column vector for row indices
        col_indices = np.arange(cols + 1)  # Column indices with one extra for the row header

        # Prepend row indices to the matrix
        matrix_with_row_indices = np.hstack((row_indices, matrix))

        # Append column indices as a header row
        matrix_with_indices = np.vstack((col_indices, matrix_with_row_indices))

        return matrix_with_indices

    # Apply function
    sum_attention_indexed = add_indices(sum_attention_flipped)
    mean_attention_indexed = add_indices(mean_attention_flipped)

    # Saving the matrices with indices
    def save_matrix_with_indices(path, matrix):
        header = ','.join([''] + [str(i) for i in range(matrix.shape[1] - 1)])  # Adjust header for zero-based column index
        np.savetxt(path, matrix, fmt='%g', delimiter=',', header=header, comments='')

    # File paths
    sum_attention_path = f'home/zwang/cosmic-ray-nn/aggregated_attention/sum_attention_sample_{sample_idx + 1}.txt'
    mean_attention_path = f'home/zwang/cosmic-ray-nn/aggregated_attention/mean_attention_sample_{sample_idx + 1}.txt'

    # Save indexed matrices
    save_matrix_with_indices(sum_attention_path, sum_attention_indexed)
    save_matrix_with_indices(mean_attention_path, mean_attention_indexed)

    # Plotting the sum attention heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(sum_attention, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.xlabel("Position in Sequence")
    plt.ylabel("Position in Sequence")
    plt.gca().invert_yaxis() 
    plt.title(f"Summed Attention Map for Sample {sample_idx+1}")
    plt.savefig(f"home/zwang/cosmic-ray-nn/aggregated_attention/sum_attention_sample_{sample_idx+1}.png")
    plt.close()  # Close to free memory

    # Plotting the mean attention heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(mean_attention, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.xlabel("Position in Sequence")
    plt.ylabel("Position in Sequence")
    plt.gca().invert_yaxis() 
    plt.title(f"Mean Attention Map for Sample {sample_idx+1}")
    plt.savefig(f"home/zwang/cosmic-ray-nn/aggregated_attention/mean_attention_sample_{sample_idx+1}.png")
    plt.close()  # Close to free memory


# num_samples = combined_entries.shape[0]
# num_heads = hp['num_heads']
# num_layers = hp['num_encoder_layers']

# for sample_idx in range(num_samples):
#     for layer_idx in range(num_layers):
#         for head_idx in range(num_heads):
#             attention_data = attention_weights[layer_idx][sample_idx, head_idx]  # Get the specific attention data

#             # Create a grid for sequence positions
#             seq_len = attention_data.shape[0]
#             x, y = np.meshgrid(np.arange(seq_len), np.arange(seq_len))

#             # Flatten the grid and attention data for scatter plotting
#             x = x.flatten()
#             y = y.flatten()
#             sizes = attention_data.flatten() * 1000  # Scale up sizes for better visibility

#             plt.figure(figsize=(10, 8))
#             scatter = plt.scatter(x, y, s=sizes, c=sizes, cmap='viridis', alpha=0.6)
#             plt.colorbar(scatter)
#             plt.xlabel("Position in Sequence")
#             plt.ylabel("Position in Sequence")
#             plt.title(f"Scatter Plot of Attention - Sample {sample_idx+1}, Layer {layer_idx+1}, Head {head_idx+1}")
#             plt.grid(True)  # Optionally add a grid
#             plt.savefig(f"home/zwang/cosmic-ray-nn/attention_scatters/sample_{sample_idx+1}_layer_{layer_idx+1}_head_{head_idx+1}.png")
#             plt.close()  # Close the figure to free up memory
