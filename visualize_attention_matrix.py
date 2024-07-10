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
    plt.figure(figsize=(8, 6))
    plt.scatter(proton_X_values[i], proton_dEdX_values[i], alpha=0.5)
    plt.title(f'Entry {i+1}: Scatter Plot of X vs dEdX')
    plt.xlabel('X')
    plt.ylabel('dEdX')
    plt.savefig(f'home/zwang/cosmic-ray-nn/entry_graphs/proton_entry_{i+1}.png')  # Save the figure to a file
    plt.close()

iron_X_values = iron_entries[:10, ..., 0]  # First 10 entries, X coordinate
iron_dEdX_values = iron_entries[:10, ..., 1]  # First 10 entries, dEdX coordinate

# Loop through the first 10 entries and save each plot as a PNG
for i in range(10):
    plt.figure(figsize=(8, 6))
    plt.scatter(iron_X_values[i], iron_dEdX_values[i], alpha=0.5)
    plt.title(f'Entry {i+1}: Scatter Plot of X vs dEdX')
    plt.xlabel('X')
    plt.ylabel('dEdX')
    plt.savefig(f'home/zwang/cosmic-ray-nn/entry_graphs/iron_entry_{i+1}.png')  # Save the figure to a file
    plt.close()

proton_first_10 = proton_entries[:10]
iron_first_10 = iron_entries[:10]
combined_entries = np.concatenate([proton_first_10, iron_first_10], axis=0)

outputs, attention_weights = model.predict(combined_entries)

first_sample_first_head_attention = attention_weights[0][0, 0]  # First layer, first sample, first head

# Visualize the attention matrix using matplotlib
plt.imshow(first_sample_first_head_attention, cmap='viridis', aspect='auto')
plt.colorbar()
plt.xlabel("Position in Sequence")
plt.ylabel("Position in Sequence")
plt.title("Attention Weights of First Head for First Sample")
plt.savefig("home/zwang/cosmic-ray-nn/first_sample_first_head.png")
plt.show()
