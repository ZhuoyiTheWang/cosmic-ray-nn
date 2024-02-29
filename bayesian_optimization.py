import numpy as np
import tensorflow as tf
import os
from matplotlib import pyplot as plt
from keras.layers import Layer, Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Concatenate, Masking, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, LambdaCallback
from keras.optimizers.schedules import ExponentialDecay
from itertools import product
from import BayesianOptimization

# Specify which GPU it trains on
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Get the processed training data
preprocessed_data = 'DataFast/zwang/data_two_prods.npz'

# Load data into a multi-array object
f = np.load(preprocessed_data, allow_pickle=True)

# Extract variables from file
mass = f['mass']
zen = f['zenith']
X = f['x']
dEdX = f['dEdX']

# Format data
zen = np.repeat(zen[:, np.newaxis], X.shape[1], axis=1)
sequential_features = np.stack([X, dEdX, zen], axis=-1)

# Split the data into training and test sets
indicesFile = 'DataFast/zwang/train_indices_bayesian.npz'
indices = np.load(indicesFile)
indices_train = indices['indices_train']
indices_test = indices['indices_test']

# Split the non-array data into train and test
x_train_sequential = sequential_features[indices_train]
x_test_sequential = sequential_features[indices_test]
y_train = mass[indices_train]
y_test = mass[indices_test]

sequence_len = sequential_features.shape[1]  # Number of events in the sequence
sequential_feature_size = 3  # Number of features per time step (X, dEdX, zen)

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout, activation):
    # Normalization and Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation=activation)(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return x + res

def transformer_decoder(inputs, encoder_output, head_size, num_heads, ff_dim, dropout, activation):
    # Encoder-Decoder Attention (cross-attention)
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, encoder_output, encoder_output)
    x = Dropout(dropout)(x)
    res = x + inputs 

    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation=activation)(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return x + res  # Add & Norm again

def build_model(sequence_len, feature_size, head_size, num_heads, ff_dim, num_encoder_layers, num_decoder_layers, dropout, activation):
    sequence_input = Input(shape=(sequence_len, feature_size))
    # x = Masking(mask_value=0, input_shape=(sequence_len, feature_size))(sequence_input)
    # x = PositionalEncoding(sequence_len, feature_size)(x)
    encoder_output = sequence_input
    for _ in range(num_encoder_layers):
        encoder_output = transformer_encoder(encoder_output, head_size, num_heads, ff_dim, dropout, activation)

    decoder_output = encoder_output
    for _ in range(num_decoder_layers):
        decoder_output = transformer_decoder(decoder_output, encoder_output, head_size, num_heads, ff_dim, dropout, activation)

    x = LayerNormalization(epsilon=1e-6)(decoder_output)
    x = Flatten()(x)
    x = Dense(1024, activation=activation)(x)
    x = Dense(512, activation=activation)(x)
    x = Dense(1)(x)  # Assuming a single output value for each time step

    return Model(inputs = sequence_input, outputs = x)

# Function to train a model and return the validation loss
def train_and_evaluate_model(ff_dim, dropout, learning_rate, num_heads, head_size, num_encoder_layers, num_decoder_layers, batch_size):

    ff_dim = int(ff_dim)
    num_encoder_layers = int(num_encoder_layers)
    num_decoder_layers = int(num_decoder_layers)
    num_heads = int(num_heads)
    head_size = int(head_size)

    early_stopping = EarlyStopping(monitor='val_loss', patience=50, mode='min', restore_best_weights=True)

    optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    model = build_model(sequence_len, sequential_feature_size, head_size, num_heads, ff_dim, num_encoder_layers, num_decoder_layers, dropout, activation='selu')
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    fit = model.fit(x_train_sequential, y_train, batch_size=int(batch_size), epochs=500, validation_split=0.25, callbacks=[early_stopping])  # Set verbose to 0 to suppress the detailed training log
    
    _, val_loss = model.evaluate(x_test_sequential, y_test)

    return -val_loss

pbounds = {
    'ff_dim': (8, 128),
    'dropout': (0.01, 0.5),
    'learning_rate': (1e-4, 1e-2),
    'num_heads': (2, 16),
    'head_size': (32, 128),
    'num_encoder_layers': (1, 24),
    'num_decoder_layers': (0, 6),  
    'batch_size': (16, 64)
}

bayesian_optimizer = BayesianOptimization(f=train_and_evaluate_model, pbounds=pbounds, random_state=42)
bayesian_optimizer.maximize(init_points=2, n_iter=10)

results = bayesian_optimizer.res

sorted_results = sorted(results, key=lambda x: x['target'], reverse=True)

# Number of top results you want to retrieve
top_n = 5  # For example, to get the top 3 results

# Retrieve the top N performing hyperparameters
top_performers = sorted_results[:top_n]

# Print the top N performers
for i, result in enumerate(top_performers, 1):
    with open(f'home/zwang/cosmic-ray-nn/training/bayesian_optimization.txt', 'a') as file:
        file.write((f"Rank {i}, Hyperparameters: {result['params']}, Validation Loss: {-result['target']}\n"))
