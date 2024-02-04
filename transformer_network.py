import numpy as np
import tensorflow as tf
import os
from matplotlib import pyplot as plt
from keras.layers import Layer, Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Concatenate, Masking
from keras.models import Model
from keras.optimizers import Adam
from itertools import product

# Specify which GPU it trains on
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Get the processed training data
preprocessed_data = 'DataFast/zwang/data_small.npz'

# Load data into a multi-array object
f = np.load(preprocessed_data, allow_pickle=True)

# Extract variables from file
mass = f['mass']
zen = f['zenith']
Xmx = f['Xmx']
X = f['x']
dEdX = f['dEdX']

# Format data
sequential_features = np.stack([X, dEdX], axis=-1)
singular_feartures = np.stack([Xmx, zen], axis=-1)

# Split the data into training and test sets
indicesFile = 'DataFast/zwang/train_indices_small.npz'
indices = np.load(indicesFile)
indices_train = indices['indices_train']
indices_test = indices['indices_test']

# Split the non-array data into train and test
x_train_sequential = sequential_features[indices_train]
x_test_sequential = sequential_features[indices_test]
x_train_singular = singular_feartures[indices_train]
x_test_singular = singular_feartures[indices_test]
y_train = mass[indices_train]
y_test = mass[indices_test]

sequence_len = sequential_features.shape[1]  # Number of events in the sequence
sequential_feature_size = 2  # Number of features per time step (X, dEdX)

class PositionalEncoding(Layer):
    def __init__(self, sequence_len, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.pos_encoding = self.positional_encoding(sequence_len, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)
        # Apply sin to even indices in the array; 2i
        sines = np.sin(angle_rads[:, 0::2])
        # Apply cos to odd indices in the array; 2i+1
        cosines = np.cos(angle_rads[:, 1::2])
        pos_encoding = np.concatenate([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout):
    # Normalization and Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x,x)
    x = Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="elu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return x + res

def build_model(sequence_len, feature_size, head_size, num_heads, ff_dim, num_layers, dropout=0.1):
    sequence_input = Input(shape=(sequence_len, feature_size))
    singular_input = Input(shape=(2,))

    x = Masking(mask_value=0, input_shape=(sequence_len, feature_size))(sequence_input)
    x = PositionalEncoding(sequence_len, feature_size)(x)
    for _ in range(num_layers):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = LayerNormalization(epsilon=1e-6)(x)
    x = GlobalAveragePooling1D()(x)
    x = Concatenate()([x, singular_input])
    x = Dense(1)(x)  # Assuming a single output value for each time step

    return Model(inputs = [sequence_input, singular_input], outputs = x)

# Transformer hyperparameters  
hyperparameters = {
    'ff_dim': [128, 256, 512], # Hidden layer size in feed forward network inside transformer
    'dropout': [0.1], # Dropout rate
    'batch_size': [32, 64], # Batch size
    'num_layers': [1, 3, 5, 7, 9], # Number of transformer layers
    'head_size': [4, 8, 16, 32], # Size of each attention head
    'num_heads': [4, 8, 12, 20] # Number of attention heads
}

# Function to train a model and return the validation loss
def train_and_evaluate_model(hp):
    model = build_model(sequence_len, sequential_feature_size, hp['head_size'], hp['num_heads'], hp['ff_dim'], hp['num_layers'], hp['dropout'])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    fit = model.fit([x_train_sequential, x_train_singular], y_train, batch_size=hp['batch_size'], epochs=25, validation_split=0.25)  # Set verbose to 0 to suppress the detailed training log
    validation_loss = np.min(fit.history['val_loss'])  # Get the best validation loss during the training
    return model, validation_loss, fit

# Initialize variables to store the best model and its performance
best_model = None
best_fit = None
best_validation_loss = np.inf
best_hp = {}
hyperparameter_iterator = 1

with open('home/zwang/cosmic-ray-nn/training_params.txt', 'w') as file:
    file.write(f"Training Details:")

for hp_values in product(*hyperparameters.values()):
    hp = dict(zip(hyperparameters.keys(), hp_values))
    print(f"Training with hyperparameters: {hp}")
    model, validation_loss, fit = train_and_evaluate_model(hp)
    with open('home/zwang/cosmic-ray-nn/training_params.txt', 'a') as file:
        file.write(f"\nCurrent model: {hyperparameter_iterator}, val_loss: {validation_loss}, hyperparameters: {hp}")

    # Plot training curves
    fig, ax = plt.subplots(1, figsize=(8,5))
    n = np.arange(len(fit.history['loss']))

    ax.plot(n, fit.history['loss'], ls='--', c='k', label='loss')
    ax.plot(n, fit.history['val_loss'], label='val_loss', color='red')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.semilogy()
    ax.grid()
    plt.title('Training and Validation Loss')
    plt.savefig(f"home/zwang/cosmic-ray-nn/training_curves/model{hyperparameter_iterator}.png")

    hyperparameter_iterator += 1

    # Update the best model if current model is better
    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        best_model = model
        best_fit = fit
        best_hp = hp
        # Announce the new best model
        print(f"New best model with val_loss: {best_validation_loss}, hyperparameters: {best_hp}")

# Save the best model
best_model.save('home/zwang/cosmic-ray-nn/best_model.h5')
print(f"Best model val_loss: {best_validation_loss}, hyperparameters: {best_hp}")

# Plot training curves
fig, ax = plt.subplots(1, figsize=(8,5))
n = np.arange(len(best_fit.history['loss']))

ax.plot(n, best_fit.history['loss'], ls='--', c='k', label='loss')
ax.plot(n, best_fit.history['val_loss'], label='val_loss', color='red')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
ax.semilogy()
ax.grid()
plt.title('Training and Validation Loss')
plt.savefig('home/zwang/cosmic-ray-nn/training_curves/best_model_training_curves.png')

with open('home/zwang/cosmic-ray-nn/best_params.txt', 'w') as file:
    file.write(f"\nBest model val_loss: {best_validation_loss}, hyperparameters: {best_hp}")

# Analyze performance
# mass_pred = model.predict([x_test_sequential, x_test_singular])
# print(mass_pred.shape)
# mass_pred = mass_pred.reshape(len(y_test))