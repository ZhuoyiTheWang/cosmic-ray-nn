import numpy as np
import tensorflow as tf
import os
from matplotlib import pyplot as plt
from keras.layers import Layer, Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from keras.models import Model
from keras.optimizers import Adam

# Specify which GPU it trains on
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Get the processed training data
preprocessed_data = 'DataFast/zwang/data_small.npz'

# Load data into a multi-array object
f = np.load(preprocessed_data, allow_pickle=True)

# Extract variables from file
mass = f['mass']
zen = f['zenith']
X = f['x']
dEdX = f['dEdX']

# Reshape zenith to a 2-D array to be concatenated with dEdX and X
zen = np.repeat(zen[:, np.newaxis], X.shape[1], axis=1)

# Format data
features = np.stack([X, dEdX, zen], axis=-1)

# Split the data into training and test sets
indicesFile = 'DataFast/zwang/train_indices_small.npz'
indices = np.load(indicesFile)
indices_train = indices['indices_train']
indices_test = indices['indices_test']

# Split the non-array data into train and test
x_train = features[indices_train]
x_test = features[indices_test]
y_train = mass[indices_train]
y_test = mass[indices_test]

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

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return x + res

def build_model(sequence_len, feature_size, head_size, num_heads, ff_dim, num_layers, dropout=0.1):
    inputs = Input(shape=(sequence_len, feature_size))
    x = PositionalEncoding(sequence_len, feature_size)(inputs)
    for _ in range(num_layers):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = LayerNormalization(epsilon=1e-6)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(1)(x)  # Assuming a single output value for each time step

    return Model(inputs, x)

sequence_len = x_train.shape[1]  # Number of events in the sequence
feature_size = 3  # Number of features per time step (X, dEdX, Zenith Angle)

# Transformer hyperparameters
head_size = 64  # Size of each attention head
num_heads = 8  # Number of attention heads
ff_dim = 256  # Hidden layer size in feed forward network inside transformer
num_layers = 4  # Number of transformer layers
dropout = 0.1  # Dropout rate

model = build_model(sequence_len, feature_size, head_size, num_heads, ff_dim, num_layers, dropout)

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
fit = model.fit(x_train, y_train, batch_size=32, epochs=3, validation_split=0.2)

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

plt.savefig('home/zwang/cosmic-ray-nn/training_curves.png')

# Analyze performance
mass_pred = model.predict(x_test)

print(mass_pred.shape)

mass_pred = mass_pred.reshape(len(y_test))