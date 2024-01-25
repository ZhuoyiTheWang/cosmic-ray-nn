import numpy as np
import os
# from keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention
# from keras.models import Model
# from keras.optimizers import Adam

# Specify which GPU it trains on
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Get the processed training data
preprocessed_data = 'DataFast/zwang/data.npz'

# Load data into a multi-array object
f = np.load(preprocessed_data, allow_pickle=True)

# Extract variables from file
mass = f['mass']
zen = f['zenith']
x_var = f['x']
dEdX = f['dEdX']

# Create train and test split
x_train = np.stack([zen, x_var, dEdX], axis=-1)

print(x_train[0])

# Split the data into training and test sets
indicesFile = 'DataFast/zwang/train_indices.npz'
indices = np.load(indicesFile)
indices_train = indices['indices_train']
indices_test = indices['indices_test']

# Split the non-array data into train and test
x_train1 = X_singular[indices_train]
x_test1 = X_singular[indices_test]
y_train = mass[indices_train]
y_test = mass[indices_test]

dEdX_train = dEdX[indices_train]
dEdX_test = dEdX[indices_test]
Xvar_train = x_var[indices_train]
Xvar_test = x_var[indices_test]
x_train2 = np.stack([dEdX_train, Xvar_train], axis=-1)
x_test2 = np.stack([dEdX_test, Xvar_test], axis=-1)

# define input shape
input_shape = (3,)

# define input layer
inputs = Input(shape=input_shape)

# Define transformer and feedforward layers as functions
def transformer_block(x):
    attention = MultiHeadAttention(num_heads=8, key_dim=64)([x, x])
    x = Dropout(0.1)(attention)
    return LayerNormalization(epsilon=1e-6)(x + attention)

def feedforward_block(x, units):
    x = Dense(units, activation='relu')(x)
    x = Dropout(0.1)(x)
    return LayerNormalization(epsilon=1e-6)(x)

# Apply transformer blocks
x = transformer_block(inputs)
x = transformer_block(x)
x = transformer_block(x)

# Apply feedforward blocks
x = feedforward_block(x, 512)
x = feedforward_block(x, 256)

# Apply final output layer
output = Dense(1, activation='linear')(x)

# Create model
model = Model(inputs=inputs, outputs=output)

# compile model
optimizer = Adam(lr=1e-4)
model.compile(optimizer=optimizer, loss='mse')

fit = model.fit([x_train1, x_train2], y_train, epochs=20, batch_size=64, validation_split=0.1)

# Analyze performance
mass_pred = model.predict([x_test1, x_test2], batch_size=64, verbose=1)[:,0]
mass_pred = mass_pred.reshape(len(y_test))