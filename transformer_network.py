import numpy as np
import tensorflow as tf
import os
import io
from matplotlib import pyplot as plt
from keras.layers import Layer, Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Masking, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, LambdaCallback, ModelCheckpoint, Callback
from keras.optimizers.schedules import ExponentialDecay
from itertools import product

# Specify which GPU it trains on
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Get the processed data
preprocessed_data = 'DataFast/zwang/data_prod_0_to_20.npz'

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
indicesFile = 'DataFast/zwang/train_indices_prod_0_to_20.npz'
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

# class PositionalEncoding(Layer):
#     def __init__(self, sequence_len, d_model, **kwargs):
#         super(PositionalEncoding, self).__init__(**kwargs)
#         self.pos_encoding = self.positional_encoding(sequence_len, d_model)

#     def get_angles(self, position, i, d_model):
#         angles = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
#         return position * angles

#     def positional_encoding(self, position, d_model):
#         angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
#                                      np.arange(d_model)[np.newaxis, :],
#                                      d_model)
#         # Apply sin to even indices in the array; 2i
#         sines = np.sin(angle_rads[:, 0::2])
#         # Apply cos to odd indices in the array; 2i+1
#         cosines = np.cos(angle_rads[:, 1::2])
#         pos_encoding = np.concatenate([sines, cosines], axis=-1)
#         pos_encoding = pos_encoding[np.newaxis, ...]
#         return tf.cast(pos_encoding, dtype=tf.float32)

#     def call(self, inputs):
#         return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

class DynamicPatienceCallback(Callback):
    def __init__(self, early_stopping_callback, loss_threshold=0.8, high_patience=100, low_patience=20, window_size=5):
        super().__init__()
        self.early_stopping_callback = early_stopping_callback
        self.loss_threshold = loss_threshold
        self.high_patience = high_patience
        self.low_patience = low_patience
        self.window_size = window_size
        self.recent_losses = []
        self.best_median = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get('val_loss')
        
        self.recent_losses.append(current_val_loss)
        if len(self.recent_losses) > self.window_size:
            self.recent_losses.pop(0) 
        
        median_loss = np.median(self.recent_losses)

        if median_loss < self.loss_threshold:
            self.early_stopping_callback.patience = self.high_patience
        else:
            self.early_stopping_callback.patience = self.low_patience

class InterruptHandler(Callback):
    def __init__(self):
        super().__init__()
        self.history={}

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            for key, value in logs.items():
                self.history.setdefault(key, []).append(value)

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
    # x = Dense(1024, activation=activation)(x)
    # x = Dense(512, activation=activation)(x)
    x = Dense(1)(x)  # Assuming a single output value for each time step

    return Model(inputs = sequence_input, outputs = x)

# Transformer hyperparameters  
hyperparameters = {
    'ff_dim': [16], # Hidden layer size in feed forward network inside transformer
    'dropout': [0.1], # Dropout rate
    'batch_size': [32], # Batch size
    'activation': ['elu'], # Activation function
    'num_encoder_layers': [16], # Number of transformer encoder layers
    'num_decoder_layers' : [0], # Number of transformer decoder layers
    'head_size': [64], # Size of each attention head
    'num_heads': [8] # Number of attention heads
}

# Initialize hyperparameter iterator
hyperparameter_iterator = 1

initial_learning_rate = 0.001
decay_steps = 0.75 * len(y_train) / hyperparameters['batch_size'][0] * 10 # Decays every 10 epochs
decay_rate = 0.96

# Learning rate scheduler
lr_schedule = ExponentialDecay(
    initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=True  # Set to False for smooth decay, True for discrete steps
)

# Function to train a model and return the validation loss
def train_and_evaluate_model(hp):
    
    def print_lr(epoch, logs):
        lr = tf.keras.backend.get_value(model.optimizer.lr)
        with open(f'home/zwang/cosmic-ray-nn/training/training_details/lr_logger_{hyperparameter_iterator}', 'a') as file:
            file.write(f"Epoch {epoch+1}: {lr:.6f}\n")

    optimizer = Adam(beta_1=0.9, beta_2=0.98, epsilon=1e-8)

    early_stopping = EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=True)
    dynamic_patience = DynamicPatienceCallback(early_stopping)
    lr_logger = LambdaCallback(on_epoch_begin=lambda epoch, logs: print_lr(epoch, logs))
    save_best_model = ModelCheckpoint('home/zwang/cosmic-ray-nn/training/training_details/best_model.h5', monitor='val_loss', save_best_only=True, mode='min')
    save_current_model = ModelCheckpoint('home/zwang/cosmic-ray-nn/training/training_details/current_model.h5')
    interrupt_handler = InterruptHandler()

    model = build_model(sequence_len, sequential_feature_size, hp['head_size'], hp['num_heads'], hp['ff_dim'], hp['num_encoder_layers'], hp['num_decoder_layers'], hp['dropout'], hp['activation'])
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    
    # Capture the summary to a string
    summary_string = io.StringIO()
    model.summary(print_fn=lambda x: summary_string.write(x + '\n'))
    summary_content = summary_string.getvalue()
    summary_string.close()

    with open(f'home/zwang/cosmic-ray-nn/model_structure_no_dense.txt', 'w') as file:
        file.write(summary_content)
    
    history = None

    try:
        fit = model.fit(x_train_sequential, y_train, batch_size=hp['batch_size'], epochs=1500, validation_split=0.25, callbacks=[early_stopping, lr_logger, dynamic_patience, save_best_model, save_current_model, interrupt_handler])
        history = fit.history
        validation_loss = np.min(history['val_loss'])  # Get the best validation loss during the training
    except KeyboardInterrupt:
        history = interrupt_handler.history
        validation_loss = np.min(history['val_loss'])

    return model, validation_loss, history

with open('home/zwang/cosmic-ray-nn/training/training_details/training_params.txt', 'w') as file:
    file.write(f"Training Details:")

for hp_values in product(*hyperparameters.values()):
    hp = dict(zip(hyperparameters.keys(), hp_values))
    print(f"Training with hyperparameters: {hp}")
    model, validation_loss, history = train_and_evaluate_model(hp)
    
    best_epoch = np.argmin(history['val_loss']) + 1
    terminal_epoch = len(history['val_loss'])
        
    with open('home/zwang/cosmic-ray-nn/training/training_details/training_params.txt', 'a') as file:
        file.write(f"\nCurrent model: {hyperparameter_iterator}, min val_loss: {validation_loss} at epoch {best_epoch}, terminated at epoch {terminal_epoch}, hyperparameters: {hp}")
    
    with open(f'home/zwang/cosmic-ray-nn/training/training_details/training_history_{hyperparameter_iterator}.txt', 'w') as file:
        for loss, val_loss in zip(history['loss'], history['val_loss']):
            file.write(f'{loss} {val_loss}\n')
    
    # Plot training curves
    fig, ax = plt.subplots(1, figsize=(8,5))
    n = np.arange(len(history['loss']))

    ax.plot(n, history['loss'], ls='--', c='k', label='loss')
    ax.plot(n, history['val_loss'], label='val_loss', color='red')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.semilogy()
    ax.grid()
    plt.title(f'Training and Validation Loss Model {hyperparameter_iterator}')
    plt.savefig(f'home/zwang/cosmic-ray-nn/training/training_details/model_{hyperparameter_iterator}_training_curves.png')

    # Analyze performance
    mass_pred = model.predict([x_test_sequential])
    mass_pred = mass_pred.reshape(len(y_test))
        
    with open(f'home/zwang/cosmic-ray-nn/training/training_details/model_{hyperparameter_iterator}_predictions.txt', 'w') as file:
            for actual, predicted in zip(y_test, mass_pred):
                file.write(f'Actual: {actual}, Predicted: {predicted}\n')
        
    hyperparameter_iterator += 1