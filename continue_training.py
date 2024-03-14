import numpy as np
import tensorflow as tf
import os
from matplotlib import pyplot as plt
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

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

model_location = 'home/zwang/cosmic-ray-nn/training/(17) Successful Long Train [elu]/best_model.h5'

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

# Function to train a model and return the validation loss
def train_and_evaluate_model(model_location):

    # early_stopping = EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=True)
    # dynamic_patience = DynamicPatienceCallback(early_stopping)
    best_model = ModelCheckpoint('home/zwang/cosmic-ray-nn/training/training_details/best_model.h5', monitor='val_loss', save_best_only=True, mode='min')
    current_model = ModelCheckpoint('home/zwang/cosmic-ray-nn/training/training_details/current_model.h5')
    interrupt_handler = InterruptHandler()

    model = load_model(model_location)
    
    history = None

    try:
        fit = model.fit(x_train_sequential, y_train, batch_size=32, epochs=1500, validation_split=0.25, callbacks=[best_model, current_model, interrupt_handler])
        history = fit.history
        validation_loss = np.min(history['val_loss'])  # Get the best validation loss during the training
    except KeyboardInterrupt:
        history = interrupt_handler.history
        validation_loss = np.min(history['val_loss'])

    return model, validation_loss, history

with open('home/zwang/cosmic-ray-nn/training/training_details/training_params.txt', 'w') as file:
    file.write(f"Continue training model at {model_location}")

model, validation_loss, history = train_and_evaluate_model(model_location)

best_epoch = np.argmin(history['val_loss']) + 1
terminal_epoch = len(history['val_loss'])
    
with open('home/zwang/cosmic-ray-nn/training/training_details/training_params.txt', 'a') as file:
    file.write(f"\nCurrent model: min val_loss: {validation_loss} at epoch {best_epoch}, terminated at epoch {terminal_epoch}")

with open(f'home/zwang/cosmic-ray-nn/training/training_details/training_history.txt', 'w') as file:
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
plt.title(f'Training and Validation Loss Model')
plt.savefig(f'home/zwang/cosmic-ray-nn/training/training_details/model_training_curves.png')

# Analyze performance
mass_pred = model.predict([x_test_sequential])
mass_pred = mass_pred.reshape(len(y_test))
    
with open(f'home/zwang/cosmic-ray-nn/training/training_details/model_predictions.txt', 'w') as file:
        for actual, predicted in zip(y_test, mass_pred):
            file.write(f'Actual: {actual}, Predicted: {predicted}\n')
