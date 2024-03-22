import numpy as np
import os
from keras.models import load_model

# Creates testing folder to record information if not already exist
directory = 'home/zwang/cosmic-ray-nn/testing/testing_details/'
os.makedirs(directory, exist_ok=True)

# Specify which GPU it runs on
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# Get the processed data and load data into a multi-array object
preprocessed_data = 'DataFast/zwang/data_prod_0_to_20.npz'
f = np.load(preprocessed_data, allow_pickle=True)

# Extract variables from file
mass = f['mass']
zen = f['zenith']
X = f['x']
dEdX = f['dEdX']

# Format data
zen = np.repeat(zen[:, np.newaxis], X.shape[1], axis=1)
sequential_features = np.stack([X, dEdX, zen], axis=-1)

# Obtain indices of the test set
indicesFile = 'DataFast/zwang/train_indices_prod_0_to_20.npz'
indices = np.load(indicesFile)
indices_test = indices['indices_test']

# Find the corresponding testing data points
x_test_sequential = sequential_features[indices_test]
y_test = mass[indices_test]

model = load_model('home/zwang/cosmic-ray-nn/training/training_details/[l: 0.1395, vl: 0.1262].h5')

mass_pred = model.predict([x_test_sequential])
mass_pred = mass_pred.reshape(len(y_test))
        
with open(f'home/zwang/cosmic-ray-nn/testing/testing_details/model_predictions.txt', 'w') as file:
        for actual, predicted in zip(y_test, mass_pred):
            file.write(f'Actual: {actual}, Predicted: {predicted}\n')

np.savez('home/zwang/cosmic-ray-nn/testing/testing_details/model_predictions.npz', actual=y_test, predicted=mass_pred)