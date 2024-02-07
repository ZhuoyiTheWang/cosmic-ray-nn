import numpy as np

preprocessed_data = 'DataFast/zwang/data_small.npz'
f = np.load(preprocessed_data, allow_pickle=True)

# Extract variables from file
mass = f['mass']

# Create train and test split
shuffled_indices = np.arange(len(mass))

# Shuffle the data randomly
np.random.shuffle(shuffled_indices)

# Split the data into training and test sets
test_size = int(0.05 * len(shuffled_indices)) # Calculate the number of samples for the test set (20%)
indices_train = shuffled_indices[test_size:]
indices_test = shuffled_indices[:test_size]
np.savez('DataFast/zwang/train_indices_small.npz', indices_train=indices_train, indices_test=indices_test)