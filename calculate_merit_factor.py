import numpy as np
import os
from matplotlib import pyplot as plt

results_folder = '/home/zwang/cosmic-ray-nn/testing/testing_details'
results = np.load(f'{results_folder}/model_predictions.npz')

truth = results['actual']
prediction = results['predicted']

filter_negatives = True

proton_predictions = prediction[truth == 0.0]
iron_predictions = prediction[truth == 4.02535169073515]

if filter_negatives:
    proton_predictions[proton_predictions < 0] = 0

proton_expected_value = np.mean(proton_predictions)
iron_expected_value = np.mean(iron_predictions)

proton_std = np.std(proton_predictions)
iron_std = np.std(iron_predictions)

merit_factor = (iron_expected_value - proton_expected_value) / (proton_std ** 2 + iron_std ** 2) ** 0.5 

if filter_negatives:
    with open(f'{results_folder}/merit_factor_with_filter.txt', 'w') as file:
        file.write(f"Merit Factor: {merit_factor}\nProton Expected Value: {proton_expected_value} \nIron Expected Value: {iron_expected_value} \nProton Standard Deviation: {proton_std} \nIron Standard Deviation: {iron_std}")
else:
    with open(f'{results_folder}/merit_factor.txt', 'w') as file:
        file.write(f"Merit Factor: {merit_factor}\nProton Expected Value: {proton_expected_value} \nIron Expected Value: {iron_expected_value} \nProton Standard Deviation: {proton_std} \nIron Standard Deviation: {iron_std}")