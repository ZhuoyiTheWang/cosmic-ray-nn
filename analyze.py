import numpy as np
import os
from matplotlib import pyplot as plt

results = np.load('/home/zwang/cosmic-ray-nn/testing/(3) General Model [l: 0.1395, vl: 0.1262]/model_predictions.npz')

truth = results['actual']
prediction = results['predicted']

proton_predictions = prediction[truth == 0.0]
helium_predictions = prediction[truth == 0.6931471805599453]
lithium_predictions = prediction[truth == 1.0986122886681098]
beryllium_predictions = prediction[truth == 1.3862943611198906]
iron_predictions = prediction[truth == 4.02535169073515]

proton_weights = np.ones_like(proton_predictions) / len(proton_predictions)
helium_weights = np.ones_like(helium_predictions) / len(helium_predictions)
lithium_weights = np.ones_like(lithium_predictions) / len(lithium_predictions)
beryllium_weights = np.ones_like(beryllium_predictions) / len(beryllium_predictions)
iron_weights = np.ones_like(iron_predictions) / len(iron_predictions)

bin_size = 0.1
min_edge = prediction.min()
max_edge = prediction.max()
bin_edges = np.arange(start=min_edge, stop=max_edge + bin_size, step=bin_size)


plt.figure(figsize=(10, 6))
plt.hist(proton_predictions, bins=bin_edges, weights=proton_weights, alpha=0.5, label='Predictions For Proton')
plt.hist(helium_predictions, bins=bin_edges, weights=helium_weights, alpha=0.5, label='Predictions For Helium', color='orange')
plt.hist(lithium_predictions, bins=bin_edges, weights=lithium_weights, alpha=0.5, label='Predictions For Lithium', color='purple')
plt.hist(beryllium_predictions, bins=bin_edges, weights=beryllium_weights, alpha=0.5, label='Predictions For Beryllium', color='green')
# plt.hist(iron_predictions, bins=bin_edges, weights=iron_weights, alpha=0.5, label='Predictions For Iron', color='brown')
plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
plt.xlabel('Predicted Value')
plt.ylabel('Percentage of Counts')
plt.title('Distribution of Predicted Values Per Element')
plt.legend()
plt.savefig(f'/home/zwang/cosmic-ray-nn/testing/(3) General Model [l: 0.1395, vl: 0.1252]/counts_vs_predicition.png')