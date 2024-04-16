import numpy as np
import matplotlib
from matplotlib import pyplot as plt

history = np.loadtxt('best_model_training_history.txt')
matplotlib.rcParams['font.weight'] = 'bold'

loss = history[:,0]
val_loss = history[:,1]

plt.rcParams['axes.labelweight'] = 'bold'  # Make axis labels bold
plt.rcParams['xtick.labelsize'] = 12  # Set x-axis tick label size
plt.rcParams['ytick.labelsize'] = 12  # Set y-axis tick label size
plt.rcParams['font.weight'] = 'bold'  # Make tick labels bold

fig, ax = plt.subplots(1, figsize=(8,5))

n = np.arange(len(loss))

ax.plot(n, val_loss, label='val_loss',  color = 'red', linewidth=1)
ax.plot(n, loss, ls='--',  label='loss', color = 'black',linewidth=1)
ax.set_xlabel('Epoch', fontsize=15, fontweight='bold')
ax.set_ylabel('Loss', fontsize=15, fontweight='bold')
ax.set_ylim(0.05, 1)
ax.legend()
ax.semilogy()
ax.grid()
plt.title("Transformer Training and Validation Loss", fontsize=18, fontweight='bold')
plt.savefig('/home/zwang/cosmic-ray-nn/Training_Curve.png', dpi = 1000)