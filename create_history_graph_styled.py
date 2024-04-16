import numpy as np
import matplotlib
from matplotlib import pyplot as plt

history = np.loadtxt('best_model_training_history.txt')
matplotlib.rcParams['font.weight'] = 'bold'

loss = history[:,0]
val_loss = history[:,1]

fig, ax = plt.subplots(1, figsize=(9, 9))

n = np.arange(len(loss))

ax.plot(n, loss, ls='--',  label='loss', color='C0', linewidth=2, zorder=1)
ax.plot(n, val_loss, label='val. loss',  color='C1', linewidth=2, zorder=0)

ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
ax.semilogy()
ax.grid()
fig.tight_layout()
ax.set_box_aspect(1)
fig.savefig('Training_Curve_Styled.pdf', dpi = 1000)
