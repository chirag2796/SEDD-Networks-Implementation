import numpy as np
import matplotlib.pyplot as plt

mse = np.array([0.24282495, 0.26967125, 0.17451018, 0.16484879, 0.1299196, 0.11651003, 0.091982044, 0.08406749, 0.083898224, 0.07996701])
val_mse = np.array([0.17676295, 0.18766682, 0.18565785, 0.13967565, 0.11568855, 0.09314684, 0.07739338, 0.0856936, 0.07806716, 0.07135745])
plt.grid('on')
# summarize history for accuracy
plt.plot(mse, label='mean_squared_error', marker='o')
plt.plot(val_mse, label='val_mean_squared_error', marker='o')
plt.title('Decoder mean_squared_error')
plt.ylabel('mean_squared_error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()