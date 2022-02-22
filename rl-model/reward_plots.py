import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('Results.csv')

results_array = [df['Returns1'], df['Returns2'], df['Returns3']]
rnd_results_array = [df['Rnd_Returns1'], df['Rnd_Returns2'], df['Rnd_Returns3']]

returns_mean = np.mean(results_array, axis=0)
rnd_returns_mean = np.mean(rnd_results_array, axis=0)

returns_std = np.std(results_array, axis=0)
rnd_returns_std = np.std(rnd_results_array, axis=0)

plt.plot(df['Iterations'], returns_mean, label='DDQN Policy')
plt.fill_between(df['Iterations'], returns_mean-returns_std, returns_mean+returns_std, alpha=.1)
plt.plot(df['Iterations'], rnd_returns_mean, label='Random Policy')
plt.fill_between(df['Iterations'], rnd_returns_mean-rnd_returns_std, rnd_returns_mean+rnd_returns_std, alpha=.1)
plt.legend()
plt.ylabel('Average Return')
plt.xlabel('Iterations')
plt.show()
