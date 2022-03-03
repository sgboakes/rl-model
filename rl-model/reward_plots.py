import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dfs = {}
num_files = 5
for i in range(1, num_files+1):
    dfs[i] = pd.read_csv('Reward-Returns-{i}.csv'.format(i=i))

iterations = dfs[1]['Iterations']

results_array = np.zeros((num_files, len(iterations)))
rnd_results_array = np.zeros((num_files, len(iterations)))
for df in dfs:
    results_array[df-1, :] = (dfs[df]['Returns'])
    rnd_results_array[df-1, :] = (dfs[df]['Rnd_Returns'])

returns_mean = np.mean(results_array, axis=0)
rnd_returns_mean = np.mean(rnd_results_array, axis=0)

returns_std = np.std(results_array, axis=0)
rnd_returns_std = np.std(rnd_results_array, axis=0)

plt.plot(iterations, returns_mean, label='DDQN Policy')
plt.fill_between(iterations, returns_mean-returns_std, returns_mean+returns_std, alpha=.1)
plt.plot(iterations, rnd_returns_mean, label='Random Policy')
plt.fill_between(iterations, rnd_returns_mean-rnd_returns_std, rnd_returns_mean+rnd_returns_std, alpha=.1)
plt.legend()
plt.ylabel('Average Return')
plt.xlabel('Iterations')
plt.show()
