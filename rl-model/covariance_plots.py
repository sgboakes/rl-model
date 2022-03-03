import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dfs = {}
num_files = 5
for i in range(1, num_files+1):
    dfs[i] = pd.read_csv('Covariance-Returns-{i}.csv'.format(i=i))

iterations = dfs[1]['Iterations']
num_sats = 25
satVisCheck = {chr(i + 97): False for i in range(num_sats)}
cov_returns = {chr(i + 97): np.zeros((num_files, len(iterations))) for i in range(num_sats)}
cov_rnd_returns = {chr(i + 97): np.zeros((num_files, len(iterations))) for i in range(num_sats)}
for i in range(num_sats):
    c = chr(i+97)
    if 'Cov_{c}'.format(c=c) in dfs[1].columns:
        satVisCheck[c] = True

for i in range(num_sats):
    c = chr(i+97)
    if satVisCheck[c]:
        for df in dfs:
            cov_returns[c][df-1, :] = dfs[df]['Cov_{c}'.format(c=c)]
            cov_rnd_returns[c][df-1, :] = dfs[df]['Cov_rnd_{c}'.format(c=c)]

cov_mean = {chr(i + 97): np.zeros((1, len(iterations))) for i in range(num_sats)}
cov_std = {chr(i + 97): np.zeros((1, len(iterations))) for i in range(num_sats)}
cov_rnd_mean = {chr(i + 97): np.zeros((1, len(iterations))) for i in range(num_sats)}
cov_rnd_std = {chr(i + 97): np.zeros((1, len(iterations))) for i in range(num_sats)}

for i in range(num_sats):
    c = chr(i+97)
    if satVisCheck[c]:
        cov_mean[c] = np.mean(np.log(cov_returns[c]), axis=0)
        cov_std[c] = np.std(np.log(cov_returns[c]), axis=0)
        cov_rnd_mean[c] = np.mean(np.log(cov_rnd_returns[c]), axis=0)
        cov_rnd_std[c] = np.std(np.log(cov_rnd_returns[c]), axis=0)

plt.figure()
for i in range(num_sats):
    c = chr(i+97)
    if satVisCheck[c]:
        plt.plot(iterations, cov_mean[c], label='Policy Average')
        plt.fill_between(iterations, cov_mean[c]-cov_std[c], cov_mean[c]+cov_std[c], alpha=.1)
# plt.legend()
# plt.yscale('log')
plt.ylabel('ln(tr(P)), Agent Policy')
plt.xlabel('Iterations')
plt.show()

plt.figure()
for i in range(num_sats):
    c = chr(i+97)
    if satVisCheck[c]:
        plt.plot(iterations, cov_rnd_mean[c], label='Random Average')
        plt.fill_between(iterations, cov_rnd_mean[c]-cov_rnd_std[c], cov_rnd_mean[c]+cov_rnd_std[c], alpha=.1)
# plt.legend()
# plt.yscale('log')
plt.ylabel('$\ln(tr(P))$, Random Policy')
plt.xlabel('Iterations')
plt.show()
