import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# ~~~~~~ Plot final covariances

dfs = {}
num_files = 5
num_sats = 25
for i in range(1, num_files+1):
    dfs[i] = pd.read_csv('Covariance-Fin-Returns-{i}.csv'.format(i=i))

satVisCheck = {chr(i + 97): False for i in range(num_sats)}
for i in range(num_sats):
    c = chr(i+97)
    if 'Cov_{c}'.format(c=c) in dfs[1].columns:
        satVisCheck[c] = True

steps = dfs[1]['Steps']
cov_returns = {chr(i + 97): np.zeros((num_files, len(steps))) for i in range(num_sats)}
cov_rnd_returns = {chr(i + 97): np.zeros((num_files, len(steps))) for i in range(num_sats)}

for i in range(num_sats):
    c = chr(i+97)
    if 'Cov_{c}'.format(c=c) in dfs[1].columns:
        satVisCheck[c] = True

for i in range(num_sats):
    c = chr(i+97)
    if satVisCheck[c]:
        for df in dfs:
            cov_returns[c][df-1, :] = dfs[df]['Cov_{c}'.format(c=c)]
            cov_rnd_returns[c][df-1, :] = dfs[df]['Cov_Rnd_{c}'.format(c=c)]

cov_mean = {chr(i + 97): np.zeros((1, len(steps))) for i in range(num_sats)}
cov_std = {chr(i + 97): np.zeros((1, len(steps))) for i in range(num_sats)}
cov_rnd_mean = {chr(i + 97): np.zeros((1, len(steps))) for i in range(num_sats)}
cov_rnd_std = {chr(i + 97): np.zeros((1, len(steps))) for i in range(num_sats)}

for i in range(num_sats):
    c = chr(i+97)
    if satVisCheck[c]:
        cov_mean[c] = np.mean(np.log(cov_returns[c]), axis=0)
        cov_std[c] = np.std(np.log(cov_returns[c]), axis=0)
        cov_rnd_mean[c] = np.mean(np.log(cov_rnd_returns[c]), axis=0)
        cov_rnd_std[c] = np.std(np.log(cov_rnd_returns[c]), axis=0)

plt.figure()
for i in range(num_sats):
    c = chr(i + 97)
    if satVisCheck[c]:
        plt.plot(steps, cov_mean[c])
        plt.fill_between(steps, cov_mean[c]-cov_std[c], cov_mean[c]+cov_std[c], alpha=.1)
plt.xlabel('Step Number')
plt.ylabel('Tr(P), Agent Policy')
# plt.yscale('log')
plt.show()

plt.figure()
for i in range(num_sats):
    c = chr(i + 97)
    if satVisCheck[c]:
        plt.plot(steps, cov_rnd_mean[c])
        plt.fill_between(steps, cov_rnd_mean[c]-cov_rnd_std[c], cov_rnd_mean[c]+cov_rnd_std[c], alpha=.1)
plt.xlabel('Step Number')
plt.ylabel('Tr(P), Random Policy')
# plt.yscale('log')
plt.show()
