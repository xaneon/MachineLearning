#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from sklearn.cluster import KMeans


num_samples = 1000
sampling_rate = 10000 # Hz
fac = 100 # for amplitude scaling

f1 =h5py.File(os.path.join("data", "ephy", "data_40_cluster_0.h5")) # sky_cells.h5
data1 = f1["data"]
items1 = [item for item in data1.items()]
waveforms1 = data1["waveforms"]
spk_waveforms1 = [item for item in waveforms1.items()][0][1]
attributes1 = [item for item in spk_waveforms1.attrs.items()]
raw_data_dims1 = spk_waveforms1.shape
waveforms1 = np.ones(raw_data_dims1) * np.nan # empty data structure
spk_waveforms1.read_direct(waveforms1)
n_samples1, n_datapts1 = waveforms1.shape

f2 =h5py.File(os.path.join("data", "ephy", "data_40_cluster_1.h5")) # sky_cells.h5
data2 = f2["data"]
items2 = [item for item in data2.items()]
waveforms2 = data2["waveforms"]
spk_waveforms2 = [item for item in waveforms2.items()][0][1]
attributes2 = [item for item in spk_waveforms2.attrs.items()]
raw_data_dims2 = spk_waveforms2.shape
waveforms2 = np.ones(raw_data_dims2) * np.nan # empty data structure
spk_waveforms2.read_direct(waveforms2)
n_samples2, n_datapts2 = waveforms2.shape
waveforms1.shape, waveforms2.shape

waveforms = np.concatenate((waveforms1, waveforms2), axis=0)
n_samples, n_datapts = waveforms.shape
waveform_data = waveforms[np.random.permutation(num_samples), :] / fac
t_ms = np.linspace(0, n_datapts/sampling_rate, n_datapts) * 1e3
waveforms.shape

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(16,10))
_ = ax.plot(t_ms, waveform_data.T, color="k")
_ = ax.set_xlabel("Zeit in Millisekunden")
_ = ax.set_ylabel("Amplitude in Millivolt")

heights = waveforms.max(axis=1)
minima = waveforms.min(axis=1)
max_minus_min = heights - minima

df = pd.DataFrame({"heights": heights, "minima": minima, "max_minux_min": max_minus_min})
df.sample(5)

kmeans = KMeans(n_clusters=2, random_state=0).fit(df.values)
labels = kmeans.labels_
centres = kmeans.cluster_centers_

fig, ax = plt.subplots(ncols=4, nrows=1, figsize=(20, 8))
for idx, name in enumerate(df.columns):
    ax[idx].plot(df["heights"].values[labels==0], df[name].values[labels==0], color="b", marker=".", linestyle="", alpha=0.5)
    ax[idx].plot(df["heights"].values[labels==1], df[name].values[labels==1], color="r", marker=".", linestyle="", alpha=0.5)
    ax[idx].set_xlabel("height")
    ax[idx].set_ylabel(name)
_ = ax[idx+1].plot(t_ms, np.mean(waveforms[labels==0, :].T, axis=1) / fac, "b", label="cell 1")
_ = ax[idx+1].plot(t_ms, np.mean(waveforms[labels==1, :].T, axis=1) / fac, "r", label="cell 2")
ax[idx+1].set_xlabel("time [ms]")
ax[idx+1].set_xlabel("amplitude [mV]")
_ = plt.legend()


