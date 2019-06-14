from __future__ import print_function
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
import os


parser = argparse.ArgumentParser(description="Compute average and standard deviation of timeseries.")
parser.add_argument("file", type=str, help="Timeseries file")
args = parser.parse_args()

data = np.loadtxt(args.file)
t = data[:, 0]
x = data[:, 1]

print("Average:       ", x.mean())
print("Standard dev.: ", x.std())

if data.shape[1] > 2:
    y = data[:, 2]
    print("Average2:      ", y.mean())
    print("Standard dev.2:", y.std())
