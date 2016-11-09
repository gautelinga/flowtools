import pandas as pd
import argparse
import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Turn csv file into hdf5.")
    parser.add_argument("csv_file", type=str, help="CSV file")
    parser.add_argument("hdf_file", type=str, help="HDF5 file")
    args = parser.parse_args()
    pts = np.asarray(pd.read_csv(args.csv_file, sep="\t", header=None))
    with h5py.File(args.hdf_file, "w") as h5f:
        h5f.create_dataset("x", data=pts)


if __name__ == "__main__":
    main()
