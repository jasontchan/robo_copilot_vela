import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import zarr
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser(description="Parse input parameters for frame visualization.")

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the data directory."
    )

    parser.add_argument(
        "--latex",
        action="store_true",
        help="Print latex output"
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Number of actions per second of arm"
    )
    
    return parser.parse_args()

def find_var(pos, fps):
    vel = np.diff(pos, axis=0) * fps
    accel = np.diff(vel, axis=0) * fps
    jerk = np.diff(accel, axis=0) * fps

    return np.linalg.norm(np.cov(vel.T), ord='fro'), \
           np.linalg.norm(np.cov(accel.T), ord='fro'), \
           np.linalg.norm(np.cov(jerk.T), ord='fro'), \
           np.cov(vel.T).trace(), \
           np.cov(accel.T).trace(), \
           np.cov(jerk.T).trace(), \


def main():
    args = get_args()
    root_dir = Path(args.data_dir)
    fps = args.fps
    latex = args.latex

    zarr_paths = {}

    # Recursively search directory for folders with .zarr files
    def find_zarr_paths(path):
        trial_paths = sorted([p.name for p in (root_dir / path).iterdir() if p.suffix == ".zarr"])
        other_paths = sorted([p.name for p in (root_dir / path).iterdir() if p.suffix != ".zarr" and
            p.is_dir()])

        if trial_paths != []:
            zarr_paths[path] = trial_paths
        else:
            for p in other_paths:
                find_zarr_paths(path / p)

    find_zarr_paths(Path())

    overall = pd.DataFrame({
        "Accel (Frob Norm)": [],
        "Jerk (Frob Norm)": [],
        "Accel (Trace)": [],
        "Jerk (Trace)": []
    })
    for p in zarr_paths.keys():
        print(f"*** {p} ***")
        data = []
        for i, trial_path in enumerate(zarr_paths[p]):
            store = zarr.DirectoryStore(root_dir / p / trial_path)
            group = zarr.open_group(store, mode="r")
            proprio = group["proprio"]
            path = np.stack((proprio['x_pos'], proprio['y_pos'], proprio['z_pos']), axis=1)
            vel_norm, accel_norm, jerk_norm, vel_tr, accel_tr, jerk_tr = find_var(path, fps)

            data += [(
                trial_path,
                accel_norm,
                jerk_norm,
                accel_tr,
                jerk_tr
            )]

        result = pd.DataFrame({
            "Trial Path": [r[0] for r in data],
            "Accel (Frob Norm)": [r[1] for r in data],
            "Jerk (Frob Norm)": [r[2] for r in data],
            "Accel (Trace)": [r[3] for r in data],
            "Jerk (Trace)": [r[4] for r in data]
        })

        # Create a new row for averages (use a placeholder for the 'trial_path')
        # Calculate the mean of each column, excluding the 'Name' column
        mean_row = result.drop(columns="Trial Path").mean()

        overall.loc[p] = mean_row

        # Add the 'Average' name for the row
        mean_row["Trial Path"] = "Average"

        result = result.round(3)

        # Append the mean values as a new row
        result.loc["Average"] = mean_row

        if latex:
            print(result.to_latex(index=False, float_format="%.3f"))
        else:
            print(result.to_string(index=False, float_format="%.3f"))
        print()

    if latex:
        print(overall.to_latex(float_format="%.3f"))
    else:
        print(overall.to_string(float_format="%.3f"))

"""
usage: path_analysis.py [-h] --data_dir DATA_DIR

Parse input parameters for frame visualization.

options:
  -h, --help           show this help message and exit
  --data_dir DATA_DIR  Path to the data directory.

Example:
# Get metrics for trajectories stored in specified directory
python3 path_analysis.py --data_dir ~/Documents/ncel/data
"""
if __name__ == "__main__":
    main()
