import argparse
import matplotlib.pyplot as plt
import colorsys
import time
import random
from pathlib import Path
from matplotlib.animation import FuncAnimation
import matplotlib.image as mpimg
import numpy as np
from diffusion_policy.data.dataset import ZarrTrialDataset
import zarr
import pandas as pd
import re

"""
Define some colors
"""
# GREEN
h = 0.35
s = 0.82
v = 0.73
r, g, b = colorsys.hsv_to_rgb(h, s, v)
GREEN = ((r, g, b), 0.6)

# RED
h = 0.0
s = 0.9
v = 0.9
r, g, b = colorsys.hsv_to_rgb(h, s, v)
RED = ((r, g, b), 0.8)

YELLOW = ('yellow', 0.6)
BLACK = ('black', 0.6)

def get_args():
    parser = argparse.ArgumentParser(description="Parse input parameters for frame visualization.")

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the data directory."
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        help="Path to the save directory. Only used if --traj is true"
    )
    
    parser.add_argument(
        "--traj",
        action="store_true",
        help="If set, display a plot of trajectories instead."
    )
    
    parser.add_argument(
        "--elev",
        type=float,
        help="Starting elevation for trajectory display in 3d"
    )

    parser.add_argument(
        "--azim",
        type=float,
        help="Starting azimuth for trajectory display in 3d"
    )

    parser.add_argument(
        "--roll",
        type=float,
        help="Starting roll for trajectory display in 3d"
    )

    parser.add_argument(
        "--d2",
        action="store_true",
        help="Only does something if --traj is set. 2d if set and 3d otherwise."
    )

    parser.add_argument(
        "--trial_number",
        type=int,
        default=0,
        help="Trial number (default: 0)."
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=1,
        help="Frame frequency or step between frames (default: 20)."
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Number of actions per second of arm (default: 10)."
    )

    parser.add_argument(
        "--pred",
        type=int,
        default=12,
        help="Number of predicted frames (default: 12)."
    )

    parser.add_argument(
        "--obs",
        type=int,
        default=1,
        help="Number of observed frames (default: 1)."
    )

    parser.add_argument(
        "--action",
        type=int,
        default=6,
        help="Action length or duration (default: 6)."
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Size of each image frame (default: 224)."
    )

    parser.add_argument(
        "--cams",
        type=int,
        nargs='+',
        default=[0],
        help="List of camera indices to use (e.g., --cams 0 1). Default: [0]."
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for colors for trajectories"
    )

    parser.add_argument(
        "--color",
        type=str,
        default = 'green',
        help="Color of trajectories. Use 'var' for variable color depending on wrong block"
    )

    return parser.parse_args()

def view_trial(trial_number: int, dataset, interval, cams, fps):
    """
    Given a trial number and a ZarrTrialDataset instance, prints:
    1. the trial number
    2. pred_horizon, obs_horizon, action_horizon
    3. number of samples generated from this trial
    4. shapes of proprioceptive, action, and image data (from the first sample)
    5. Illustrates the first, second, middle, second-last, and last complete samples by
        plotting a composite image. For each sample, the observation frames (from frame 0 to
        frame obs_horizon) are concatenated horizontally; these rows are stacked vertically.
    """
    print("Trial Number:", trial_number)
    print("Prediction Horizon:", dataset.pred_horizon)
    print("Observation Horizon:", dataset.obs_horizon)
    print("Action Horizon:", dataset.action_horizon)

    # Filter indices for the given trial from dataset.sample_index.
    trial_sample_indices = [i for i, s in enumerate(dataset.sample_index) if s[0] == trial_number]
    num_samples = len(trial_sample_indices)
    print("Number of samples from this trial:", num_samples)

    if num_samples == 0:
        print("No samples found for trial", trial_number)
        return

    # Use one sample (the first one) to print data shapes.
    sample = dataset[trial_sample_indices[0]]
    print("Shape of agent_pos (proprio):", sample["agent_pos"].shape)
    print("Shape of action:", sample["action"].shape)
    print("Shape of image:", sample["image"].shape)

    selected = trial_sample_indices[::interval]

    def best_grid_square(n, target_aspect=2/3):
        best_diff = float('inf')
        best_shape = (1, n)

        for rows in range(1, n + 1):
            cols = int(np.ceil(n / rows))
            aspect = rows / cols
            diff = abs(aspect - target_aspect)
            if diff < best_diff:
                best_diff = diff
                best_shape = (rows, cols)

        return best_shape

    def stack_square_images(images):
        n, C, H, W = images.shape
        rows, cols = best_grid_square(n)

        grid_rows = []
        for i in range(rows):
            row_images = images[i * cols : (i + 1) * cols]
            if len(row_images) > 0:
                # Ensure `row_images` is a list of arrays before concatenating
                row_images_list = [row_images[j] for j in range(len(row_images))]
                
                # Stack images along width (axis=2)
                row = np.concatenate(row_images_list, axis=2)  # Shape: (C, H, W*cols)
                grid_rows.append(row)

        # Stack the rows along the height (axis=2)
        grid = np.concatenate(grid_rows, axis=1)  # Shape: (C, H*rows, W*cols)
        
        # Permute to (H*rows, W*cols, C)
        grid_permuted = grid.transpose(1, 2, 0)  # Shape: (H*rows, W*cols, C)
        return grid_permuted


    # Create a list to store each sample's composite observation image.
    rows = []
    for idx in selected:
        sample = dataset[idx]
        # sample["image"] has shape (obs_horizon, num_views, C, H, W).
        # We'll use the first view (index 0) for illustration.
        try:
            obs_imgs = sample["image"][:, cams, :, :, :]  # shape: (obs_horizon, C, H, W)
        except:
            raise Exception("Camera view not available")

        obs_horizon, views, C, H, W = obs_imgs.shape
        obs_imgs = obs_imgs.reshape(obs_horizon * views, C, H, W)
        rows.append(stack_square_images(obs_imgs))

    # Plot the composite image.
    fig, ax = plt.subplots()

    idx = [0]  # current frame index
    paused = [False]

    imshow_obj = ax.imshow(rows[0])
    ax.set_title(f"Frame 1/{len(rows)}")
    ax.axis("off")

    def update(frame):
        if not paused[0]:
            idx[0] = (idx[0] + 1) % len(rows)
            imshow_obj.set_data(rows[idx[0]])
            ax.set_title(f"Frame {idx[0] + 1}/{len(rows)}")
        return imshow_obj,

    def on_key(event):
        if event.key == ' ':
            paused[0] = not paused[0]  # toggle pause
        elif paused[0] and event.key in ['right', 'd']:
            idx[0] = (idx[0] + 1) % len(rows)
            imshow_obj.set_data(rows[idx[0]])
            ax.set_title(f"Frame {idx[0] + 1}/{len(rows)}")
            fig.canvas.draw()
        elif paused[0] and event.key in ['left', 'a']:
            idx[0] = (idx[0] - 1) % len(rows)
            imshow_obj.set_data(rows[idx[0]])
            ax.set_title(f"Frame {idx[0] + 1}/{len(rows)}")
            fig.canvas.draw()

    fig.canvas.mpl_connect('key_press_event', on_key)

    ani = FuncAnimation(fig, update, interval=int(1000/fps), blit=False)
    plt.show()

def read_data():
    path_name = "blocks_id{}.csv"
    wrong_blocks = {}
    wrong_blocks['choose_block'] = {}
    for i in range(2, 5):
        df = pd.read_csv(path_name.format(i))
        wrong_block = {}
        wrong_block['pilot'] = df[df['policy'] == 'Pilot'][['success', 'wrong block']].to_numpy()
        wrong_block['horizon'] = df[df['policy'] == 'NECL'][['success', 'wrong block']].to_numpy()
        wrong_block['baseline'] = df[df['policy'] == 'Baseline'][['success', 'wrong block']].to_numpy()
        wrong_blocks['choose_block'][i] = wrong_block

    return wrong_blocks

def get_color(root_dir, wrong_blocks, i):
    task = None
    id = None
    policy = None
    for path in [a for a in root_dir.split('/') if a != '']:
        match = re.match(r"id(\d+)", path)
        if match:
            id = match.group(1)
        if path in ['choose_block']:
            task = path
        if path in ['pilot', 'horizon', 'baseline']:
            policy = path

    if task is not None and id is not None and policy is not None and int(id) in [2, 3, 4]:
        if i >= len(wrong_blocks[task][int(id)][policy]):
            print(f"Warning: trial {i} does not exist in the CSV.")
            color = GREEN
        elif wrong_blocks[task][int(id)][policy][i][0]:
            color = GREEN
        elif wrong_blocks[task][int(id)][policy][i][1]:
            color = YELLOW
        else:
            color = GREEN
    else:
        color = YELLOW

    return color

def display_trajectories(root_dir, d2, save=False, elev=None, azim=None, roll=None, seed=None, save_dir=None, color=None):
    if seed is not None:
        random.seed(seed)

    root = Path(root_dir)
    # get paths ending in zarr
    trial_paths = sorted([p for p in root.iterdir() if p.suffix == ".zarr"])

    fig = plt.figure(figsize=(6, 6))

    # get the data for when there is a wrong block
    wrong_blocks = read_data()

    if d2:
        ax = fig.add_subplot(111)
    else:
        if save_dir is not None: # don't add background image if not saving
            # Load your image
            img = mpimg.imread('blocks.jpg')  # or .png, etc.

            # Check if the image has 3 channels (RGB)
            if img.ndim == 3 and img.shape[2] == 3:
                # Convert to grayscale by averaging the R, G, B channels
                img_gray = np.mean(img, axis=2)
            else:
                # If the image is already in grayscale, keep it as is
                img_gray = img

            # Add background image **as a separate Axes**
            bg_ax = fig.add_axes((0, 0, 1, 1), zorder=0)  # full canvas
            bg_ax.imshow(img_gray, cmap="gray")
            bg_ax.axis('off')  # no ticks, no frame

        ax = fig.add_subplot(111, projection='3d')

        if save_dir is not None: # don't preadjust axes if not saving
            # Transparent background of 3D plot
            ax.patch.set_alpha(0)

            # Transparent panes (the walls of the 3D box)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False

            ax.set_xlim(0.30, 0.57)
            ax.set_ylim(-0.17, 0.15)
            ax.set_zlim(-0.01, 0.23)

    for i, trial_path in enumerate(trial_paths):
        # Open the zarr group in read-only mode.
        store = zarr.DirectoryStore(str(trial_path))
        group = zarr.open_group(store, mode="r")
        proprio = group["proprio"]
        print(i, trial_path, len(proprio))


        color_v = None
        if color == 'var':
            color_v = get_color(root_dir, wrong_blocks, i)
        elif color == 'green':
            color_v = GREEN
        elif color == 'yellow':
            color_v = YELLOW
        elif color == 'red':
            color_v = RED
        elif color == 'black':
            color_v = BLACK
        else:
                color_v = GREEN
        if d2:
            ax.plot(
                proprio['y_pos'],
                proprio['x_pos'],
                color=color_v[0],
                alpha=color_v[1]
            )
            ax.invert_yaxis()
        else:
            ax.plot(
                proprio['x_pos'],
                proprio['y_pos'],
                proprio['z_pos'],
                color=color_v[0],
                alpha=color_v[1]
            )

    if not d2 and elev is not None and azim is not None and roll is not None:
        ax.view_init(elev=elev, azim=azim, roll=roll)

    if not save:
        plt.show()

        if d2:
            elev = None
            azim = None
            roll = None
        else:
            elev = float(ax.elev)  # Capture elevation angle
            azim = float(ax.azim)  # Capture azimuth angle
            roll = float(ax.roll)
            print(f"{elev = }, {azim = }, {roll = }")

        if save_dir != None:
            display_trajectories(root_dir, d2, True, elev, azim, roll, seed, save_dir, color)
    else:
        ax.set_axis_off()
        filename = '_'.join([a for a in root_dir.split('/') if a != ''][-3:])
        plt.savefig(Path(save_dir) / f"{filename}.svg", dpi=300, transparent=True, bbox_inches='tight')

def main():
    # Example parameters
    # root_dir = "/home/robomaster/data/zarr_test"  # Folder containing trial_1.zarr, trial_2.zarr, etc.
    args = get_args()
    root_dir = args.data_dir  # Folder containing trial_1.zarr, trial_2.zarr, etc.
    save_dir = args.save_dir
    traj = args.traj
    d2 = args.d2
    trial_number = args.trial_number
    interval = args.interval
    fps = args.fps
    cams = args.cams
    seed = args.seed if args.seed != None else time.time()
    elev = args.elev
    azim = args.azim
    roll = args.roll
    color = args.color

    pred_horizon = args.pred  # e.g., must be a multiple of chunk size
    obs_horizon = args.obs
    action_horizon = args.action
    image_size = args.image_size

    dataset = ZarrTrialDataset(root_dir, pred_horizon, obs_horizon, action_horizon, image_size=image_size)

    if traj:
        display_trajectories(root_dir, d2, False, elev, azim, roll, seed, save_dir, color)
    else:
        view_trial(trial_number, dataset, interval, cams, fps)

"""
Arguments
    --data_dir (str, required):
        Path to the data directory. This argument is required.

    --save_dir
        If set, saves a png image of the trajectory view (--traj must be used).

    --traj (option):
        If set, display only a 2d or 3d map of trajectories (see --d2).

    --d2 (option):
        If set together with --traj, display 2d plot, else 3d plot.

    --trial_number (int, default: 0):
        Trial number to visualize. Useful for selecting a specific run or experiment.

    --interval (int, default: 1):
        Sampling interval between frames. For example, an interval of 2 will show every other frame.

    --fps (int, default: 10):
        Playback speed, defined as the number of frames per second.

    --pred (int, default: 12):
        pred_horizon

    --obs (int, default: 6):
        obs_horizon (also controls number of images displayed in animation)

    --action (int, default: 6):
        action_horizon

    --image_size (int, default: 224):
        Height and width (in pixels) of each square image frame.

    --cams (List[int], default: [0]):
        List of camera indices to visualize. Specify one or more camera IDs, e.g., --cams 0 1.

    --seed (int):
        Seed for RNG for color display options.

Defaults mimic real-time playback (10 fps)
Images will automatically tile to as close to 2x3 as possible.
Space in playback will pause/unpause. Left and right arrow keys in pause will move frame by frame.

Example
# Display both views, first (0th) trial, and show only one observation at a time
python3 dataset.py --data_dir /home/palpatine/data/... --cams 0 1 --obs 1

# Show trajectory plot, saving output image in asdf.png and using random seed 0
python3 dataview.py --data_dir ~/Documents/ncel/data/ --traj --save_dir "asdf" --seed 0

# command for plotting
python3 dataview.py --data_dir ~/Documents/necl/inference_data/choose_block/id1/horizon/ --traj --save_dir imgs --elev 27 --azim 30 --roll 0
"""
if __name__ == "__main__":
    main()
