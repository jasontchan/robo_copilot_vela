# data_loader.py
import io

import matplotlib.pyplot as plt
import zarr
from PIL import Image


def load_zarr_data(zarr_store_dir):
    """
    Opens the unified Zarr store located at zarr_store_dir and returns a dictionary
    with three datasets: 'images', 'proprio', and 'actions'.
    """
    store = zarr.DirectoryStore(zarr_store_dir)
    root = zarr.open_group(store, mode="r")

    data = {
        "images": root["images"],
        "proprio": root["proprio"],
        "actions": root["actions"],
    }
    return data


def display_frame(frame_number, data):
    """
    Given a frame number and the loaded data dictionary (from load_zarr_data),
    prints the corresponding proprioceptive and action data, decodes the JPEG images,
    and displays them using matplotlib.
    """
    # Get the structured data for the frame.
    proprio_entry = data["proprio"][frame_number]
    action_entry = data["actions"][frame_number]

    print(f"Frame: {frame_number}")
    print("Proprioceptive data:")
    for key in proprio_entry.dtype.names:
        print(f"  {key}: {proprio_entry[key]}")
    print("Action data:")
    for key in action_entry.dtype.names:
        print(f"  {key}: {action_entry[key]}")

    # Retrieve the JPEG-encoded images for all cameras.
    images = data["images"][frame_number]  # Expecting a list of JPEG bytes.
    n_cameras = len(images)

    # Create a figure to display the images side by side.
    plt.figure(figsize=(5 * n_cameras, 5))
    for i in range(n_cameras):
        jpeg_bytes = images[i]
        # Decode JPEG bytes using PIL.
        img = Image.open(io.BytesIO(jpeg_bytes))
        ax = plt.subplot(1, n_cameras, i + 1)
        ax.imshow(img)
        ax.set_title(f"Camera {i}")
        ax.axis("off")
    plt.show()


# Example usage:
if __name__ == "__main__":
    # Set the path to your Zarr store (e.g., the nested directory under your save_dir).
    zarr_store_path = "/home/robomaster/data/choose_block_rand_init_andy/2025_04_18_14_04_59.zarr"  # Adjust this path.
    # zarr_store_path = "/home/robomaster/inference_data/drawers/test/baseline/2025_08_14_17_59_42.zarr"
    # zarr_store_path = "/home/robomaster/inference_data/choose_block/test/baseline/2025_08_14_18_21_36.zarr"
    # Load the data.
    data = load_zarr_data(zarr_store_path)

    # Display a specific frame (e.g., frame number 10).
    frame_to_display = 1
    display_frame(frame_to_display, data)
