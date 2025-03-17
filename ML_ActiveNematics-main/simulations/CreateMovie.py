import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py
from pathlib import Path
from tqdm import tqdm
import multiprocessing


def create_movie(filename):
    with h5py.File(filename, "r") as h5f:
        # For the HDF5 files that we have, we should get only the ['director'] key if we run print(list(h5f.keys()))
        # We can now read this into a numpy array, as follows:
        director = h5f['director']

        # The dimensions of the director array are as follows:
        time_points = director.shape[0]
        height = director.shape[2]
        width = director.shape[3]
        # director.shape[1] is always 2, by construction, to allow for both nx and ny values

        fig, ax = plt.subplots(figsize=(15, 15))
        x, y = np.meshgrid(np.arange(height), np.arange(width))

        # Just for convenience, a progress bar
        process_id = multiprocessing.current_process()._identity[0] - 1     # This is just so that the progress bars display separately for each file
        progress_bar = tqdm(total=time_points, desc=f"Animating {filename}", leave=True, position=process_id)

        def update(t):
            # print(f"t = {t}")

            ax.clear()

            # Then, we can reset the streamplot with the appropriate nx and ny values
            u = director[t][0]      # nx
            v = director[t][1]      # ny

            # Update stream
            ax.streamplot(x, y, u, v, density=5, linewidth=0.5, arrowsize=1e-10, color='b')
            ax.set_title(f"t = {t}")

            progress_bar.update(1)

        ani = animation.FuncAnimation(fig, update, frames = range(time_points), blit=False)     # Creates the animation

        # Make the output directory (if it doesn't already exist), and save the file in the output directory
        output_dir = Path(filename).parent.name
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        ani.save(Path(output_dir) / (Path(filename).stem + ".mp4"), writer='ffmpeg', fps=1)

        plt.close(fig)
        progress_bar.close()


def convert_folder_to_movies(dir):
    # Takes every HDF5 in the directory dir and converts it to movie.  This takes a while so we will multiprocess.
    files = [file for file in Path(dir).iterdir() if file.is_file() and file.suffix == ".hdf5"]
    num_processes = min(multiprocessing.cpu_count(), len(files))

    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(create_movie, files)


if __name__ == '__main__':
    convert_folder_to_movies("../data/activity_assay")
    convert_folder_to_movies("../data/elastic_activity_assay")