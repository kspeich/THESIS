import numpy as np
from pathlib import Path
import h5py

def get_dfield(filename):
    with h5py.File(filename, 'r') as f:
        data = f['data'][:]                 # Extract as a numpy array

        # This should be a 2-dimensional array, with the ANGLE of the director field at each x and y point.
        # We extract the nx and ny values by taking the cosine and sine, respectively.
        nx = np.cos(data)
        ny = np.sin(data)

        return np.stack((nx, ny), axis=0)
    
def get_dimensions(dir):
    files = [file for file in Path(f"{str(dir)}").iterdir() if file.is_file() and file.suffix == ".mat"]
    timepoints = len(files)

    # Now we do length and width.  We want to make sure they are all the same among all timepoints
    shapes = []
    for file in files:
        with h5py.File(file, 'r') as f:
            shapes.append(f['data'].shape)

    if all(shape == shapes[0] for shape in shapes):
        return timepoints, shapes[0][0], shapes[0][1]
    else:
        print("Arrays have different dimensions")
        return None


def convert_to_hdf5(dir):
    # This function takes a subdirectory of ../rawdata/ and compiles it all into one HDF5 file

    # Create an output directory if it doesn't exist
    output_dir = Path("../processeddata")
    output_dir.mkdir(parents=True, exist_ok=True)

    timepoints, length, width = get_dimensions(dir)

    # Now we extract the data from the files
    director = np.zeros((timepoints, 2, length, width))
    for t in range(timepoints):
        director[t] = get_dfield(f"{dir}/{t:06d}.mat")
        
    with h5py.File(f"../processeddata/{Path(dir).name}.hdf5", 'w') as h5f:
        h5f.create_dataset('director', data=director)


def convertall(dir):
    subdirs = [d for d in Path(dir).iterdir() if d.is_dir()]
    for subdir in subdirs:
        print(f"Converting {str(subdir)}")
        convert_to_hdf5(subdir)

if __name__ == '__main__':
    convertall("../rawdata/")