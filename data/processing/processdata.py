import subprocess
from pathlib import Path
from tqdm import tqdm
import multiprocessing

def loaddata(fpath, t):
    # This function runs the Matlab file loaddata.m (on the same variables) in commandline

    print(f"Processing {fpath}/analysis/dfield/{t - 1:06d}.bin")

    # Create an output directory if it doesn't exist
    output_dir = Path(f"../rawdata/{Path(fpath).name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # The output file should match the file the Matlab code is reading
    output = output_dir / Path(f"{t - 1:06d}.mat")
    
    cmd = f"matlab -batch \"data = loaddata('{fpath}', {t}, 'dfield', 'float'); save('{str(output)}', 'data', '-v7.3')\""
    subprocess.run(cmd, shell=True, check=True)

def loadall(dir):
    # This function runs loaddata() on every file in the given directory

    subdirs = [d for d in Path(dir).iterdir() if d.is_dir()]
    for subdir in subdirs:
        files = [file for file in Path(f"{str(subdir)}/analysis/dfield").iterdir() if file.is_file() and file.suffix == ".bin"]
        nfiles = len(files)

        # The idea is we want to speed things up by multiprocessing each file.
        args = [(str(subdir), t) for t in range(1, nfiles + 1)]
        num_processes = min(multiprocessing.cpu_count(), len(files))

        # progress_bar = tqdm(total=nfiles, desc=f"Processing subdirectory {str(subdir)}", leave=True)        # I felt fancy today

        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.starmap(loaddata, args)
            # for _ in pool.starmap(loaddata, args):
            #     progress_bar.update(1)                                                                      # Felt very fancy

if __name__ == '__main__':
    loadall("/tigress/SHAEVITZ/kc32/Monolayerpaperdata/Data/")