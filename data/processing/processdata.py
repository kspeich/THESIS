import subprocess
from pathlib import Path

def loaddata(fpath, t, dataname, datatype):
    # This function runs the Matlab file loaddata.m (on the same variables) in commandline

    # Create an output directory if it doesn't exist
    output_dir = Path(f"../rawdata/{Path(fpath).name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # The output file should match the file the Matlab code is reading
    output = output_dir / Path(f"{t - 1:06d}.mat")

    print(f"Processing {fpath}/{t - 1:06d}.bin")
    
    cmd = f"matlab -batch \"data = loaddata('{fpath}', {t}, '{dataname}', '{datatype}'); save('{str(output)}', 'data')\""
    subprocess.run(cmd, shell=True, check=True)

if __name__ == '__main__':
    dir = "/tigress/SHAEVITZ/kc32/Monolayerpaperdata/Data/"
    subdirs = [d for d in Path(dir).iterdir() if d.is_dir()]

    dataname = "dfield"
    datatype = "float"
    
    for subdir in subdirs:
        nfiles = len([file for file in Path(f"{str(subdir)}/analysis/{dataname}").iterdir() if file.is_file() and file.suffix == ".bin"])

        for t in range(1, nfiles + 1):
            loaddata(str(subdir), t, dataname, datatype)