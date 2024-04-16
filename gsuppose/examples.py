from numpy import loadtxt
from tifffile import imread
from pathlib import Path

def load_dataset(name: str):
    data_dir = Path(__file__).parent / "examples" / name
    sample = imread(data_dir / "sample.tif")
    psf = imread(data_dir / "psf.tif")
    initial_positions = loadtxt(data_dir / "npc_initial_positions.csv", dtype=float, delimiter=",")

    return sample, psf, initial_positions
