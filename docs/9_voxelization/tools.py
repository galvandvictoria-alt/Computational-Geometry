
import shutil
import trimesh
import os
import numpy as np
import subprocess
from os import listdir
from pathlib import Path

def convert_binvox_to_numpy(file_path):
    """Load a binvox file and convert it to a numpy array.
    
    Args:
        file_path: Path to the binvox file to load
        
    Returns:
        Numpy array representing the voxel grid (0s and 1s)
    """
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return

    try:
        voxels = trimesh.load(file_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    matrix_int = voxels.encoding.dense.astype(int)
    print(f"Voxel grid dimensions: {matrix_int.shape}")
    return matrix_int

def process_voxel(input_path, output_path, resolution=128):
    """Convert a 3D model to a voxel representation using binvox.
    
    Args:
        input_path: Path to the input 3D model file (STL, OBJ, etc.)
        output_path: Directory where the binvox file will be saved
        resolution: Voxel grid resolution (default: 128)
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    output_path.mkdir(parents=True, exist_ok=True)
    command = ["./binvox", "-e", "-d", str(resolution), str(input_path)]
    
    print(f"Processing: {input_path.name}...")
    
    try:
        subprocess.run(command, check=True, text=True)
        generated_file = input_path.with_suffix(".binvox")
        
        if generated_file.exists():
            final_path = output_path / generated_file.name
            shutil.move(str(generated_file), str(final_path))
            print(f"Saved at: {final_path}")
        else:
            print("Error: Generated output file not found.")
            
    except subprocess.CalledProcessError as e:
        print(f"Error executing binvox: {e}")
    except Exception as e:
        print(f"Unexpected error occurred: {e}")

def export_to_obj(points, filename):
    """Export an array of Nx3 points to OBJ format.
    
    Args:
        points: Nx3 numpy array containing point coordinates
        filename: Output filename for the OBJ file
    """
    print(f"\nExporting {len(points)} points to {filename}...")
    
    with open(filename, 'w') as f:
        # Write a simple header (optional)
        f.write(f"# Aligned object generated from numpy\n")
        
        # Iterate over each point and write it with 'v' prefix
        for point in points:
            f.write(f"v {point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
            
    print("Export completed!")