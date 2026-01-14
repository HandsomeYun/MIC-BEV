import os
import numpy as np

# Root directory where your maps are stored
root_dir = '/home/yun/MIC-BEV_Official/data/map'

# Traverse all subdirectories
for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith('.npy'):
            file_path = os.path.join(dirpath, filename)
            try:
                # Load the .npy file
                data = np.load(file_path)
                
                # Flip vertically along Y-axis (axis=0)
                flipped = np.flipud(data)
                
                # Save it back (overwrite)
                np.save(file_path, flipped)
                print(f"Flipped: {file_path}")
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")
