import scipy.io
import pandas as pd
import numpy as np
import os
from scipy.sparse import csc_matrix

# Define the path to the .mat file and the output file
mat_file_path = os.path.join('data', 'raw', 'synthetic_1.0.mat')
output_file_path = os.path.join('data', 'processed', 'synthetic_1.0.txt')

# Load the .mat file
mat_data = scipy.io.loadmat(mat_file_path)

# Function to extract information by key
def extract_key_info(key, mat_data):
    info = f"Key: '{key}'\n"
    data = mat_data[key]
    info += f"Type: {type(data)}\n"
    
    if isinstance(data, np.ndarray):
        info += f"Shape: {data.shape}\n"
        if data.ndim == 2:
            df = pd.DataFrame(data)
            info += "Summary statistics:\n"
            info += df.describe().to_string() + "\n"
        elif data.ndim == 1:
            info += f"Data: {data}\n"
    elif isinstance(data, csc_matrix):
        info += f"Sparse matrix with shape: {data.shape}\n"
        info += f"Data: {data}\n"
    else:
        info += f"Data: {data}\n"
    
    info += "\n"
    return info

# Extract information for each key
all_info = "MAT File Keys and Data Information\n"
all_info += "=" * 40 + "\n\n"
for key in mat_data.keys():
    if key not in ['__header__', '__version__', '__globals__']:
        all_info += extract_key_info(key, mat_data)

# Save the information to a file
with open(output_file_path, 'w') as f:
    f.write(all_info)

print(f"Data information saved to {output_file_path}")
