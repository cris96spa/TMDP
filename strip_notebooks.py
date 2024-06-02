import os
import subprocess

# Directory containing the Jupyter notebooks
directory = "."

# Iterate over all files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".ipynb"):
        filepath = os.path.join(directory, filename)
        try:
            # Run the nbstripout command on each notebook file
            subprocess.run(['nbstripout', filepath], check=True)
            print(f"Stripped output from {filename}")
        except subprocess.CalledProcessError as e:
            print(f"Error stripping {filename}: {e}")

print("Completed stripping outputs from all notebooks.")
