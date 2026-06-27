import os

import kagglehub

path = kagglehub.dataset_download("jessicali9530/lfw-dataset")
print("Path to dataset files:", path)

# Let's list the files to see what we have
for root, dirs, files in os.walk(path):
    print(f"{root}: {len(files)} files")
    for d in dirs:
        print(f"  Dir: {d}")
    if len(files) < 10:
        for f in files:
            print(f"  File: {f}")
    break
