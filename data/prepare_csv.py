import os
import pandas as pd

gt_dir = "./datasets/groundTruth"
jerlov_dir = "./datasets/inputJerlov"
input_dir = "./datasets/input"

files = sorted(os.listdir(gt_dir))
jerlov_files = sorted(os.listdir(jerlov_dir))
input_files = sorted(os.listdir(input_dir))

data = list(zip(files, jerlov_files, input_files))
df = pd.DataFrame(data, columns=["groundTruth", "jerlov", "input"])
df.to_csv("./datasets/dataset.csv", index=False)

print("dataset.csv generated!")
