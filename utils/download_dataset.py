import kagglehub
import shutil
import os

path = kagglehub.dataset_download("harrywang/acquired-podcast-transcripts-and-rag-evaluation")
target_path = "../custom_data/"
os.makedirs(target_path, exist_ok=True)
shutil.move(path, target_path)
print("Dataset moved to:", target_path)