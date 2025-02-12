import kagglehub
import shutil

path = kagglehub.dataset_download("harrywang/acquired-podcast-transcripts-and-rag-evaluation")

target_path = "../custom_data/"

shutil.move(path, target_path)

print("Dataset moved to:", target_path)
