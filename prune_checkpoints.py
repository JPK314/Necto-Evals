import os
import shutil

INPUT_DIR = f"{os.getcwd()}\\all_models"
OUTPUT_DIR = f"{os.getcwd()}\\models"
N_TOTAL_MODELS = 50

try:
    shutil.rmtree(OUTPUT_DIR)
except:
    pass
os.mkdir(OUTPUT_DIR)
    
max_idx = 0
files = os.listdir(INPUT_DIR)
for file in files:
    try:
        model_num = int(file.split("-")[-1][:-3])
    except:
        continue
    if model_num > max_idx:
        max_idx = model_num
step_size = max_idx // N_TOTAL_MODELS
for idx in range(N_TOTAL_MODELS):
    shutil.copy(f"{INPUT_DIR}\\necto-{idx * step_size}.pt", f"{OUTPUT_DIR}\\necto-{idx * step_size}.pt")