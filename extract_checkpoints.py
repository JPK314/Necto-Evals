import os
import shutil

INITIAL_DIR = "C:\\Users\\User\Downloads\\ppos-20230820T144148Z-001\\ppos"
OUTPUT_DIR = f"{os.getcwd()}\\models"

try:
    os.mkdir(OUTPUT_DIR)
except:
    pass
idx = 0
for root, dirs, files in os.walk(INITIAL_DIR):
    if "checkpoint.pt" in files:
        shutil.copy(f"{root}\\checkpoint.pt", f"{OUTPUT_DIR}\\necto-{idx}.pt")
        idx += 1
        print(idx)