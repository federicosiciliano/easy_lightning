import os
import shutil
from pathlib import Path

def init_project():
    src = Path(__file__).parent / "cfg"
    dst = Path.cwd() / "cfg"
    if dst.exists():
        print("cfg folder was already in the current directory!")
    else:
        shutil.copytree(src, dst)
        print(f"'cfg' folder copied in: {dst}")
