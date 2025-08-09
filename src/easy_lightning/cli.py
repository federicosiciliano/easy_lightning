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

    src = Path(__file__).parent / "ntb"
    dst = Path.cwd() / "ntb"
    if dst.exists():
        print("ntb folder was already in the current directory!")
    else:
        shutil.copytree(src, dst)
        print(f"'ntb' folder copied in: {dst}")
