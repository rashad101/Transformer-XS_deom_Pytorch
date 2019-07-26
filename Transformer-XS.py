import torch
from torch import nn
import os, re
import copy, time
import math, sys
import numpy as np
import subprocess
from data_utils import Dataset


MAX_LEN = 200

if torch.cuda.is_available():
    device = "cuda"
    print("Using GPU")
else:
    device = "cpu"
    print("Using CPU")


def download_data():
    if not os.path.isdir("data/wikitext-101/"):
        print("Downloading data...")
        subprocess.run("wget -c https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip -P data".split())
        print("Unzipping data...")
        subprocess.run(["unzip", "data/wikitext-103-v1.zip", "-d", "data/"])
        print("Done...")
    else:
        print("Found data...")


download_data()