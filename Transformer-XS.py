import torch
from torch import nn
import os, re
import copy, time
import math, sys
import spacy
import numpy as np
import subprocess
from ww import f
from tqdm import tqdm
from collections import defaultdict
from data_utils import Dataset


MAX_LEN = 200
en = spacy.load("en_core_web_sm")
data_path = 'data/wikitext-103/'
train_lines = 1801350
test_lines = 4358
valid_lines = 3760
vocab = defaultdict(int)
split = 'train'
L = eval(f'{split}_lines')


if torch.cuda.is_available():
    device = "cuda"
    print("Using GPU")
else:
    device = "cpu"
    print("Using CPU")


def download_data():
    if not os.path.isdir("data/wikitext-103/"):
        print("Downloading data...")
        subprocess.run("wget -c https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip -P data".split())
        print("Unzipping data...")
        subprocess.run(["unzip", "data/wikitext-103-v1.zip", "-d", "data/"])
        print("Done...")
    else:
        print("Found data...")


def create_vocab():
    with open(data_path + f'wiki.{split}.tokens') as f:
        _progress = 0
        buffer = []
        for line in tqdm(f, total=L):
            line = buffer.append(line.strip())
            if len(buffer) > 40000:
                buffer = ' '.join(buffer)
                tokens = list(en.tokenizer(buffer.lower()))
                buffer = []
                for w in tokens:
                    vocab[w.text] += 1

        # One last time to clean the buffer
        buffer = ' '.join(buffer)
        tokens = list(en.tokenizer(buffer.lower()))
        buffer = []
        for w in tokens:
            vocab[w.text] += 1


download_data()
#create_vocab()

lang_file = "./models/wiki103.large.lang"
if not os.path.isfile(lang_file):
    print("Creating vocab file...")
    en_lang = Lang('wiki')
    en_lang.buildLang(open(data_path + f'wiki.{split}.tokens'), num_lines=train_lines)
    with open(lang_file, 'wb') as f:
        pickle.dump(f, en_lang)
else:
    print("Loading vocab file...")
    en_lang = pickle.load(open('./models/wiki103.large.lang', 'rb'))