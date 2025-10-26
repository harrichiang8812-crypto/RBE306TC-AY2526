import os
from pathlib import Path

path = "/data-shared/server01/data1/haochuan/CharacterRecords2025May-051/Images/GNRBN-20250512PF-Batch816-CheckAnimation-EncoderCvCvCvCv-MixerMaxMaxMaxMaxRes3@3/"

dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]


mark = "Train"
for style in dirs:
    currentStyleDir = os.path.join(path, style)
    files = [str(p) for p in Path(currentStyleDir).rglob("*.png") if mark in p.name]
