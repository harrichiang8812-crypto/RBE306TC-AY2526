# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import os
import glob

import imageio
import scipy.misc as misc
import numpy as np
import copy as cp

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('agg')
import pylab
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import cv2


import os
import random
import numpy as np
import torch
from PIL import Image
import logging
from datetime import datetime

from collections import defaultdict

from pathlib import Path

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from typing import Union, Sequence, Tuple


import math 

GRAYSCALE_AVG = 127.5

LABEL_LENGTH=6
class Logging(logging.StreamHandler):
    """
    Custom StreamHandler that avoids adding a newline to logging messages.
    """
    def emit(self, record):
        try:
            # Ensure the record's message is a string
            msg = self.format(record)
            if not isinstance(msg, str):
                msg = str(msg)

            stream = self.stream
            if not getattr(self, 'terminator', '\n'):  # If terminator is set to empty
                stream.write(msg)
            else:
                stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


def PrintInfoLog(handler, message, end='\n', dispTime=True):
    handler.terminator = end  # Set the terminator
    
    # Get the current date and time with full details
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Log message with full timestamp if it's a new line, else continue the same line
    if dispTime and end == '\n':
        logging.info(f"[{current_time}] {str(message)}")
    else:
        logging.info(str(message))



    
def FindKeys(dict, val): return list(value for key, value in dict.items() if key in val)

def SplitName(inputStr):
    result = []
    temp = ""
    for i in inputStr:
        if i.isupper() :
            if temp != "":
                result.append(temp)
            temp = i
        else:
            temp = temp+ i

    if temp != "":
        result.append(temp)
    return result

def write_to_file(path,write_list):
    file_handle = open(path,'w')
    for write_info in write_list:
        file_handle.write(str(write_info))
        file_handle.write('\n')
    file_handle.close()
    print("Write to File: %s" % path)

def read_from_file(path):
    # get label0 for the targeted content input txt
    output_list = list()
    with open(path) as f:
        for line in f:
            this_label = line[:-1]
            if len(this_label)<LABEL_LENGTH and not this_label == '-1':
                    for jj in range(LABEL_LENGTH-len(this_label)):
                        this_label = '0'+ this_label
            # line = u"%s" % line
            output_list.append(this_label)


    return output_list

def read_file_to_dict(file_path):
    line_dict = {}
    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file):
            line = int(line)
            line = str(line)
            # 移除每行末尾的换行符并存储到字典
            line_dict[line] = line_number
    return line_dict


def image_show(img):
    img_out = cp.deepcopy(img)
    img_out = np.squeeze(img_out)
    img_shapes=img_out.shape
    if len(img_shapes)==2:
        curt_channel_img = img_out
        min_v = np.min(curt_channel_img)
        curt_channel_img = curt_channel_img - min_v
        max_v = np.max(curt_channel_img)
        curt_channel_img = curt_channel_img/ np.float32(max_v)
        img_out = curt_channel_img*255
    elif img_shapes[2] == 3:
        channel_num = img_shapes[2]
        for ii in range(channel_num):
            curt_channel_img = img[:,:,ii]
            min_v = np.min(curt_channel_img)
            curt_channel_img = curt_channel_img - min_v
            max_v = np.max(curt_channel_img)
            curt_channel_img = curt_channel_img / np.float32(max_v)
            img_out[:,:,ii] = curt_channel_img*255
    else:
        print("Channel Number is INCORRECT:%d" % img_shapes[2])
    plt.imshow(np.float32(img_out)/255)
    pylab.show()

def compile_frames_to_gif(frame_dir, gif_file):
    frames = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
    print(frames)
    images = [misc.imresize(imageio.imread(f), interp='nearest', size=0.33) for f in frames]
    imageio.mimsave(gif_file, images, duration=0.1)
    return gif_file





def correct_ckpt_path(real_dir,maybe_path):
    maybe_path_dir = str(os.path.split(os.path.realpath(maybe_path))[0])
    if not maybe_path_dir == real_dir:
        return os.path.join(real_dir,str(os.path.split(os.path.realpath(maybe_path))[1]))
    else:
        return maybe_path



def softmax(x):
    x = x-np.max(x, axis= 1, keepdims=True)
    f_x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    return f_x


def create_if_not(path):
    #create path if not exist
    if not os.path.exists(path):
        os.makedirs(path)
    

    

def cv2torch(file_path,transform):
    return transform(Image.open(file_path).convert('L'))

def string2tensor(string):
    return torch.tensor(int(string))

# def set_random(seed_id=1234):
#     #set random seed for reproduce
#     random.seed(seed_id)
#     np.random.seed(seed_id)
#     torch.manual_seed(seed_id)   #for cpu
#     torch.cuda.manual_seed_all(seed_id) #for GPU
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = True

def read_file_to_dict(file_path):
    line_dict = {}
    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file):
            line = int(line)
            line = str(line)
            # 移除每行末尾的换行符并存储到字典
            line_dict[line] = line_number
    return line_dict

def unormalize(tensor):
    # 反归一化操作
    tensor = tensor * 0.5 + 0.5  # 将 [-1, 1] 范围的值映射回 [0, 1]
    # 转换为 numpy 数组并且确保类型为 uint8
    output = tensor.int()
    return output

def MergeAllDictKeys(dict_list):
    merged = defaultdict(list)
    for d in dict_list:
        for k, v in d.items():
            merged[k].append(v)
    
    return dict(merged)



def GetChars(path):
    chars = list()
    with open(path) as f:
        for line in f:

            line = u"%s" % line
            char_counter = 0
            for char in line:

                current_char = line[char_counter]
                chars.append(current_char)



                char_counter += 1
    return chars


def GB2312CharMapper(targetTxtPath: str,
                     level1Path: str = "../YamlLists/GB2312-L1.txt",
                     level2Path: str = "../YamlLists/GB2312-L2.txt"):
    """
    One‑stop routine that
      • loads GB 2312 level‑1 and level‑2 tables,
      • builds a char → label‑0 look‑up,
      • scans *targetTxtPath* and returns the mapped label list (+ char list).

    Returns
    -------
    labelList : list[str]
        Concatenated label‑0 codes (e.g. "176161") in file order.
    charList  : list[str]
        The corresponding characters, 1‑to‑1 with *labelList*.
    """
    # ── 1. read both GB2312 level files ───────────────────────────────
    fullCharList, fullLabelList = [], []
    for txtPath, zoneOffset in ((level1Path, 16), (level2Path, 56)):
        rawChars = [c for line in Path(txtPath).read_text(encoding="utf‑8").splitlines()
                    for c in line.strip()]
        startZone = 160 + zoneOffset            # 0xA0 + zone
        # zone = startZone + idx//94 ; pos = 161 + idx%94
        labels = [f"{startZone + idx // 94}{160 + 1 + idx % 94}"
                  for idx in range(len(rawChars))]
        fullCharList.extend(rawChars)
        fullLabelList.extend(labels)

    # build look‑up dict once
    charToLabel = dict(zip(fullCharList, fullLabelList))

    # ── 2. map every char in *targetTxtPath* ──────────────────────────
    labelList, charList = [], []
    for char in (c for line in Path(targetTxtPath).read_text(encoding="utf‑8").splitlines()
                    for c in line.strip()):
        if char not in charToLabel:
            raise ValueError(f"Character '{char}' not found in GB2312 tables.")
        charList.append(char)
        labelList.append(charToLabel[char])


    charList = GetChars(targetTxtPath)
    return labelList,charList


from pathlib import Path
import numpy as np

# Assumes DrawSingleChar and GRAYSCALE_AVG are defined elsewhere
# from your_module import DrawSingleChar, GRAYSCALE_AVG

def GenerateFontsFromOtts(chars):
    """
    For each character in *chars*, render it using every .ttf/.otf font
    found recursively under *fontRoot*. Each glyph is rendered to a 
    64×64 grayscale image, normalized to the range [-1, 1].

    Parameters
    ----------
    chars : Sequence[str]
        List of characters to be rendered.
    fontRoot : str | Path
        Root directory that contains .ttf/.otf font files (searched recursively).
    grayscaleAvg : float
        The average grayscale value to normalize pixel intensities into [-1, 1].

    Returns
    -------
    np.ndarray
        A 4D tensor of shape (nChars, nFonts, 64, 64), dtype float32.
        Each entry is a normalized grayscale glyph image.
    """
    
    fontRoot: Union[str, Path, Sequence[str]] = \
        "/data0/haochuan/CASIA_Dataset/Sources/PrintedSources/64_FoundContentPrototypeTtfOtfs/Simplified"
    grayscaleAvg=128.0
    
    # Recursively collect all font file paths under the specified root
    fontPaths = sorted(
        str(p) for p in Path(fontRoot).rglob("*")
        if p.suffix.lower() in {".ttf", ".otf"}
    )
    if not fontPaths:
        raise FileNotFoundError(f"No font files found under {fontRoot!s}")

    glyphTensor = []
    for ch in chars:
        glyphs = [
            np.asarray(DrawSingleChar(char=ch, fontPath=fp))[:, :, 0]   # (64, 64)
            for fp in fontPaths
        ]
        glyphs = np.stack(glyphs, axis=0).astype(np.float32)      # (nFonts, 64, 64)
        glyphs = glyphs / grayscaleAvg - 1.0                      # normalise
        glyphTensor.append(glyphs)

    return torch.from_numpy(np.stack(glyphTensor, axis=0)) # (nChars, nFonts, 64, 64)
    


def DrawSingleChar(char: str,
                         fontPath: str,
                         canvasSize: int = 256,
                         glyphBox: int = 150,
                         outSize: Union[int, Tuple[int, int]] = 64) -> Image.Image:
    """
    Render *char* with *fontPath* and return a super‑sharp, antialiased
    glyph image (default 64×64) produced by an AREA➜LANCZOS‑4 pipeline.

    Parameters
    ----------
    char       : str
        The character to render.
    fontPath   : str
        Path to a .ttf/.otf font that contains the glyph.
    canvasSize : int
        Side length of the temporary drawing canvas.
    glyphBox   : int
        Side length of the intermediate square glyph before final down‑sampling.
    outSize    : int | (w, h)
        Final output size.  If int, the image is square.

    Returns
    -------
    PIL.Image.Image
        A high‑quality, antialiased glyph image.
    """
    # ------------------------------------------------------------------ #
    # 1. Draw glyph on a white canvas                                    #
    # ------------------------------------------------------------------ #
    canvas = Image.new("RGB", (canvasSize, canvasSize), (255, 255, 255))
    draw   = ImageDraw.Draw(canvas)
    font   = ImageFont.truetype(fontPath, size=150)
    draw.text((20, 20), char, fill=(0, 0, 0), font=font)

    imgArr = np.asarray(canvas)[:, :, 0]                   # channel 0 only
    if 0 not in imgArr:                                    # empty glyph
        blank = Image.new("L", (outSize, outSize), color=255)
        return blank.convert("RGB")

    # ------------------------------------------------------------------ #
    # 2. Tight bounding box & square expansion                           #
    # ------------------------------------------------------------------ #
    rows, cols = np.where(imgArr == 0)
    top, bottom = rows.min(), rows.max()
    left, right = cols.min(), cols.max()

    h, w = bottom - top, right - left
    side = max(h, w) + (1 - max(h, w) % 2)                # force even
    padH = (side - h) // 2
    padW = (side - w) // 2

    top    = max(0, top  - padH)
    bottom = min(canvasSize - 1, bottom + padH)
    left   = max(0, left - padW)
    right  = min(canvasSize - 1, right + padW)

    glyph = imgArr[top:bottom + 1, left:right + 1]

    # pad missing borders if clipping occurred
    padHmiss = max(0, side - glyph.shape[0])
    padWmiss = max(0, side - glyph.shape[1])
    if padHmiss or padWmiss:
        glyph = np.pad(
            glyph,
            ((padHmiss // 2, padHmiss - padHmiss // 2),
             (padWmiss // 2, padWmiss - padWmiss // 2)),
            constant_values=255
        )

    # ------------------------------------------------------------------ #
    # 3. Paste centered on a fresh canvas                                #
    # ------------------------------------------------------------------ #
    glyphRgb = np.repeat(glyph[..., None], 3, axis=2).astype(np.uint8)
    glyphImg = Image.fromarray(glyphRgb).resize((glyphBox, glyphBox),
                                                Image.Resampling.LANCZOS)
    bigCanvas = Image.new("RGB", (canvasSize, canvasSize), (255, 255, 255))
    offset = (canvasSize - glyphBox) // 2
    bigCanvas.paste(glyphImg, (offset, offset))

    # ------------------------------------------------------------------ #
    # 4. Two‑stage AREA → LANCZOS‑4 down‑sampling with OpenCV            #
    # ------------------------------------------------------------------ #
    # PIL → OpenCV (RGB → BGR)
    cvImg = cv2.cvtColor(np.asarray(bigCanvas), cv2.COLOR_RGB2BGR)

    # Normalise outSize to (w, h)
    if isinstance(outSize, int):
        tgtW = tgtH = outSize
    else:
        tgtW, tgtH = map(int, outSize)

    # Progressive AREA halving
    while min(cvImg.shape[:2]) // 2 >= min(tgtH, tgtW) * 2:
        cvImg = cv2.resize(cvImg, (0, 0), fx=0.5, fy=0.5,
                           interpolation=cv2.INTER_AREA)

    # Final Lanczos‑4 pass
    cvImg = cv2.resize(cvImg, (tgtW, tgtH), interpolation=cv2.INTER_LANCZOS4)

    # Back to PIL (BGR → RGB) and return
    cvImg = cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cvImg)


def TransformToDisplay(outTensor):
    # outTensor: shape (N, 64, 64), values in [-1, 1]
    outTensor=outTensor.to('cpu').squeeze()
    nChars, h, w = outTensor.shape
    targetAspectRatio = 5 / 7  # width / height

    # ── 1. 计算目标排布行列数（接近目标宽高比） ───────────────────────────────
    gridCols = math.ceil(math.sqrt(nChars * targetAspectRatio))
    gridRows = math.ceil(nChars / gridCols)

    # ── 2. padCount：只补尾部，不补前面 ───────────────────────────────────
    padCount = gridCols * gridRows - nChars
    if padCount > 0:
        padTail = torch.full((padCount, h, w), 1.0, dtype=outTensor.dtype)
        outTensor = torch.cat([outTensor, padTail], dim=0)

    # ── 3. 映射到 [0, 255] 区间，方便显示 ─────────────────────────────────
    imgArray = ((outTensor + 1) * 127.5).clamp(0, 255).byte().numpy()  # shape (N, 64, 64)

    # ── 4. reshape → grid 拼图 ─────────────────────────────────────────
    grid = imgArray.reshape(gridRows, gridCols, h, w)                     # (rows, cols, h, w)
    grid = np.transpose(grid, (0, 2, 1, 3)).reshape(gridRows * h, gridCols * w)

    # ── 5. 保存和显示 ─────────────────────────────────────────────────
    gridImage = Image.fromarray(grid, mode='L')
    
    
    return gridImage
    # gridImage.save(path)
    
    
def MakeGifFromPngs(handler,
    image_paths,
    output_path="animation.gif",
    duration=500,
    loop=0,
    resize_to=None
):
    """
    Create a GIF animation from a list of PNG images.

    Parameters:
    -----------
    image_paths : list[str]
        List of paths to PNG images (should be in desired order).
    output_path : str
        File path to save the output GIF.
    duration : int
        Duration of each frame in milliseconds.
    loop : int
        Number of loops (0 = infinite).
    resize_to : tuple[int, int] or None
        Resize all frames to this (width, height), if specified.
    """
    if not image_paths:
        raise ValueError("image_paths is empty!")

    frames = [Image.open(p).convert("RGB") for p in image_paths]

    if resize_to:
        frames = [img.resize(resize_to, Image.Resampling.LANCZOS) for img in frames]

    frames[0].save(
        output_path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop
    )
    
    return frames


def BaisedRandomK(K, N):
    """
    从 1 到 min(K, N) 中选择一个整数 n，
    小值更可能被选中。
    """
    max_choice = min(K, N)
    # 生成反比例权重，例如 [1/1, 1/2, ..., 1/max_choice]
    weights = [1 / (i + 1) for i in range(max_choice)]
    # 使用权重进行带偏采样
    n = random.choices(range(1, max_choice + 1), weights=weights, k=1)[0]
    return n