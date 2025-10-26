import os
import random
import sys
import torch
# import torchvision.transforms.functional as F

# Suppress warnings and TensorFlow32
#torch.set_warn_always(False)
#import warnings
#warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
#import shutup
#shutup.please()

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import yaml
import cv2
from time import time
sys.path.append('./')
from Tools.Utilities import cv2torch, read_file_to_dict
import time
from tqdm import tqdm
displayInterval = 25
# from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np
from Tools.Utilities import PrintInfoLog
from Tools.Utilities import MergeAllDictKeys
import glob
import torchvision.transforms.functional as F
from Tools.Utilities import GB2312CharMapper,GenerateFontsFromOtts
from pathlib import Path

# from Tools.TestTools import get_chars_set_from_level1_2, get_revelant_data


def RotationAugmentationToChannels(img, config, fill=1, style=False):
    """
    Applies random rotation and shear transformations to each channel of an image.
    
    Args:
        img (torch.Tensor): Input image tensor with shape (C, H, W).
        config (list): List containing rotation, shear, and translation parameters.
        fill (int): Value to use for padding.
        style (bool): If True, applies different rotation to each channel.
        
    Returns:
        torch.Tensor: Transformed image with shape (C, H, W).
    """
    rotation = config[0]
    shear = config[1]
    degrees = (-rotation, rotation)
    shear = (-shear, shear)
    angle = random.uniform(*degrees)
    shear = (random.uniform(-shear[0], shear[0]), random.uniform(-shear[1], shear[1]))

    transformed_channels = []  # List to store transformed channels
    
    for c in range(img.shape[0]):  # Apply transformation to each channel
        if style:  # Generate new random rotation for each style channel
            angle = random.uniform(*degrees)
            shear = (random.uniform(-shear[0], shear[0]), random.uniform(-shear[1], shear[1]))
        
        single_channel_img = img[c:c+1, :, :]  # Extract single channel (shape 1xHxW)

        # Apply affine transformation
        transformed_channel = F.affine(single_channel_img, angle=angle, translate=(0, 0),
                                       scale=1.0, shear=shear, fill=fill)

        transformed_channels.append(transformed_channel)

    # Stack all channels back together
    transformed_image = torch.cat(transformed_channels, dim=0)
    return transformed_image


def TranslationAugmentationToChannels(img, config, fill=1, style=False):
    """
    Applies random translation (shifting) transformations to each channel of an image.
    
    Args:
        img (torch.Tensor): Input image tensor with shape (C, H, W).
        config (list): List containing translation parameters.
        fill (int): Value to use for padding.
        style (bool): If True, applies different translation to each channel.
        
    Returns:
        torch.Tensor: Transformed image with shape (C, H, W).
    """
    max_translate = config[2]
    tx = random.randint(-max_translate, max_translate)
    ty = random.randint(-max_translate, max_translate)

    transformed_channels = []  # List to store transformed channels
    
    for c in range(img.shape[0]):  # Apply translation to each channel
        if style:  # Generate new random translation for each style channel
            tx = random.randint(-max_translate, max_translate)
            ty = random.randint(-max_translate, max_translate)

        single_channel_img = img[c:c+1, :, :]  # Extract single channel (shape 1xHxW)

        # Pad the image to make room for the shift
        padded_img = F.pad(single_channel_img, (abs(tx), abs(ty), abs(tx), abs(ty)), fill=fill)

        # Get the original image size
        _, h, w = single_channel_img.shape

        # Crop the image back to original size
        cropped_img = F.crop(padded_img, abs(ty) if ty < 0 else 0, abs(tx) if tx < 0 else 0, h, w)
        transformed_channels.append(cropped_img)

    # Stack all channels back together
    transformed_image = torch.cat(transformed_channels, dim=0)
    return transformed_image



# Basic transformation for converting numpy arrays to PyTorch tensors
transformSingleContentGT = transforms.Compose([
    transforms.ToTensor(),  # Converts numpy.ndarray to torch.Tensor
    # Optionally normalize to [-1, 1] range using Normalize
    # transforms.Normalize((0.5,), (0.5,))
])

# Rotation, shear, and translation configurations for different augmentation modes
rotationTranslationFull = [10, 10, 15]  # Full rotation, shear, and translation values
rotationTranslationHalf = [5, 5, 10]    # Half rotation, shear, and translation
rotationTranslationMinor = [3, 3, 5]    # Minor rotation, shear, and translation
rotationTranslationZero = [0, 0, 0]     # No rotation, shear, or translation

# Data augmentation transformations for combined content and style (Full augmentations)
transformFullCombinedContentGT = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # 50% chance of horizontal flip
    transforms.RandomVerticalFlip(p=0.5),    # 50% chance of vertical flip
    # Random resizing and cropping to (64, 64) with 75-100% scaling
    transforms.RandomResizedCrop(size=(64, 64), scale=(0.75, 1.0), antialias=True),
])

transformFullStyle = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # 50% chance of horizontal flip
    transforms.RandomVerticalFlip(p=0.5),    # 50% chance of vertical flip
    # Random resizing and cropping for style images (no interpolation, just resize)
    transforms.RandomResizedCrop(size=(64, 64), scale=(0.75, 1.0), antialias=True),
    transforms.ToTensor(),  # Converts to PyTorch tensor
    # Optionally normalize to [-1, 1] range using Normalize
    # transforms.Normalize((0.5,), (0.5,))
])

# Full transformations include both content + GT transformations and style transformations
fullTransformation = [transformFullCombinedContentGT, transformFullStyle, rotationTranslationFull]

# Data augmentation transformations for half-level augmentation
transformHalfCombinedContentGT = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.25),  # 25% chance of horizontal flip
    transforms.RandomVerticalFlip(p=0.25),    # 25% chance of vertical flip
    # Random resizing and cropping with smaller scale range for less augmentation
    transforms.RandomResizedCrop(size=(64, 64), scale=(0.85, 1.0), antialias=True),
])

transformHalfStyle = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.25),  # 25% chance of horizontal flip
    transforms.RandomVerticalFlip(p=0.25),    # 25% chance of vertical flip
    transforms.RandomResizedCrop(size=(64, 64), scale=(0.85, 1.0), antialias=True),
    transforms.ToTensor(),  # Converts to PyTorch tensor
    # Optionally normalize to [-1, 1] range using Normalize
    # transforms.Normalize((0.5,), (0.5,))
])

# Half transformations set with less aggressive augmentations
halfTransformation = [transformHalfCombinedContentGT, transformHalfStyle, rotationTranslationHalf]

# Minor data augmentation for minimal changes in images
transformMinorCombinedContentGT = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.1),  # 10% chance of horizontal flip
    transforms.RandomVerticalFlip(p=0.1),    # 10% chance of vertical flip
    # Smaller resizing and cropping for minor augmentation
    transforms.RandomResizedCrop(size=(64, 64), scale=(0.95, 1.0), antialias=True),
])

transformMinorStyle = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.1),  # 10% chance of horizontal flip
    transforms.RandomVerticalFlip(p=0.1),    # 10% chance of vertical flip
    transforms.RandomResizedCrop(size=(64, 64), scale=(0.95, 1.0), antialias=True),
    transforms.ToTensor(),  # Converts to PyTorch tensor
    # Optionally normalize to [-1, 1] range using Normalize
    # transforms.Normalize((0.5,), (0.5,))
])

# Minor transformations for content + style images
minorTransformation = [transformMinorCombinedContentGT, transformMinorStyle, rotationTranslationMinor]

# No augmentations applied, just basic conversions
transformZeroCombinedContentGT = transforms.Compose([])
transformZeroStyle = transforms.Compose([
    transforms.ToTensor(),  # Converts numpy.ndarray to torch.Tensor
    # Optionally normalize to [-1, 1] range using Normalize
    # transforms.Normalize((0.5,), (0.5,))
])

# Zero transformations: No data augmentation
zeroTransformation = [transformZeroCombinedContentGT, transformZeroStyle, rotationTranslationZero]

# Defining augmentation dictionaries for different modes of training/testing
# Zero augmentation mode
transformTrainZero = {
    'singleContentGT': transformSingleContentGT,
    'combinedContentGT': zeroTransformation[0],
    'style': zeroTransformation[1],
    'rotationTranslation': zeroTransformation[2],
}

# Minor augmentation mode for training
transformTrainMinor = {
    'singleContentGT': transformSingleContentGT,
    'combinedContentGT': minorTransformation[0],
    'style': minorTransformation[1],
    'rotationTranslation': minorTransformation[2],
}

# Half augmentation mode for training
transformTrainHalf = {
    'singleContentGT': transformSingleContentGT,
    'combinedContentGT': halfTransformation[0],
    'style': halfTransformation[1],
    'rotationTranslation': halfTransformation[2],
}

# Full augmentation mode for training
transformTrainFull = {
    'singleContentGT': transformSingleContentGT,
    'combinedContentGT': fullTransformation[0],
    'style': fullTransformation[1],
    'rotationTranslation': fullTransformation[2],
}

# Test-time transformation (no augmentation, just tensor conversion)
transformTest = {
    'singleContentGT': transformSingleContentGT,
    'combinedContentGT': zeroTransformation[0],
    'style': zeroTransformation[1],
    'rotationTranslation': zeroTransformation[2],
}


class CharacterDataset(Dataset):
    def __init__(self, config, sessionLog, is_train=True):
        """
        Initialize the dataset by loading the content, ground truth, and style data from YAML files.
        
        Args:
            config (object): Configuration object containing dataset paths and parameters.
            sessionLog (object): Logging session to print logs.
            is_train (bool): Flag to indicate if the dataset is for training or testing.
        """
        self.is_train = is_train
        self.sessionLog = sessionLog
        self.config = config

        # Load paths to YAML files for train/test
        filesIdentified = glob.glob(config.datasetConfig.yamls+"*.yaml")
        if is_train:
            self.gtYaml = [ii for ii in filesIdentified if 'TrainGroundTruth' in ii]
            self.styleYaml = [ii for ii in filesIdentified if 'TrainStyleReference' in ii]
        else:
            self.gtYaml = [ii for ii in filesIdentified if 'TestGroundTruth' in ii]
            self.styleYaml = [ii for ii in filesIdentified if 'TestStyleReference' in ii]

        self.contentYaml = os.path.join(config.datasetConfig.yamls, 'Content.yaml')

        # Load the number of content and style inputs
        self.input_content_num = config.datasetConfig.inputContentNum
        self.input_style_num = config.datasetConfig.availableStyleNum

        strat_time = time.time()  # Start timer to measure data loading time
        self.gtDataList = self.CreateDataList(self.gtYaml)  # Load the ground truth data
        listLabel0 = [ii[1] for ii in self.gtDataList]
        listLabel1 = [ii[2] for ii in self.gtDataList]
        sortedListLabel1 = sorted(range(len(listLabel1)), key=lambda i: listLabel1[i])

        
        
        # Load content and style YAML files
        with open(self.contentYaml, 'r', encoding='utf-8') as f:
            PrintInfoLog(self.sessionLog, "Loading " + self.contentYaml + '...', end='\r')
            contentFiles = yaml.load(f.read(), Loader=yaml.FullLoader)
            PrintInfoLog(self.sessionLog, "Loading " + self.contentYaml + ' completed.')

        styleFiles = list()
        for _path in self.styleYaml:
            with open(_path, 'r', encoding='utf-8') as f:
                PrintInfoLog(self.sessionLog, "Loading " + _path + '...', end='\r')
                styleFiles.append(yaml.load(f.read(), Loader=yaml.FullLoader)) 
                PrintInfoLog(self.sessionLog, "Loading " + _path + ' completed.')
        self.styleFileDict = MergeAllDictKeys(styleFiles)
        
        
        self.contentList, self.styleListFull = [], []
        # Prepare content and style lists for each data point
        styleFiles = self.styleFileDict  # assign for compatibility with the original variable name
        for idx, (_, label0, label1) in tqdm(enumerate(self.gtDataList), total=len(self.gtDataList), desc="Loading: "):
            
            # contentFiles[label0] should already be a flat list — no need to reassign
            self.contentList.append(contentFiles[label0])
            
            # flatten list-of-lists from styleFiles[label1] on the fly
            flat_style_list = []
            for group in styleFiles[label1]:
                if isinstance(group, list):
                    flat_style_list.extend(group)
                else:
                    flat_style_list.append(group)
            self.styleListFull.append(flat_style_list)
            # self.styleList.append(random.sample(flat_style_list, self.input_style_num))
    
        # if not is_train:
        PrintInfoLog(self.sessionLog, "Reordering by label0 (content labels) ...", end='\r')
        sortedListLabel0 = sorted(range(len(listLabel0)), key=lambda i: listLabel0[i])
        self.gtDataList = [self.gtDataList[i] for i in sortedListLabel0]
        self.contentList = [self.contentList[i] for i in sortedListLabel0]
        self.styleListFull = [self.styleListFull[i] for i in sortedListLabel0]
        PrintInfoLog(self.sessionLog, "Reordering by label0 (content labels) completed.")

        # Initialize labels and one-hot encoding vectors
        self.label0order = config.datasetConfig.loadedLabel0Vec
        
        # One-hot encoding for content and style labels
        self.label1order = config.datasetConfig.loadedLabel1Vec
        self.onehotContent, self.onehotStyle = [0 for _ in range(len(self.label0order))], [0 for _ in range(len(self.label1order))]

        
        end_time = time.time()  # Measure the end time for data loading
        
        
        
        PrintInfoLog(self.sessionLog, f'Dataset cost: {(end_time - strat_time):.2f}s')
        
    
    
    def RegisterEvaulationContentGts(self, path, mark, debug, gtListFile):
        PrintInfoLog(self.sessionLog, f'Find GTs for %s ...' % mark, end='\r')
        start_time = time.time()

        # Load only the label pairs that are needed
        if debug:
            self.evalListLabel0, charList = GB2312CharMapper(targetTxtPath='../Scripts/EvalTxts/debug.txt')
        else:
            self.evalListLabel0, charList = GB2312CharMapper(targetTxtPath='../Scripts/EvalTxts/过秦论.txt')

        self.evalContents = GenerateFontsFromOtts(chars=charList)
        if self.config.debug:
            self.evalContents = self.evalContents[:, 0:5, :, :]

        # Construct the set of (label1, label0) pairs needed
        self.evalStyleLabels = [label for label, sublists in self.styleFileDict.items() if all(sublists)]
        required_pairs = set((int(l1), int(l0)) for l1 in self.evalStyleLabels for l0 in self.evalListLabel0)

        # Step 1: Stream GT files and build filtered lookup table
        lookup = dict()
        for txt_path in Path(gtListFile).glob('*.txt'):
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('@')
                    if len(parts) < 4:
                        continue
                    l0, l1, p = int(parts[1]), int(parts[2]), parts[3]
                    if (l1, l0) in required_pairs:
                        lookup[(l1, l0)] = (p, l0, l1)

        PrintInfoLog(self.sessionLog, f'Find GTs for %s completed.' % mark)

        # Visual display grid
        cols, rows, h, w = self.evalContents.shape
        grid = self.evalContents.permute(1, 2, 0, 3)
        bigImg = grid.reshape(rows * h, cols * w)
        bigImg = 255 - bigImg
        bigImg = bigImg.T
        img = Image.fromarray(bigImg.cpu().numpy().astype('uint8'))
        if mark == 'Train':
            img.save(os.path.join(path, 'ExampleContents.png'))

        # Step 2: Generate targetGts from lookup
        targetGts = [
            [lookup.get((int(label1), int(label0)), ('', int(label0), int(label1)))
            for label0 in self.evalListLabel0]
            for label1 in self.evalStyleLabels
        ]

        # Step 3: Convert targetGts into normalized image tensors
        self.evalGts = []
        for gt in targetGts:
            outTensorList = []
            for item in gt:
                if item[0] != '':
                    img_tensor = cv2torch(item[0], transform=transformSingleContentGT)
                    img_tensor = (img_tensor - 0.5) * 2
                else:
                    img_tensor = torch.full((1, 64, 64), 1)
                outTensorList.append(img_tensor)

            outTensor = torch.cat(outTensorList, dim=0)
            self.evalGts.append((outTensor, gt[0][2]))

        end_time = time.time()
        PrintInfoLog(self.sessionLog, f'EvalExample Resigtration completes for %s:{(end_time - start_time): .2f}s' % mark)
        return img

        
    def ResetStyleList(self, epoch, mark):
        """
        Efficiently resample style images for each data point before a new epoch.
        Stores result in self.styleList.
        """
        PrintInfoLog(self.sessionLog, "Reset " + mark + " StyleList at Epoch: %d " % epoch, end='\r')    
        full = self.styleListFull  # local reference for speed
        K = self.input_style_num
        self.styleList = [
            (random.sample(styles, K) if len(styles) >= K else random.choices(styles, k=K))
            for styles in full
        ]
        PrintInfoLog(self.sessionLog, "Reset " + mark + " StyleList at Epoch: %d " % epoch + 'completed.')

    def __getitem__(self, index):
        """
        Retrieve a single item from the dataset by index.
        
        Args:
            index (int): Index of the item to retrieve.
        
        Returns:
            tuple: Containing content tensor, style tensor, ground truth tensor, one-hot encoded content label, and one-hot encoded style label.
        """
        # Load and process content tensors
        tensorContent = (torch.cat([cv2torch(content, self.augment['singleContentGT']) for content in self.contentList[index]], dim=0) - 0.5) * 2
        content = self.contentList[index][0][:-4].split('_')[-2]  # Extract content label from file path
        content = self.label0order[content]
        onehotContent = torch.tensor(self.onehotContent)
        onehotContent[content] = 1

        # Load and process ground truth tensor
        tensorGT = (cv2torch(self.gtDataList[index][0], self.augment['singleContentGT']) - 0.5) * 2
        gtAndContent = torch.cat((tensorContent, tensorGT), 0)

        # Apply rotation and translation augmentations
        gtAndContent = RotationAugmentationToChannels(gtAndContent, self.augment['rotationTranslation'])
        gtAndContent = TranslationAugmentationToChannels(gtAndContent, self.augment['rotationTranslation'])
        gtAndContent = self.augment['combinedContentGT'](gtAndContent)

        # Separate content and ground truth tensors
        tensorContent = gtAndContent[:-1, :, :]
        tensorGT = torch.unsqueeze(gtAndContent[-1, :, :], 0)

        # Load and process style tensors
        currentStyleNum=len(self.styleList[index])
        tensorStyle = (torch.cat([cv2torch(reference_style, self.augment['style']) for reference_style in self.styleList[index]], dim=0) - 0.5) * 2
        perm = torch.randperm(currentStyleNum, device=tensorStyle.device)[:self.input_style_num]
        tensorStyle = torch.index_select(tensorStyle, 0, perm)
        tensorStyle = RotationAugmentationToChannels(tensorStyle, self.augment['rotationTranslation'], style=True)
        tensorStyle = TranslationAugmentationToChannels(tensorStyle, self.augment['rotationTranslation'], style=True)

        # Extract style label
        style = self.styleList[index][0][:-4].split('_')[-1]
        while style[0] == '0' and len(style) > 1:
            style = style[1:]
        style = self.label1order[style]
        onehotStyle = torch.tensor(self.onehotStyle)
        onehotStyle[style] = 1

        # permContentIdx = torch.randperm(tensorContent.size(0))     
        # tensorContent = tensorContent[permContentIdx]
        
        return tensorContent.float(), tensorStyle.float(), tensorGT.float(), onehotContent.float(), onehotStyle.float()

    def __len__(self):
        """
        Get the total number of items in the dataset.
        
        Returns:
            int: Length of the dataset.
        """
        return len(self.gtDataList)

    def CreateDataList(self, yamlName):
        """
        Load ground truth data from a YAML file and create a list of data points.
        
        Args:
            yamlName (str): Path to the YAML file containing the data.
        
        Returns:
            list: A list of tuples containing paths, content labels, and style labels.
        """
        data_list = []
        iteration_files=list()
        for _path in yamlName:
            with open(_path, 'r', encoding='utf-8') as f:
                PrintInfoLog(self.sessionLog, "Loading " + _path + '...', end='\r')
                iteration_files.append(yaml.load(f.read(), Loader=yaml.FullLoader))
                PrintInfoLog(self.sessionLog, "Loading " + _path + ' completed.')
        iterationFileDict = MergeAllDictKeys(iteration_files)
        
        counter = 0
        for _, (_, values) in tqdm(enumerate(iterationFileDict.items()), total=len(iterationFileDict.items()), desc="Test"):
            for _i_, value in enumerate(values):
                counter = counter+ 1
                path, label0, label1 = value
                data_list.append((path, label0, label1))
        return data_list

    