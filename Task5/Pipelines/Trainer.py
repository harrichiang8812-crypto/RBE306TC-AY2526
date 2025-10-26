#Trainer.py
import logging
import os
import shutil
import sys
import numpy as np
import torch

# Disable TensorFloat-32 (TF32) globally for higher precision in GPU computations
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Disable warnings about future deprecation and other non-essential warnings
#torch.set_warn_always(False)
#import warnings
#warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
#import shutup  # External package to suppress warnings
#shutup.please()

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from time import time
from tqdm import tqdm
from typing import List

import multiprocessing
import glob
import random

import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime

# from torch.cuda.amp import autocast, GradScaler
# scaler = GradScaler()
from PIL import Image, ImageDraw, ImageFont


# Constants for model training
MIN_gradNorm = 0.1  # Minimum gradient norm threshold for clipping
MAX_gradNorm = 1.0  # Maximum gradient norm threshold for clipping


eps = 1e-9  # Small value to avoid division by zero in calculations

INITIAL_TRAIN_EPOCHS = 7  # Epoch threshold for transitioning augmentation methods
START_TRAIN_EPOCHS = 3  # Epoch after which simple augmentations start
from Pipelines.LoggerTB import NUM_SAMPLE_PER_EPOCH

sys.path.append('./')
from Pipelines.DatasetWnet import CharacterDataset
from Networks.PlainGenerators.PlainWNetBase import WNetGenerator as PlainWnet
from LossAccuracyEntropy.Loss import Loss
from Tools.Utilities import Logging, PrintInfoLog
from Tools.Utilities import SplitName

from Pipelines.DatasetWnet import transformTrainZero, transformTrainMinor, transformTrainHalf, transformTrainFull, transformSingleContentGT
from Tools.Utilities import BaisedRandomK
from Pipelines.LoggerTB import TBLogger as Logger


N_CRITIC = 5
WARMUP_EPOCS = 1
RAMP_EPOCHS = 5


# Define data augmentation modes for various stages of training
DataAugmentationMode = {
    'NoAugmentation': ['START', 'START', 'START'],
    'SimpleAugmentation': ['INITIAL', 'INITIAL', 'INITIAL'],
    'HardAumentation': ['FULL', 'FULL', 'FULL'],
    'SimpleAugmentationSchecule': ['START', 'INITIAL', 'INITIAL'],
    'HardAugmentationSchecule': ['START', 'INITIAL', 'FULL'],
}

class ThresholdScheduler:
    """
    Scheduler for dynamically managing threshold values such as gradient norms
    """
    def __init__(self, initial_threshold, decay_factor, min_threshold=0.0001):
        """
        Args:
            initial_threshold (float): Initial threshold value.
            decay_factor (float): Factor by which the threshold is decayed.
            min_threshold (float): Minimum value the threshold can reach.
        """
        self.threshold = initial_threshold
        self.decay_factor = decay_factor
        self.min_threshold = min_threshold

    def Step(self):
        """Decays the threshold value at each step, ensuring it doesn't fall below the minimum threshold."""
        self.threshold = max(self.min_threshold, self.threshold * self.decay_factor)

    def GetThreshold(self):
        """Returns the current threshold value."""
        return self.threshold


def SetWorker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



class Trainer(nn.Module):
    """
    Trainer class responsible for managing the entire training process, including data loading,
    model initialization, optimization, and logging.
    """
    def __init__(self, hyperParams=-1, penalties=-1):
        super().__init__()
        
        self.separator = "########################################################################################################"

        # Store hyperparameters and penalties
        self.config = hyperParams
        self.penalties = penalties

        # If not resuming training, clean up the existing log, image, and experiment directories
        if self.config.userInterface.resumeTrain == 0:
            if os.path.exists(self.config.userInterface.logDir):
                shutil.rmtree(self.config.userInterface.logDir)
            if os.path.exists(self.config.userInterface.trainImageDir):
                shutil.rmtree(self.config.userInterface.trainImageDir)
            if os.path.exists(self.config.userInterface.expDir):
                shutil.rmtree(self.config.userInterface.expDir)

        # Create directories for logs, training images, and experiment data if they don't exist
        os.makedirs(self.config.userInterface.logDir, exist_ok=True)
        os.makedirs(self.config.userInterface.trainImageDir, exist_ok=True)
        os.makedirs(self.config.userInterface.expDir, exist_ok=True)
        os.makedirs(self.config.userInterface.expDir+'/Generator', exist_ok=True)
        os.makedirs(self.config.userInterface.expDir+'/Discriminator', exist_ok=True)
        os.makedirs(self.config.userInterface.expDir+'/Frameworks', exist_ok=True)
        

        # Create console handler to display logs in the terminal
        self.sessionLog = Logging(sys.stdout)
        self.sessionLog.terminator = ''  # Prevent automatic newline in log output
        logging.getLogger().addHandler(self.sessionLog)
        
        # Initialize TensorBoard writer for tracking training progress
        writer = SummaryWriter(self.config.userInterface.logDir)
        self.dispLogTrain = Logger(writer=writer, cfg=self.config, sessionLog=self.sessionLog)
        self.dispLogVerifyTrain = Logger(writer=writer, cfg=self.config, sessionLog=self.sessionLog)
        self.dispLogVerifyTest = Logger(writer=writer, cfg=self.config, sessionLog=self.sessionLog)

        # Configure logging to include timestamps (year, month, day, hour, minute, second)
        current_time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        logging.basicConfig(filename=os.path.join(self.config.userInterface.logDir, current_time + "-Log.txt"),
                            level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')


        # Data augmentation strategy based on the configuration
        self.augmentationApproach = DataAugmentationMode[self.config.datasetConfig.augmentation]
        self.debug = self.config.debug  # Flag to enable debug mode

        self.iters = 0  # Initialize iteration counter
        self.startEpoch = 0  # Initialize starting epoch

        # Model initialization: select the appropriate WNetGenerator model based on configuration
        self.generator = PlainWnet(self.config, self.sessionLog)
        self.generator.train()  # Set the model to training mode
        self.generator.cuda()  # Move the model to GPU
        
        # Apply Xavier initialization to layers (Conv2D and Linear)
        for m in self.generator.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)  # Set biases to 0
            elif isinstance(m, (nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)  # Set biases to 0
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)  # Set weights to 1 for BatchNorm layers
                m.bias.data.zero_()  # Set biases to 0 for BatchNorm layers
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)  # Set weights to 1 for InstanceNorm layers
                m.bias.data.zero_()    # Set biases to 0 for InstanceNorm layers

        # Optimizer selection based on the config (Adam, SGD, or RMSprop)
        if self.config.trainParams.optimizer == 'adam':
            self.optimizerG = torch.optim.Adam(self.generator.parameters(), lr=self.config.trainParams.initLrG, betas=(0.0, 0.9),
                                              weight_decay=self.penalties.PenaltyGeneratorWeightRegularizer)
        elif self.config.trainParams.optimizer == 'sgd':
            self.optimizerG = torch.optim.SGD(self.generator.parameters(), lr=self.config.trainParams.initLrG, momentum=0.9)
        elif self.config.trainParams.optimizer == 'rms':
            self.optimizerG = torch.optim.RMSprop(self.generator.parameters(), lr=self.config.trainParams.initLrG,
                                                 alpha=0.99, eps=1e-08,
                                                 weight_decay=self.penalties.PenaltyGeneratorWeightRegularizer)
            

        # Loss function
        self.sumLoss = Loss(self.config, self.sessionLog, self.penalties, [WARMUP_EPOCS,RAMP_EPOCHS])

        # Set number of workers for loading data based on the debug mode
        self.workerGenerator = torch.Generator()
        self.workerGenerator.manual_seed(self.config.seed)
        if not self.debug:
            self.workersNum = 24
            if 'HW' in self.config.expID:
                gtFileTxtDir = "../GTList/HW/"
            elif 'PF' in self.config.expID:
                gtFileTxtDir = "../GTList/PF/"
        else:
            self.workersNum = 1  # No multiprocessing in debug mode
            gtFileTxtDir = "../GTList/Debug/"
        
        # Log the number of threads used for reading data
        PrintInfoLog(self.sessionLog, f"Reading Data: {self.workersNum}/{multiprocessing.cpu_count()} Threads")

        # Initialize training and testing datasets and data loaders
        self.trainset = CharacterDataset(self.config, sessionLog=self.sessionLog)
        self.testSet = CharacterDataset(self.config, sessionLog=self.sessionLog, is_train=False)
        
        
        self.trainset.RegisterEvaulationContentGts(self.config.userInterface.trainImageDir,'Train', debug=self.debug, gtListFile=gtFileTxtDir)
        self.testSet.RegisterEvaulationContentGts(self.config.userInterface.trainImageDir,'Test', debug=self.debug, gtListFile=gtFileTxtDir)
        

        # Learning rate scheduler with exponential decay
        lrGamma = np.power(0.01, 1.0 / (self.config.trainParams.epochs - 1))
        self.lrScheculerG = torch.optim.lr_scheduler.ExponentialLR(gamma=lrGamma, optimizer=self.optimizerG)

        # Gradient norm scheduler
        gradNormGamma = np.power(0.75, 1.0 / (self.config.trainParams.epochs - 1))
        self.gradGNormScheduler = ThresholdScheduler(MIN_gradNorm, gradNormGamma)

        # Resume training from the latest checkpoint if required
        if self.config.userInterface.resumeTrain == 1:
            PrintInfoLog(self.sessionLog, f'Load model from {self.config.userInterface.expDir}')
            
            # for generator 
            listFiles = glob.glob(self.config.userInterface.expDir+'/Generator' + '/*.pth')
            latestFile = max(listFiles, key=os.path.getctime)  # Get the latest checkpoint file
            ckptG = torch.load(latestFile)  # Load the checkpoint
            self.generator.load_state_dict(ckptG['stateDict'])  # Load the model weights
            self.optimizerG.load_state_dict(ckptG['optimizer'])  # Load the optimizer state
            
            
            # for framework
            listFiles = glob.glob(self.config.userInterface.expDir+'/Frameworks' + '/*.pth')
            latestFile = max(listFiles, key=os.path.getctime)  # Get the latest checkpoint file
            ckptFramework = torch.load(latestFile)  # Load the checkpoint
            self.startEpoch = ckptFramework['epoch']  # Get the starting epoch from the checkpoint
            
            

            # Reset learning rate after loading
            for param_group in self.optimizerG.param_groups:
                param_group['lr'] = self.config.trainParams.initLrG
                

            # Apply the learning rate and gradient schedulers up to the start epoch
            for _ in range(self.startEpoch):
                self.gradGNormScheduler.Step()  # Adjust gradient norm threshold
                self.lrScheculerG.step()  # Adjust learning rate
            self.startEpoch = self.startEpoch

        # Initialize a dictionary to record gradient values
        self.gradG = {}
        self.gradD = {}

        class _gradient():
            def __init__(self):
                self.value = 0.0
                self.count = 0

        # Populate the gradient dictionary with parameters from the model
        for idx, (name, param) in enumerate(self.generator.named_parameters()):
            subName, layerName = name.split('.')[0], name.split('.')[1]
            if subName not in self.gradG:
                self.gradG.update({subName: {}})
            if layerName not in self.gradG[subName]:
                self.gradG[subName].update({layerName: _gradient()})
            self.gradG[subName][layerName].count = self.gradG[subName][layerName].count + 1
            
        # Log that the trainer is ready
        PrintInfoLog(self.sessionLog, 'Trainer prepared.')

        
        
    def TrainOneEpoch(self, epoch):
        # Reset the data augmentation if resuming training from an early epoch
        if epoch < START_TRAIN_EPOCHS:
            self.trainset, self.trainLoader = self.ResetDataLoader(epoch = epoch+1, thisSet=self.trainset, 
                                                                   info=self.augmentationApproach[0], 
                                                                   mark="TrainingSet",  isTrain=True)
        elif epoch < INITIAL_TRAIN_EPOCHS:
            self.trainset, self.trainLoader = self.ResetDataLoader(epoch = epoch+1,  thisSet=self.trainset, 
                                                                   info=self.augmentationApproach[1], 
                                                                   mark="TrainingSet",  isTrain=True)
        else:
            self.trainset, self.trainLoader = self.ResetDataLoader(epoch = epoch+1, thisSet=self.trainset, 
                                                                   info=self.augmentationApproach[2], 
                                                                   mark="TrainingSet",  isTrain=True)

        
        # torch.autograd.set_detect_anomaly(True)
        """Train the model for a single epoch."""
        self.generator.train()  # Set the model to training mode
        time1 = time()  # Track start time for the epoch
        thisRoundStartItr1 = 0  # Used to track when to write summaries
        thisRoundStartItr2 = 0 
        
        
        trainProgress = tqdm(enumerate(self.trainLoader), total=len(self.trainLoader),
                             desc="Training @ Epoch %d" % (epoch+1))  # Progress bar for the training epoch
        
        for idx, (contents, styles, gt, onehotContent, onehotStyle) in trainProgress:
                
            self.generator.train()  # Set the model to training mode
            
            # Move data to the GPU and require gradients
            contents, styles, gt, onehotContent, onehotStyle = contents.cuda().requires_grad_(), \
                styles.cuda().requires_grad_(), gt.cuda().requires_grad_(), \
                onehotContent.cuda().requires_grad_(), onehotStyle.cuda().requires_grad_()
            reshapedStyle = styles.reshape(-1, 1, 64, 64)

            
            # Forward pass: generate content and style features, categories, and final output (fake images)
            encodedContentFeatures, encodedStyleFeatures, encodedContentCategory, encodedStyleCategory, generated, vae = \
                self.generator(contents, reshapedStyle, gt)

            # Prepare inputs for the loss function
            lossInputs = {'InputContents': contents, 
                          'InputStyles': reshapedStyle, 
                          'GT': gt, 'fake':generated,
                          'encodedContents':[encodedContentFeatures,encodedContentCategory],
                          'encodedStyles':[encodedStyleFeatures,encodedStyleCategory],
                          'oneHotLabels':[onehotContent, onehotStyle]}
            

            # with autocast():
            # Compute the total generator loss and detailed loss breakdown
            dictLosses = self.sumLoss(epoch, lossInputs, 
                                     SplitName(self.config.generator.mixer)[1:-1],
                                     mode='Train')
        

            # Update the progress bar with the current loss
            dispID = self.config.expID.replace('Exp','')
            dispID = dispID.replace('Encoder','E')
            dispID = dispID.replace('Mixer','M')
            dispID = dispID.replace('Decoder','D')
            trainProgress.set_description(dispID + " Training @ Epoch: %d/%d, LossL1: %.3f" %
                                          (epoch+1, self.config.trainParams.epochs, dictLosses['lossL1']), refresh=True)

            
            self.optimizerG.zero_grad()
            dictLosses['sumLossG'].backward()  # Compute gradients
        
            # Gradient norm clipping and adjustment (if enabled)
            if self.config.trainParams.gradientNorm:
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=MAX_gradNorm)  # Clip gradients
                for name, param in self.generator.named_parameters():
                    if param.grad is not None:
                        gradNorm = torch.norm(param.grad)
                        if gradNorm < self.gradGNormScheduler.GetThreshold():
                            param.grad = param.grad + eps  # Prevent zero gradients by adding epsilon
                            gradNorm = torch.norm(param.grad)  # Recompute gradient norm
                            scaleFactor = self.gradGNormScheduler.GetThreshold() / (gradNorm + eps)  # Adjust scale factor
                            param.grad = param.grad*scaleFactor  # Scale up the gradient
                
                self.optimizerG.step()

            
            # Log and write summaries at regular intervals
            need_sum, need_essay, progress = self.dispLogTrain.ShouldWriteSummaryTrain(
                    idx                 = idx,
                    epoch               = epoch,
                    dataset_len         = len(self.trainset),
                    batch_size          = self.config.trainParams.batchSize,
                    last_summary_prog   = thisRoundStartItr1,
                    last_anim_prog      = thisRoundStartItr2,
                )
            
            if need_sum:
                essay_imgs = []
                current_epoch_float = (progress + epoch * NUM_SAMPLE_PER_EPOCH) / NUM_SAMPLE_PER_EPOCH
                if need_essay:
                    trainEssay = self.dispLogTrain.WritingEssay(self.trainset, current_epoch_float, 'TrainingSet', self.generator)
                    testEssay  = self.dispLogTrain.WritingEssay(self.testSet,  current_epoch_float, 'TestingSet',  self.generator)
                    essay_imgs = [trainEssay, testEssay]
                    thisRoundStartItr2 = progress       # animation update
            
                self.dispLogTrain.Write2TB(
                    generator=self.generator,
                    eval_contents=contents,
                    eval_styles=styles,
                    eval_gts=gt,
                    eval_fakes=generated,
                    step=epoch * NUM_SAMPLE_PER_EPOCH + int(idx / (len(self.trainLoader)) * NUM_SAMPLE_PER_EPOCH)+1,
                    mark='Train',
                    loss_dict=dictLosses,
                    grad_g=self.gradG,
                    lr_g=self.lrScheculerG.get_lr()[0], 
                    grad_thresh=self.gradGNormScheduler.GetThreshold(),
                    essay_img = essay_imgs
                )
                
                # thisRoundStartItr1 = idx * float(NUM_SAMPLE_PER_EPOCH) / len(self.trainLoader)
                thisRoundStartItr1 = progress
                
            self.iters = self.iters+ 1  # Update iteration count
        time2 = time()  # Track end time for the epoch
        PrintInfoLog(self.sessionLog, 'Training completed @ Epoch: %d/%d training time: %f mins, L1Loss: %.3f' % 
                     (epoch+1, self.config.trainParams.epochs, (time2-time1)/60, dictLosses['lossL1']))  # Log epoch time and loss

    
    def TestOneEpoch(self, epoch, thisSet, mark):
        thisSet, thisLoader = self.ResetDataLoader(epoch = epoch+1, 
                                                   thisSet=thisSet, info='ZERO', 
                                                   mark=mark, isTrain=False)
        
        # # Disable training mode for evaluation
        is_train = False
        self.generator.eval()  # Set the model to evaluation mode
        
        with torch.no_grad():  # Disable gradient calculations for efficiency

            
            # """Evaluate the model on the test dataset for a single epoch."""
            
            time1 = time()  # Track start time
            thisRoundStartItr1 = 0
            testProgress = tqdm(enumerate(thisLoader), total=len(thisLoader), desc="Verifying Statistics @ %s Epoch %d" % (mark, epoch+1))
            for idx, (contents, styles, gt, onehotContent, onehotStyle) in testProgress:
                # Move data to the GPU
                contents, styles, gt, onehotContent, onehotStyle = contents.cuda(), \
                    styles.cuda(), gt.cuda(), onehotContent.cuda(), onehotStyle.cuda()

                # Reshape style input
                reshaped_styles = styles.reshape(-1, 1, 64, 64)

                # Forward pass
                encodedContentFeatures, encodedStyleFeatures, encodedContentCategory, encodedStyleCategory, generated, vae = \
                    self.generator(contents, reshaped_styles, gt, is_train=is_train)

                # Prepare inputs for the loss function
                lossInputs = {'InputContents': contents, 
                                'InputStyles': reshaped_styles, 
                                'GT': gt, 'fake':generated,
                                'encodedContents':[encodedContentFeatures,encodedContentCategory],
                                'encodedStyles':[encodedStyleFeatures,encodedStyleCategory],
                                'oneHotLabels':[onehotContent, onehotStyle]}
                if vae != -1:
                    lossInputs.update({'vae': vae})
                
                
                # with autocast():
                dictLosses = self.sumLoss(epoch, lossInputs,
                                          SplitName(self.config.generator.mixer)[1:-1], mode=mark)

                
                # Update progress bar with current loss
                dispID = self.config.expID.replace('Exp','')
                dispID = dispID.replace('Encoder','E')
                dispID = dispID.replace('Mixer','M')
                dispID = dispID.replace('Decoder','D')
                testProgress.set_description(dispID + " Verifying at %s @ Epoch: %d/%d, LossL1: %.3f" %
                                                (mark, epoch+1, self.config.trainParams.epochs, dictLosses['lossL1']), refresh=True)

                # # Write summaries at regular intervals
                if 'Test' in mark:
                    thisLogger = self.dispLogVerifyTest
                elif 'Train' in mark:
                    thisLogger = self.dispLogVerifyTrain
                need_sum, progress = thisLogger.ShouldWriteSummaryTest(
                            idx                 = idx,
                            epoch               = epoch,
                            dataset_len         = len(thisSet),     # 或者你已有的变量
                            batch_size          = self.config.trainParams.batchSize,
                            last_summary_prog   = thisRoundStartItr1,
                        )
                
                if need_sum:
                    thisLogger.Write2TB(generator=self.generator, 
                        eval_contents=contents,eval_styles=styles,eval_gts=gt,eval_fakes=generated,
                                    step=epoch * NUM_SAMPLE_PER_EPOCH + int(idx / len(thisLoader) * NUM_SAMPLE_PER_EPOCH)+1,
                                    mark="Verifying@"+mark,
                                    loss_dict=dictLosses)
                    # thisRoundStartItr1 = idx * float(NUM_SAMPLE_PER_EPOCH) / len(thisLoader)
                    thisRoundStartItr1 = progress

        time2 = time()  # Track end time for the epoch
        PrintInfoLog(self.sessionLog, 'Verifying completed @ %s @ Epoch: %d/%d verifying time: %f mins, L1Loss: %.3f' % 
                     (mark, epoch+1, self.config.trainParams.epochs, (time2-time1)/60, dictLosses['lossL1']))  # Log epoch time and loss

    def Pipelines(self):
        """Main training and evaluation loop."""
        train_start = time()  # Start time of the entire training process
        training_epoch_list = range(self.startEpoch, self.config.trainParams.epochs, 1)  # List of epochs to train
        PrintInfoLog(self.sessionLog, self.separator)
        
        
        if (not self.config.userInterface.skipTest) and self.startEpoch != 0:
            self.TestOneEpoch(self.startEpoch-1, thisSet=self.trainset, mark='TrainingSet')  # Test at the start if not skipping   
            self.TestOneEpoch(self.startEpoch-1, thisSet=self.testSet,  mark='TestingSet')  # Test at the start if not skipping     

        for epoch in training_epoch_list:
            # Reset data augmentation strategies based on the epoch
            PrintInfoLog(self.sessionLog, self.separator)
            
            # Train and test model at each epoch
            self.TrainOneEpoch(epoch)
            if not self.config.userInterface.skipTest and\
                (epoch < START_TRAIN_EPOCHS \
                    or (epoch < INITIAL_TRAIN_EPOCHS and epoch % 3 == 0) \
                    or (epoch % 5 == 0) \
                    or epoch == self.config.trainParams.epochs-1):
                
                self.TestOneEpoch(epoch, thisSet=self.testSet,mark='TestingSet')  
            
            if not self.config.userInterface.skipTest and (epoch==0 or epoch==self.config.trainParams.epochs-1 or (epoch+1)%5==0):
                self.TestOneEpoch(epoch, thisSet=self.trainset, mark='TrainingSet')  
                  

            # Save model checkpoint at the end of the epoch
            stateG = {'stateDict': self.generator.state_dict(),'optimizer': self.optimizerG.state_dict()}
            torch.save(stateG, self.config.userInterface.expDir+'/Generator' + '/CkptEpoch%d.pth' % (epoch+1))
            stateFramework = {'epoch': epoch+1}
            torch.save(stateFramework, self.config.userInterface.expDir+'/Frameworks' + '/CkptEpoch%d.pth' % (epoch+1))
            
            logging.info(f'Trained model has been saved at Epoch {epoch+1}.')

            # Step the learning rate scheduler and gradient norm scheduler
            self.lrScheculerG.step()
            self.gradGNormScheduler.Step()
        # After training, log the total time taken and close the TensorBoard writer
        train_end = time()  # Record the end time of training
        training_time = (train_end - train_start) / 3600  # Convert total training time to hours
        self.dispLogTrain.writer.close()  # Close the TensorBoard writer
        logging.info('Training finished, tensorboardX writer closed')
        logging.info('Training total time: %f hours.' % training_time)    
        
    def ResetDataLoader(self, epoch, thisSet, info, mark, isTrain):
        """
        Set the data augmentation mode for training.
        
        Args:
            info (str): Augmentation mode ('START', 'INITIAL', 'FULL', 'NONE').
        """
        
        PrintInfoLog(self.sessionLog, "Reset " + mark + " StyleList and Switch Data Augmentation %s for %s at Epoch: %d ..." % (info, mark, epoch), end='\r')  
        if info == 'START':
            thisSet.augment = transformTrainMinor
        elif info == 'INITIAL':
            thisSet.augment = transformTrainHalf
        elif info == 'FULL':
            thisSet.augment = transformTrainFull
        elif info =='ZERO':
            thisSet.augment = transformTrainZero
        elif info == 'NONE':
            thisSet.augment = None
            
        full = thisSet.styleListFull  # 原始样本
        K = self.config.datasetConfig.availableStyleNum  # 需要的采样数量

        thisSet.styleList = []
        for styles in full:
            if len(styles) == 0:
                thisSet.styleList.append([])  # 无可选项则为空
                continue
            
            n = BaisedRandomK(K, len(styles))
            sampled = random.sample(styles, n)
            
            # 若不够 K，则从 sampled 中有放回采样补齐
            if n < K:
                extra = random.choices(sampled, k=K - n)
                final = sampled + extra
            else:
                final = sampled  # 恰好 K 个或更多时直接返回

            thisSet.styleList.append(final)
        
        thisLoader = DataLoader(thisSet, batch_size=self.config.trainParams.batchSize,
                                num_workers=self.workersNum, pin_memory=True, 
                                shuffle=(isTrain==True), drop_last=False, 
                                persistent_workers=False, 
                                worker_init_fn=SetWorker, generator=self.workerGenerator)
        PrintInfoLog(self.sessionLog, "Reset StyleList and Switch Data Augmentation %s for %s at Epoch: %d completed." % (info, mark, epoch))  
        return thisSet, thisLoader