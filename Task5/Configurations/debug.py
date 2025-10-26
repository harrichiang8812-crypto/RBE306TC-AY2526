
import os
dataPathRoot = '/data0/haochuan/'


hyperParams = {
        'debugMode':1,
        'expID':'Debug-Task5',# experiment name prefix
        'expDir': '/data-shared/server01/data1/haochuan/CharacterRecordsDebug/',


        'YamlPackage': '../YamlLists/Debug/',
        
       
        
        'FullLabel0Vec': '/data-shared/server09/data0/haochuan/CASIA_Dataset/LabelVecs/Debug-Label0.txt',
        'FullLabel1Vec': '/data-shared/server09/data0/haochuan/CASIA_Dataset/LabelVecs/Debug-Label1.txt',

        
        # training configurations
        'augmentation':'HardAugmentationSchecule',
        
        
        'inputContentNum':5,


        # input params
        'imgWidth':64,
        'channels':1,

        # optimizer setting
        'optimizer':'adam',
        'gradNorm': 1,

        # feature extractor parametrers
        'TrueFakeExtractorPath': [],
        'ContentExtractorPath':['/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/Debug/Ckpts/Content/VGG11Net/BestExtractor.pth'],
        'StyleExtractorPath':  ['/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/Debug/Ckpts/Style/VGG11Net/BestExtractor.pth'],
        
        # learning hypers
        'initLrD': 0.00005,
        'initLrG': 0.00002
        

}


penalties = {
        'PenaltyGeneratorWeightRegularizer': 0.0001,
        'PenaltyDiscriminatorWeightRegularizer':0.0003,
        'PenaltyReconstructionL1':1,
        'PenaltyConstContent':0.2,
        'PenaltyConstStyle':0.2,
        'PenaltyDiscriminatorCategory': 0,
        'GeneratorCategoricalPenalty': 1,

        'PenaltyStyleFeatureExtractor':[1],
        'PenaltyContentFeatureExtractor':[1],
        'PenaltyVaeKl': 1,
        'PenaltyAdversarial':0.2,
        'PenaltyDiscriminatorPenalty':10

}

