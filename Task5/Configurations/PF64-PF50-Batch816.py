import os
dataPathRoot = '/data0/haochuan/'


hyperParams = {
        'debugMode':1,
        'expID':'20250520PF-Batch816-Task5',# experiment name prefix
        'expDir': '/data-shared/server01/data1/haochuan/CharacterRecords2025May08/',
        
        'YamlPackage': '../YamlLists/PF64-PF80/',
        
        'FullLabel0Vec': '/data-shared/server09/data0/haochuan/CASIA_Dataset/LabelVecs/PF64-Label0.txt',
        'FullLabel1Vec': '/data-shared/server09/data0/haochuan/CASIA_Dataset/LabelVecs/PF80-Label1.txt',

        
        # training configurations
        'augmentation':'HardAugmentationSchecule', 
        # Options: 'NoAugmentation', 'SimpleAugmentation', 'HardAumentation', 'SimpleAugmentationSchecule', 'HardAugmentationSchecule'
        
        'inputContentNum':64,

        # input params
        'imgWidth':64,
        'channels':1,

        # optimizer setting
        'optimizer':'adam',
        'gradNorm': 1,

        # feature extractor parametrers
        'TrueFakeExtractorPath': [],
        'ContentExtractorPath':[
                '/data-shared/server09/data1/haochuan/Character/PtTrainedFeatureExtractors/PF64-Contents/VGG11Net/BestExtractor.pth',
                '/data-shared/server09/data1/haochuan/Character/PtTrainedFeatureExtractors/PF64-Contents/VGG13Net/BestExtractor.pth',
                '/data-shared/server09/data1/haochuan/Character/PtTrainedFeatureExtractors/PF64-Contents/ResNet18/BestExtractor.pth',
                '/data-shared/server09/data1/haochuan/Character/PtTrainedFeatureExtractors/PF64-Contents/ResNet34/BestExtractor.pth',
                '/data-shared/server09/data1/haochuan/Character/PtTrainedFeatureExtractors/PF64-Contents/ResNet50/BestExtractor.pth'
                ],
        
        'StyleExtractorPath':  [
                '/data-shared/server09/data1/haochuan/Character/PtTrainedFeatureExtractors/PF80-Style/VGG11Net/BestExtractor.pth',
                '/data-shared/server09/data1/haochuan/Character/PtTrainedFeatureExtractors/PF80-Style/VGG13Net/BestExtractor.pth',
                '/data-shared/server09/data1/haochuan/Character/PtTrainedFeatureExtractors/PF80-Style/ResNet18/BestExtractor.pth',
                '/data-shared/server09/data1/haochuan/Character/PtTrainedFeatureExtractors/PF80-Style/ResNet34/BestExtractor.pth',
                '/data-shared/server09/data1/haochuan/Character/PtTrainedFeatureExtractors/PF80-Style/ResNet50/BestExtractor.pth'
                ],
        
        # learning hypers
        'initLrD': 0.00003,
        'initLrG': 0.00007,


}


penalties = {
        'PenaltyGeneratorWeightRegularizer': 0.0001,
        'PenaltyDiscriminatorWeightRegularizer':0.0003,
        'PenaltyReconstructionL1':3,
        'PenaltyConstContent':0.2,
        'PenaltyConstStyle':0.2,
        'PenaltyDiscriminatorCategory': 0,
        'GeneratorCategoricalPenalty': 0.,
        'PenaltyVaeKl': 1,        
        'PenaltyContentFeatureExtractor': [1,1,1,1,1],
        'PenaltyStyleFeatureExtractor':[1,1,1,1,1],
        'PenaltyAdversarial':1,
        'PenaltyDiscriminatorPenalty':10
        
}

