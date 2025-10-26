import os
dataPathRoot = '/data0/haochuan/'


hyperParams = {
        'debugMode':1,
        'expID':'20250509PF-Batch243248',# experiment name prefix
        'expDir': '/data-shared/server01/data1/haochuan/CharacterRecords2025May-051/',

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
                '/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/20240928New/Ckpts/Content/VGG11Net/BestExtractor.pth',
                '/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/20240928New/Ckpts/Content/VGG13Net/BestExtractor.pth',
                '/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/20240928New/Ckpts/Content/VGG16Net/BestExtractor.pth',
                '/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/20240928New/Ckpts/Content/VGG19Net/BestExtractor.pth',
                '/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/20240928New/Ckpts/Content/ResNet18/BestExtractor.pth',
                '/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/20240928New/Ckpts/Content/ResNet34/BestExtractor.pth',
                '/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/20240928New/Ckpts/Content/ResNet50/BestExtractor.pth'
                ],
        
        'StyleExtractorPath':  [
                '/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/20240928New/Ckpts/Style/VGG11Net/BestExtractor.pth',
                '/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/20240928New/Ckpts/Style/VGG13Net/BestExtractor.pth',
                '/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/20240928New/Ckpts/Style/VGG16Net/BestExtractor.pth',
                '/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/20240928New/Ckpts/Style/VGG19Net/BestExtractor.pth',
                '/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/20240928New/Ckpts/Style/ResNet18/BestExtractor.pth',
                '/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/20240928New/Ckpts/Style/ResNet34/BestExtractor.pth',
                '/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/20240928New/Ckpts/Style/ResNet50/BestExtractor.pth'
                ],
        
        # learning hypers
        'initLrD': 0.00002,
        'initLrG': 0.00005


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
        'PenaltyContentFeatureExtractor': [1,1,1,1,1,1,1],
        'PenaltyStyleFeatureExtractor':[1,1,1,1,1,1,1],
        'PenaltyAdversarial':0.7,
        'PenaltyDiscriminatorPenalty':10
        
}

