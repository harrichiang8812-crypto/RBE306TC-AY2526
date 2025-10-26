import os
import sys
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import yaml
sys.path.append('./')
from Tools.Utilities import cv2torch, read_file_to_dict, string2tensor
from tqdm import tqdm
from torchvision.transforms import InterpolationMode
import glob


class FeatureExtractorDataset(Dataset):
    def __init__(self,config,is_train, augmentation):
        self.type = config['type']
        self.augmentation = augmentation
       
        self.content_yaml = os.path.join(config['YamlPackages'], 'Content.yaml')
        
        filesIdentified = glob.glob(config['YamlPackages']+"*.yaml")
        self.gtYamlTrain = [ii for ii in filesIdentified if 'TrainGroundTruth' in ii]
        self.gtYamlTest = [ii for ii in filesIdentified if 'TestGroundTruth' in ii]
        
        
        # self.train_style_yaml = os.path.join(config['YamlPackages'], 'TrainStyleReference.yaml')
        # self.test_style_yaml = os.path.join(config['YamlPackages'], 'TestStyleReference.yaml')
       
        # if config.debug:
        #     self.content_yaml = os.path.join(config.yamls, 'Content.yaml')
        #     self.train_style_yaml = os.path.join(config.yamls, 'TrainStyleReference.yaml')
        #     self.test_style_yaml = os.path.join(config.yamls, 'TestStyleReference.yaml')
        # else:
        #     self.content_yaml = os.path.join(config.yamls, 'Content.yaml')
        #     self.train_style_yaml = os.path.join(config.yamls, 'TrainStyleReference.yaml')
        #     self.test_style_yaml = os.path.join(config.yamls, 'TestStyleReference.yaml')
            
        # self.content_yaml = os.path.join(config.yamls, 'Content.yaml')
        # self.train_style_yaml = os.path.join(config.yamls, 'TrainStyleReference.yaml')
        # self.test_style_yaml = os.path.join(config.yamls, 'TestStyleReference.yaml')
        self.is_train = is_train
        self.dataset = []
        #self.num = config.outputnum
        
        self.order = read_file_to_dict(config['labelVecTxt'])
        
        
        if 'content' in self.type:
            with open(self.content_yaml, 'r', encoding='utf-8') as f:
                print("Loading "+ self.content_yaml + '...', end='\r')
                files = yaml.load(f.read(), Loader=yaml.FullLoader)
                print("Loading "+ self.content_yaml + ' completed.')
                
                for idx, (k,values) in tqdm(enumerate(files.items()),  total=len(files.items()), desc="Load the Train set"):
                    for file_path in values:
                        # file_path = os.path.join(data_path,value)
                        self.train_set.append((file_path ,self.order[k]))

        elif 'style' in self.type:
            # files = {}
            # self.num = config.styleNum
            # order_txt = cfg['Label1_list']
            # data_path = os.path.join(base_dir,style_dir)
            # 读取参考样式的YAML文件
            # trainFiles=[]
            for _file in self.gtYamlTrain:
                with open(_file, 'r', encoding='utf-8') as f:
                    print("Loading "+ _file + '...', end='\r')
                    thisTrainFile=yaml.load(f.read(), Loader=yaml.FullLoader)
                    for idx, (key, values) in tqdm(enumerate(thisTrainFile.items()),  total=len(thisTrainFile), desc="Load the Train set"):
                        _path, _label0, _label1 = values
                        self.dataset.append((_path ,self.order[_label1]))
                    
                    print("Loading "+ _file + ' completed.')
                    
                    

            # 读取验证样式的YAML文件
            # testFiles=[]
            for _file in self.gtYamlTest:
                with open(_file, 'r', encoding='utf-8') as f:
                    print("Loading "+ _file + '...', end='\r')
                    thisTestFile=yaml.load(f.read(), Loader=yaml.FullLoader)
                    for idx, (key, values) in tqdm(enumerate(thisTestFile.items()),  total=len(thisTestFile), desc="Load the Train set"):
                        _path, _label0, _label1 = values
                        self.dataset.append((_path ,self.order[_label1]))
                    
                    # testFiles.append(yaml.load(f.read(), Loader=yaml.FullLoader))
                    # files.update(train_files)  # 将内容更新到files字典中
                    print("Loading "+ _file + ' completed.')

        

            # for thisTrainFile in trainFiles:
            #     for idx, (key, values) in tqdm(enumerate(thisTrainFile.items()),  total=len(thisTrainFile), desc="Load the Train set"):
            #         _path, _label0, _label1 = values
            #         self.train_set.append((_path ,self.order[_label1]))
                    
                    
            #         # for file_path in values:
            #         #     # file_path = os.path.join(data_path,value)
            #         #     self.train_set.append((file_path ,self.order[k]))
                    
            # for thisTestFile in testFiles:
            #     for idx, (k,values) in tqdm(enumerate(thisTestFile.items()),  total=len(thisTestFile), desc="Load the Test set"):
            #         for file_path in values:
            #             # file_path = os.path.join(data_path,value)
            #             self.train_set.append((file_path ,self.order[k]))


    def __getitem__(self, index):
        if self.augmentation=='NA':
            transform = transforms.Compose([                
                transforms.ToTensor(),  # 将 numpy.ndarray 转换为 torch.Tensor
                transforms.Normalize((0.5,), (0.5,)),  # 归一化到 [-1, 1]
            ])
        elif self.augmentation == 'Half':
                transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomResizedCrop(size=(64,64), scale=(0.75,1.0), antialias=True),                
                transforms.ToTensor(),  # 将 numpy.ndarray 转换为 torch.Tensor
                transforms.Normalize((0.5,), (0.5,)),  # 归一化到 [-1, 1]
            ])
        elif self.augmentation == 'Full':
                transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomAffine(degrees=(-15,15), interpolation=InterpolationMode.BICUBIC, fill=(255), center=(32,32), translate=(0.15,0.15), scale=(0.75,1.25), shear=(15,15)),
                transforms.RandomResizedCrop(size=(64,64), scale=(0.75,1.0), antialias=True),                
                transforms.ToTensor(),  # 将 numpy.ndarray 转换为 torch.Tensor
                transforms.Normalize((0.5,), (0.5,)),  # 归一化到 [-1, 1]
                
            ])
        image = cv2torch(self.dataset[index][0],transform)
        label = string2tensor(self.dataset[index][1])

        raw_label = list(self.order.keys())[label]
        if not self.is_train:
            return image,label,raw_label
        else:
            return image,label
    
    def __len__(self):
        return len(self.dataset) 
    
    
# if __name__ == "__main__":
#     cfg['content_yaml'] = 'cmy/test_list/content_dir.yaml'
#     cfg['GT_style_yaml'] = 'cmy/test_list/train_GT_dir.yaml'
#     cfg['reference_style_yaml'] = 'cmy/test_list/train_reference_style_dir.yaml'
#     casia_dataset = Feature_Dataset(cfg,type='content',is_train=False)
#     casia_loader = DataLoader(casia_dataset, batch_size=8, shuffle=False,drop_last=True)
#     for image,label in casia_loader:
#         print(label)