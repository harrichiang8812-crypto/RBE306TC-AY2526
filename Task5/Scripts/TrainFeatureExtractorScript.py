import argparse
import os
import sys

sys.path.append('../')  # Add the parent directory to the system path to access required modules
os.chdir(sys.path[0])  # Change the current directory to the script path


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter
from Networks.FeatureExtractor.FeatureExtractorBase import FeatureExtractorBase as FeatureExtractor
from Pipelines.DatasetFeatureExtractor import FeatureExtractorDataset as Dataset
from tqdm import tqdm
sys.path.append('./')
import shutil



NUM_SAMPLE_PER_EPOCH = 1000  # Number of samples processed per epoch
RECORD_PCTG = NUM_SAMPLE_PER_EPOCH / 5  # Percentage of samples to record during training




def main(cfg):
    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if cfg['type'] == 'content':
        cfg['labelVecTxt']= '/data-shared/server09/data0/haochuan/CASIA_Dataset/LabelVecs/PF64-Label0.txt'
        outputnum=3755
    elif cfg['type'] == 'style-hw300':
        cfg['labelVecTxt']= '/data-shared/server09/data0/haochuan/CASIA_Dataset/LabelVecs/HW300-Label1.txt'
        outputnum=300
    elif cfg['type'] == 'style-pf80':
        cfg['labelVecTxt']= '/data-shared/server09/data0/haochuan/CASIA_Dataset/LabelVecs/PF80-Label1.txt'
        outputnum=80
        
    # 加载数据集
    workersNum=12
    batchsize=512
    train_dataset = Dataset(cfg, is_train=True, augmentation='Full')
    train_loader = DataLoader(train_dataset, batch_size=batchsize, 
                              num_workers=workersNum, pin_memory=True, 
                              shuffle=True, drop_last=False, 
                              persistent_workers=True)
    modelSavePath=cfg['modelSavePath']
    modelLogPath=cfg['modelSavePath']+'Logs'

    # 初始化模型
    model = FeatureExtractor(outputNums=outputnum, modelSelect=cfg['network'], type=cfg['type']).to(device)  
    model.train()
    if 'content' in cfg['type']:
        modelSavePath = os.path.join(modelSavePath, 'Content')
        modelLogPath = os.path.join(modelLogPath, 'Content')
    elif 'style' in cfg['type']:
        modelSavePath = os.path.join(modelSavePath, 'Style')
        modelLogPath = os.path.join(modelLogPath, 'Style')
    modelSavePath = os.path.join(modelSavePath, cfg['network'])
    modelLogPath = os.path.join(modelLogPath, cfg['network'])
    
    # logPath = os.path.join(modelSavePath, 'Logs')
    if cfg['resume'] and os.path.exists(modelSavePath):
        model.load_state_dict(torch.load(os.path.join(modelSavePath, "style_best_extractor_model.pth")))
    else:
        if os.path.exists(modelSavePath):
            shutil.rmtree(modelSavePath)
            shutil.rmtree(modelLogPath)
        os.makedirs(modelSavePath, exist_ok=True)
        os.makedirs(modelLogPath, exist_ok=True)
    writer = SummaryWriter(log_dir=modelLogPath)


    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 初始化最佳loss为正无穷
    best_loss = float('inf')
    num_epochs = cfg['epochs']
    
    

    # 训练模型
    for epoch in range(num_epochs):
        running_loss = 0.0
        thisRoundStartItr = 0  # Used to track when to write summaries
        for idx, (images, labels ) in tqdm(enumerate(train_loader), 
                                           total=len(train_loader),
                                           desc=cfg['network']+ ': Training Epoch: %d/%d' % (epoch+1, num_epochs)):
            images, labels = images.to(device).float(), labels.to(device)
            optimizer.zero_grad()
            
            outputs,_ = model.extractor(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if idx==0 or idx == len(train_loader)-1 or \
                idx * float(NUM_SAMPLE_PER_EPOCH) / len(train_loader) - thisRoundStartItr > RECORD_PCTG:
                thisRoundStartItr = idx * float(NUM_SAMPLE_PER_EPOCH) / len(train_loader)
                writer.add_scalar('Train/Loss', loss.item(), 
                                  epoch * NUM_SAMPLE_PER_EPOCH + int(idx / (len(train_loader)) * NUM_SAMPLE_PER_EPOCH)+1)
            
            running_loss += loss.item()

        # 计算平均loss
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}')
    
        # 如果当前loss是最佳的，保存模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(modelSavePath, "BestExtractor.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f'Saved Best Model at Epoch {epoch+1} with Loss: {best_loss}')

    # 保存最后一个epoch的模型
    final_model_path = os.path.join(modelSavePath, "FinalExtractor.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f'Saved Final Model at Epoch {num_epochs}')

    # 输出最终的loss
    print(f'Final Loss: {avg_loss}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str,  default='2', help='GPU to use')
    parser.add_argument('--modelSavePath',type=str, default='/data-shared/server09/data1/haochuan/Character/PtTrainedFeatureExtractors/Debug/',help='checkpoint path')
    parser.add_argument('--type',type=str,default='style',help='')
    parser.add_argument('--network',type=str,default='VGG11Net',help='')
    parser.add_argument('--epochs',type=int,default=500,help='')
    parser.add_argument('--resume',type=int,default=0,help='')
    parser.add_argument('--YamlPackages',type=str,default=0,help='')
    
    args = parser.parse_args()
    cfg={}
    cfg['gpu'] = args.gpu
    cfg['modelSavePath'] = args.modelSavePath
    cfg['type'] = args.type
    cfg['network'] = args.network
    cfg['resume'] = args.resume

    # cfg['YamlPackages'] = '../YamlLists/Debug/'
    cfg['YamlPackages'] = '../YamlLists/'+ args.YamlPackages + '/'
    cfg['epochs']=args.epochs
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu']
    main(cfg)