import sys
import torch
#torch.set_warn_always(False)
#import warnings
#warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
#import shutup
#shutup.please()

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np

sys.path.append('./')
from Networks.FeatureExtractor.FeatureExtractorBase import FeatureExtractorBase as FeatureExtractor
from Networks.PlainGenerators.PlainWNetBase import WNetGenerator
HighLevelFeaturePenaltyPctg = [0.1, 0.15, 0.2, 0.25, 0.3]
eps = 1e-9
MAX_gradNorm = 1.0
MIN_gradNorm = 1e-1
GP_LAMBDA = 10

from Tools.Utilities import PrintInfoLog
import copy

class Loss(nn.Module):
    def __init__(self, config, sessionLog, penalty, EpochSegs):
        super(Loss, self).__init__()
        
        self.sessionLog = sessionLog
        self.warmup, self.ramp = EpochSegs

        # penalities
        self.PenaltyReconstructionL1 = penalty['PenaltyReconstructionL1']
        self.PenaltyConstContent = penalty['PenaltyConstContent']
        self.PenaltyConstStyle = penalty['PenaltyConstStyle']
        self.GeneratorCategoricalPenalty = penalty['GeneratorCategoricalPenalty']
        self.PenaltyContentFeatureExtractor = penalty['PenaltyContentFeatureExtractor']
        self.PenaltyStyleFeatureExtractor = penalty['PenaltyStyleFeatureExtractor']
        self.PenaltyVaeKl = penalty['PenaltyVaeKl']
        self.adversarial = penalty['PenaltyAdversarial']
        self.PenaltyDiscriminatorPenalty = penalty['PenaltyDiscriminatorPenalty']
        
        self.contentExtractorList = []
        if 'extractorContent' in config:
            PrintInfoLog(self.sessionLog, "Content Encoders: ", end='')
            counter = 0
            for contentExtractor in config['extractorContent']:
                thisContentExtractor = FeatureExtractor(outputNums=len(config.datasetConfig.loadedLabel0Vec), 
                                                        modelSelect=contentExtractor.name,
                                                        type='content').extractor
                thisContentExtractor.eval()
                thisContentExtractor.cuda()
                self.NameMappingLoading(thisContentExtractor, contentExtractor.path)
                self.contentExtractorList.append(copy.deepcopy(thisContentExtractor))
                if counter != len(config['extractorContent']) - 1:
                    PrintInfoLog(self.sessionLog, ", ", end='')
                counter = counter + 1
            PrintInfoLog(self.sessionLog, "Loaded.")
                    
        self.styleExtractorList = []
        if 'extractorStyle' in config:
            PrintInfoLog(self.sessionLog, "Style Encoders: ", end='')
            counter = 0
            for styleExtractor in config['extractorStyle']:
                thisStyleExtractor = FeatureExtractor(outputNums=len(config.datasetConfig.loadedLabel1Vec), 
                                                        modelSelect=styleExtractor.name,
                                                        type='style').extractor
                thisStyleExtractor.eval()
                thisStyleExtractor.cuda()
                self.NameMappingLoading(thisStyleExtractor, styleExtractor.path)
                self.styleExtractorList.append(copy.deepcopy(thisStyleExtractor))
                if counter != len(config['extractorStyle']) - 1:
                    PrintInfoLog(self.sessionLog, ", ", end='')
                counter = counter + 1
            PrintInfoLog(self.sessionLog, "Loaded.")

    def NameMappingLoading(self, extractor, path):
        loaded = torch.load(path)
        loadedItems = list(loaded.items())
        thisExtractorDict = extractor.state_dict()
        count = 0
        for key, value in thisExtractorDict.items():
            layer_name, weights = loadedItems[count]      
            thisExtractorDict[key] = weights
            count = count + 1
        extractor.load_state_dict(thisExtractorDict)
        PrintInfoLog(self.sessionLog, path.split('/')[-2], end='')

    def GeneratorLoss(self, encodedContentFeatures, encodedStyleFeatures, encodedContentCategory, encodedStyleCategory,
                      generated, GT, content_onehot, style_onehot, lossInputs=None):
        # l1 loss
        lossL1 = torch.mean(torch.abs(generated - GT))
        
        # const_content loss
        GT_content_enc = encodedContentFeatures['groundtruth']
        lossConstContentReal = F.mse_loss(encodedContentFeatures['real'], GT_content_enc)
        lossConstContentFake = F.mse_loss(encodedContentFeatures['fake'], GT_content_enc)

        # const_style loss
        GT_style_enc = encodedStyleFeatures['groundtruth']
        reshaped_styles = encodedStyleFeatures['real']  # batchsize * input style num * channels * width * height
        reshaped_styles = reshaped_styles.permute(1, 0, 2, 3, 4)  # input style num *batchsize * channels * width * height
        lossConstStyleReal = [F.mse_loss(enc_style, GT_style_enc) for enc_style in reshaped_styles]
        lossConstStyleReal = torch.mean(torch.stack(lossConstStyleReal))
        lossConstStyleFake = F.mse_loss(encodedStyleFeatures['fake'], GT_style_enc)

        # category loss
        lossCategoryContentReal, lossCategoryContentFake = 0, 0
        GT_content_category = encodedContentCategory['groundtruth']
        for fake_logits, GT_logits, onehot in zip(encodedContentCategory['real'], GT_content_category, content_onehot):
            GT_logits = torch.nn.functional.softmax(GT_logits, dim=0)
            lossCategoryContentReal = lossCategoryContentReal + F.cross_entropy(GT_logits, onehot)
            fake_logits = torch.nn.functional.softmax(fake_logits, dim=0)            
            lossCategoryContentFake = lossCategoryContentFake + F.cross_entropy(fake_logits, onehot)
        lossCategoryContentReal = lossCategoryContentReal/len(GT_content_category)
        lossCategoryContentFake = lossCategoryContentFake/len(GT_content_category)

        lossCategoryStyleReal, lossCategoryStyleFake = 0, 0
        GT_style_category = encodedStyleCategory['groundtruth']
        for fake_logits, GT_logits, onehot in zip(encodedStyleCategory['real'], GT_style_category, style_onehot):
            GT_logits = torch.nn.functional.softmax(GT_logits, dim=0)
            lossCategoryStyleReal = lossCategoryStyleReal + F.cross_entropy(GT_logits, onehot)
            fake_logits = torch.nn.functional.softmax(fake_logits, dim=0)            
            lossCategoryStyleFake = lossCategoryStyleFake + F.cross_entropy(fake_logits, onehot)
        lossCategoryStyleReal = lossCategoryStyleReal/len(GT_style_category)
        lossCategoryStyleFake = lossCategoryStyleFake/len(GT_style_category)
        
        losses = [lossL1, lossConstContentReal, lossConstStyleReal, lossConstContentFake, lossConstStyleFake,
                  lossCategoryContentReal, lossCategoryContentFake, lossCategoryStyleReal, lossCategoryStyleFake]
        
        return losses


    def CriticLoss(self, realScore, fakeScore):
        """
        Computes WGAN Critic loss.
        
        Args:
            gt: Real images (ground truth)
            generated: Generated images
            
        Returns:
            real_scores: Scores on real images
            fake_scores: Scores on generated images
            critic_loss: Wasserstein loss for the Critic
        """
        
        critic_loss = torch.mean(fakeScore) - torch.mean(realScore)

        return critic_loss
    
    def ComputePenaltyDiscriminatorPenalty(self, discriminator, real_samples, fake_samples, lambda_gp=10):
        """
        Calculates the gradient penalty loss for WGAN-GP.
        
        Args:
            real_samples: A batch of real images (B, C, H, W)
            fake_samples: A batch of fake images (B, C, H, W)
            lambda_gp: Gradient penalty coefficient (default 10)
        
        Returns:
            gradient_penalty: A scalar tensor for gradient penalty loss
        """
        batch_size = real_samples.size(0)
        
        # Random interpolation between real and fake samples
        alpha = torch.rand(batch_size, 1, 1, 1, device=real_samples.device)
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        
        # Pass interpolated samples through the critic
        d_interpolates = discriminator(interpolates)
        
        # For WGAN, the output should be (B, 1), so we need gradients w.r.t interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]  # Take the first element from returned tuple
        
        # Gradients: (B, C, H, W) → compute norm per sample
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)  # L2 norm per sample
        
        # Compute penalty: (|gradient_norm - 1|)^2
        gradient_penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()
        
        return gradient_penalty
    
    def FeatureExtractorLoss(self, GT, imgFake):
        # content_extractor
        contentSumMSE = 0.0
        contentMSEList = []
        for idx1, thisContentExtractor in enumerate(self.contentExtractorList):
            thisContentMSE = 0
            with torch.no_grad():
                _, GT_content_features = thisContentExtractor(GT)
                _, fake_content_features = thisContentExtractor(imgFake)
            if not len(HighLevelFeaturePenaltyPctg) == len(GT_content_features) == len(fake_content_features):
                PrintInfoLog(self.sessionLog, 'content length not paired')
                return
            for idx2, (GT_content_feature, fake_content_feature) in enumerate(zip(GT_content_features, fake_content_features)):
                thisContentMSE = thisContentMSE + F.mse_loss(GT_content_feature, fake_content_feature) * HighLevelFeaturePenaltyPctg[idx2]
            thisContentMSE = thisContentMSE/sum(HighLevelFeaturePenaltyPctg)
            contentMSEList.append(thisContentMSE)
            contentSumMSE = contentSumMSE + thisContentMSE * self.PenaltyContentFeatureExtractor[idx1]
        contentSumMSE = contentSumMSE / (sum(self.PenaltyContentFeatureExtractor) + eps)

        # style_extractor
        styleSumMSE = 0.0
        styleMSEList = []
        for idx1, thsiStyleExtractor in enumerate(self.styleExtractorList):
            thisStyleMSE = 0
            with torch.no_grad():
                _, GT_style_features = thsiStyleExtractor(GT)
                _, fake_style_features = thsiStyleExtractor(imgFake)
            if not len(HighLevelFeaturePenaltyPctg) == len(GT_style_features) == len(fake_style_features):
                PrintInfoLog(self.sessionLog, 'style length not paired')
                return
            for idx2, (GT_style_feature, fake_style_feature) in enumerate(zip(GT_style_features, fake_style_features)):
                thisStyleMSE = thisStyleMSE + F.mse_loss(GT_style_feature, fake_style_feature) * HighLevelFeaturePenaltyPctg[idx2]
            thisStyleMSE = thisStyleMSE/sum(HighLevelFeaturePenaltyPctg)
            styleMSEList.append(thisStyleMSE)
            styleSumMSE = styleSumMSE + thisStyleMSE * self.PenaltyStyleFeatureExtractor[idx1]
        styleSumMSE = styleSumMSE / (sum(self.PenaltyStyleFeatureExtractor) + eps)

        return contentSumMSE, styleSumMSE, contentMSEList, styleMSEList

    def forward(self, epoch, 
                lossInputs=None, mixerArchitecture=[], critic=None, mode='NA'):
        
        
        encodedContentFeatures,encodedContentCategory=lossInputs['encodedContents']
        encodedStyleFeatures,encodedStyleCategory=lossInputs['encodedStyles']
        generated = lossInputs['fake']
        GT = lossInputs['GT']
        content_onehot, style_onehot = lossInputs['oneHotLabels']
        
        # 计算其它损失
        losses = self.GeneratorLoss(encodedContentFeatures, encodedStyleFeatures, encodedContentCategory, encodedStyleCategory,
                                    generated, GT, content_onehot, style_onehot, lossInputs=lossInputs)
        lossL1, lossConstContentReal, lossConstStyleReal, lossConstContentFake, lossConstStyleFake, \
        lossCategoryContentReal, lossCategoryContentFake, lossCategoryStyleReal, lossCategoryStyleFake = losses

        # 计算其它损失
        deepPerceptualContentSum, deepPerceptualStyleSum, contentMSEList, styleMSEList = \
            self.FeatureExtractorLoss(GT=GT, imgFake=generated)

        
        
        deepLossContent = sum([x * y for x, y in zip(contentMSEList, self.PenaltyContentFeatureExtractor)])/sum(self.PenaltyContentFeatureExtractor)
        deepLossStyle = sum([x * y for x, y in zip(styleMSEList, self.PenaltyStyleFeatureExtractor)])/sum(self.PenaltyContentFeatureExtractor)

        sumLossG = lossL1 * self.PenaltyReconstructionL1 + \
                (lossConstContentReal + lossConstContentFake) * self.PenaltyConstContent + \
                (lossConstStyleReal + lossConstStyleFake) * self.PenaltyConstStyle + \
                    +deepLossContent+deepLossStyle
                
        # sumLossG_Warmup = lossL1 * self.PenaltyReconstructionL1 + \
        # (lossConstContentReal + lossConstContentFake) * self.PenaltyConstContent + \
        # (lossConstStyleReal + lossConstStyleFake) * self.PenaltyConstStyle + \
        # fullKL * self.PenaltyVaeKl

        lossDict = {'lossL1': lossL1,
                    'lossConstContentReal': lossConstContentReal,
                    'lossConstStyleReal': lossConstStyleReal,
                    'lossConstContentFake': lossConstContentFake,
                    'lossConstStyleFake': lossConstStyleFake,
                    'lossCategoryContentReal': lossCategoryContentReal,
                    'lossCategoryContentFake': lossCategoryContentFake,
                    'lossCategoryStyleReal': lossCategoryStyleReal,
                    'lossCategoryStyleFake': lossCategoryStyleFake,
                    'deepPerceptualContent': deepPerceptualContentSum,
                    'deepPerceptualStyle': deepPerceptualStyleSum,
                    'deepPerceptualContentList': contentMSEList,
                    'deepPerceptualStyleList': styleMSEList,
                    'sumLossG': sumLossG}
      
        return lossDict

    def calculate_kl_divergence(self, vaeMeans, vaeLogVars):
        """
        计算 VAE 的 KL 散度损失
        """
        # kl_loss = 0.0
        # for mean, log_var in zip(vaeMeans, vaeLogVars):
        #     kl_loss = kl_loss + (-0.5) * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        # return kl_loss
        return (-0.5) * torch.sum(1 + vaeLogVars - vaeMeans.pow(2) - vaeLogVars.exp())

    def GetAdvPenalty(self, epoch):
        if epoch < self.warmup:
            return 0.0
        elif epoch < self.warmup + self.ramp:
            return self.adversarial * (epoch - self.warmup + 1) / self.ramp
        else:
            return self.adversarial