# Vision Transformer

## Introduction

This repository is incorporated from the origin: https://github.com/jeonsworld/ViT-pytorch
Please be advised to only use this repository for your RBE306TC Task 4 as ACADEMIC AND RESEARCH PURPOSE ONLY.
This repository is NOT ALLOWED for any commercial applications.

![ViT.png](https://s2.loli.net/2022/01/19/w3CyXNrhEeI7xOF.png)

Network for Vision Transformer. The pytorch version. 

If this works for you, please give me a star, this is very important to me.😊

## Quick start


1. Install respective anaconda envnironment.

```shell
cd torch_Vision_Transformer
pip install -r requirements.txt
```
2. Install Pytorch from pytorch.org with respective command displayed.

3. [Optional] Download the **flower dataset**.
```shell
wget https://github.com/Runist/image-classifier-keras/releases/download/v0.2/dataset.zip
unzip dataset.zip
```
4. Modifying the [config.py](https://github.com/harrichiang8812-crypto/RBE306TC-AY2526/blob/main/Task4/config.py).

5. Find pretrain weights via [/data-shared/NAS/RBE306TC/Tools/PretrainViTs/].
6. Start train your model.

```shell
python train.py
```
7. Open tensorboard to watch loss, learning rate etc. You can also see training process and training process and validation prediction.

```shell
tensorboard --logdir ./summary/log
```
![tensorboard.png](https://s2.loli.net/2022/10/12/p7KtB1uXMkqvreN.png)

8. Get prediction of model.

```shell
python predict.py
```

## Train your dataset

You need to store your data set like this:

```shell
├── train
│   ├── daisy
│   ├── dandelion
│   ├── roses
│   ├── sunflowers
│   └── tulips
└── validation
    ├── daisy
    ├── dandelion
    ├── roses
    ├── sunflowers
    └── tulips
```

## Reference

Appreciate the work from the following repositories:

- [WZMIAOMIAO](https://github.com/WZMIAOMIAO)/[vision_transformer](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/vision_transformer)


## License

Code and datasets are released for non-commercial and research purposes **only**. For commercial purposes, please contact the authors.
