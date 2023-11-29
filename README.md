# FFC3DLungDIR: Fast Fourier Convolution-based Model for 4D-CT Lung Deformable Image Registration

This is the code for 4D-CT Lung Deformable Image Registration.

## Proposed Framework

You can see the overall framework named FFC3DLungDIR which has been used in this study. Inspired by, [Fast Fourier Convolution](https://papers.nips.cc/paper/2020/hash/2fd5d41ec6cfab47e32164d5624269b1-Abstract.html) paper, we utilized the 3D version of this operator (FFC3D) to increase the receptive field of typical convolution operators to capture global and local information in feature maps simultaneously.

<p align="center">
  <img src="https://github.com/mohammadhashemii/FFC3DLungDIR/blob/main/documents/FFC3DLungDIR_framework.png">	
</p>

## Quick starts

### Requirements

In order to install the required packages, run this command in your cmd or terminal:

`pip install -r requirements.txt`

### Data preparation

For training the model, we have used [**CREATIS**](https://www.creatis.insa-lyon.fr/rio/popi-model) dataset consisting 6 sets of 4D-CT lung images. As a test set, we utilized publicly available [**DIRLAB**](http://www.dir-lab.com/) dataset including 10 sets of 4D-CT lung images each containing images of ten respiratory cycle phases.

### Training

Configure [`FFCResnetGenerator_settings.yaml`](https://github.com/mohammadhashemii/FFC3DLungDIR/blob/main/configs/FFCResnetGenerator_settings.yaml) to build your desired architecture. Also, for setting the training parameters e.g., hyperparameters, modify [`training_settings.yaml`](https://github.com/mohammadhashemii/FFC3DLungDIR/blob/main/configs/training_settings.yaml). Then, for training it, run [`train.py`](https://github.com/mohammadhashemii/FFC3DLungDIR/blob/main/train.py) using the following command in your terminal or cmd:

```
python train.py --exp [experiment number] --training_config_path path/to/model/config --model_config_path [path/to/model/config] --training_config_path [path/to/training/config]
```

### Qualitative results

<p align="center">
  <img src="https://github.com/mohammadhashemii/FFC3DLungDIR/blob/main/documents/intensity_difference.png">	
</p>
