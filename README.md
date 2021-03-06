# MU-Net
This is an implementation of our paper: MU-Net: A multiscale unsupervised network for remote sensing image registration.
![Proposed Framework in the Paper](https://github.com/yeyuanxin110/MU-Net/blob/main/githubPic/MU-Net.png)
## Preparation
Our code is performed in Pytorch 1.8.0 basis on Python 3.8.
## Introduction
network.py: Our DNN architectures, implemented on three scales.

generation.py:  Generate the trainging or testing data (image pairs) by datasets provided by the paper or your own datasets. 

dataset.py: Loading data process during training or testing.

loss.py: Store various loss functions.

train.py : Training Process.

STN.py: Similarity,Affine or Homography Transformation based on STN.

descriptor: Store the CFOG or LSS dense descriptor. To use them, you may need to install matlab calling program in your Python.

## Datasets
The multi-modal original image pairs adopted in the paper have been uploaded to Google Drive. You could download them and put them into generation.py to generate the training or testing image pairs.

![Optical-Optical dataset](https://github.com/yeyuanxin110/MU-Net/blob/main/githubPic/Optical-Optical.png)
Optical-Optical dataset: https://drive.google.com/file/d/1U0fpCnizcl33TgdRwvfQpqOr1Ojcj6a9/view?usp=sharing

![Optical-Infrared dataset](https://github.com/yeyuanxin110/MU-Net/blob/main/githubPic/Optical-Infrared.png)
Optical-Infrared dataset: https://drive.google.com/file/d/1c4Ao4CoMerntNVf2Qn3hY0eEtwURh8iM/view?usp=sharing

![Optical-SAR dataset](https://github.com/yeyuanxin110/MU-Net/blob/main/githubPic/Optical-SAR.png)
Optical-SAR dataset: https://drive.google.com/file/d/181IEtG6ciBsQGhM6TgEDfv8yglAWsKxy/view?usp=sharing

![Optical-RasterMap dataset](https://github.com/yeyuanxin110/MU-Net/blob/main/githubPic/Optical-Map.png)
Optical-RasterMap dataset: https://drive.google.com/file/d/1kIqXy3-KCTLwaPaxTrEFKSt49LvZnWAU/view?usp=sharing
