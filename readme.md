# SDA-YOLO
Codes for my paper "SDA-YOLO: Semi-supervised Domain Adaptive YOLO for Cross-Domain Object Detection" for ICME2022

## Abstarct

*The cross-domain discrepancy has always been what domain adaptive object detection aims to alleviate. This problem is more prominent when a significant distribution difference appearing between the source data used for training and the target data from the real application scenario. In this paper, we propose a novel semi-supervised domain adaptive YOLO (SDA-YOLO) method to improve cross-domain detection performance by integrating the efficient YOLOv5. Specifically, we adapt knowledge distillation to assist the student model obtain the instance-level features in the unlabeled target domain. We also draw support from style transfer to cross-generate pseudo images in different domains for remedying the image-level differences. We evaluate our proposed SDA-YOLO on public benchmarks including PascalVOC, Clipart1k, Cityscapes, and Foggy Cityscapes. Moreover, the experimental details on the yawning detection dataset collected in various schools and classrooms are also reported. The final results show the impressive improvement of our method in cross-domain obejct detection task.*

## Brief Description

SDA-YOLO is designed for domain adaptative cross-domain object detection based on the knowledge distillation framework and robust `YOLOv5`. The network architecture is as below. 

![example1](./images/figure1.png)

So far, we have trained and evaluated it on two adaptation tasks: **PascalVOC → Clipart1k**, **CityScapes → CityScapes Foggy**.

## Installition

**Environment:** Anaconda, Python3.8, PyTorch1.10.0(CUDA11.2), wandb

```bash
$ git clone https://github.com/hnuzhy/SDA-YOLO.git
$ pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Codes are only evaluated on GTX3090 with CUDA11.2 and PyTorch1.10.0. You can install the same verison if needed
# [method 1][directly install from the official website][may slow]
$ pip3 install torch==1.10.0+cu111 torchvision==0.11.1+cu111 torchaudio==0.10.0+cu111 \
  -f https://download.pytorch.org/whl/cu111/torch_stable.html
  
# [method 2]download from the official website and install offline][faster]
$ wget https://download.pytorch.org/whl/cu111/torch-1.10.0%2Bcu111-cp38-cp38-linux_x86_64.whl
$ wget https://download.pytorch.org/whl/cu111/torchvision-0.11.1%2Bcu111-cp38-cp38-linux_x86_64.whl
$ wget https://download.pytorch.org/whl/cu111/torchaudio-0.10.0%2Bcu111-cp38-cp38-linux_x86_64.whl
$ pip3 install torch*.whl
```

## Dataset Preparing

**PascalVOC → Clipart1k**

* **PascalVOC(2007+2012)**: Please follow the instructions in [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to prepare VOC datasets. Or you can follow the scripts in file [VOC.yaml](./data/yamls_bak/VOC.yaml) to build VOC datasets.
* **Clipart1k**: This datast is originally released in [cross-domain-detection](https://github.com/naoto0804/cross-domain-detection). Dataset preparation instruction is also in it [Cross Domain Detection/datasets](https://github.com/naoto0804/cross-domain-detection/tree/master/datasets).
* **VOC-style → Clipart-style**: Images translated by [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) are available in the website [dt_clipart](https://github.com/naoto0804/cross-domain-detection/tree/master/datasets#download-domain-transferred-images-for-step1-cyclegan) by running `bash prepare_dt.sh`.
* **Clipart-style → VOC-style**: We trained a new image style transfer model based on [CUT(ECCV2020)](https://github.com/taesungp/contrastive-unpaired-translation). The generated 1k VOC-style images are uploaded in [google drive](https://drive.google.com/drive/folders/1Z5Wv6SV-atBNEsi_zBprlg0uVIw3EKGA?usp=sharing).
* **VOC foramt → YOLOv5 format**: Change labels and folders placing from VOC foramt to YOLOv5 format. Follow the script [convert_voc2clipart_yolo_label.py](./data/formats/convert_voc2clipart_yolo_label.py)

**CityScapes → CityScapes Foggy**

* **CityScapes**: Download from the official [website](https://www.cityscapes-dataset.com/downloads/). Images ***leftImg8bit_trainvaltest.zip (11GB) [md5]***; Annotations ***gtFine_trainvaltest.zip (241MB) [md5]***.
* **CityScapes Foggy**: Download from the official [website](https://www.cityscapes-dataset.com/downloads/). Images ***leftImg8bit_trainval_foggyDBF.zip (20GB) [md5]***; Annotations are the same with `CityScapes`. Note, we chose foggy images with `beta=0.02` out of three kind of choices `(0.01, 0.02, 0.005)`.
* **Normal-style → Foggy-style** and **Foggy-style → Normal-style**: We taked target domain images as fake source images and vice versa. 
* **VOC foramt → YOLOv5 format**: Follow the script [convert_CitySpaces_yolo_label.py](./data/formats/convert_CitySpaces_yolo_label.py) and [convert_CitySpacesFoggy_yolo_label.py](./data/formats/convert_CitySpacesFoggy_yolo_label.py)


## Training and Testing

* **yamls**

We put the paths of the dataset involved in the training in the yaml file. Five kind of paths are need to be setted. Taking [pascalvoc0712_clipart1k_VOC.yaml](./data/yamls_sda/pascalvoc0712_clipart1k_VOC.yaml) as an example.

`path`: root path of datasets;

`train_source_real`: subpaths of real source images with labels for training. e.g., **PascalVOC(2007+2012)**;

`train_source_fake`: subpaths of fake source images with labels for training. e.g., **PascalVOC(2007+2012) Clipart-style**;

`train_target_real`: subpaths of real target images without labels for training. e.g., **Clipart1k**;

`train_tatget_fake`: subpaths of fake target images without labels for training. e.g., **Clipart1k VOC-style**;

`test_target_real`: subpaths of real target images with labels for testing;

`nc`: number of classes;

`names`: class names list.

* **training**

* **testing**


## References

* [YOLOv5 🚀 in PyTorch > ONNX > CoreML > TFLite](https://github.com/ultralytics/yolov5)
* [UMT(A Pytorch Implementation of Unbiased Mean Teacher for Cross-domain Object Detection (CVPR 2021))](https://github.com/kinredon/umt)


