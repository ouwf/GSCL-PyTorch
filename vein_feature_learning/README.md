## Vein feature learning

PyTorch implementation of the feature learning scheme in [GSCL: Generative Self-supervised Contrastive Learning for Vein-based Biometric Verification](https://ieeexplore.ieee.org/abstract/document/10428026).

## Data preparation
1. Download the [synthetic finger vein sample set](https://portland-my.sharepoint.com/:u:/g/personal/weifengou2-c_my_cityu_edu_hk/EdltFgKYephGonGXunb7zX0B1moW2dg283bZ3GLfrYpfGw?e=OP3Xxe) for model pretraining.
2. Download the [preprocessed FV-USM dataset](https://portland-my.sharepoint.com/:u:/g/personal/weifengou2-c_my_cityu_edu_hk/EZR-zf6MCxJOikdLh5Eb7X0BeiJEiIZ6cFLRWgCFdEWf-Q?e=uOV5aE) for model finetuning.

## Training
### Pretrain resnet-18 on synthetic fv samples with SimCLR
```bash
$ bash run_pretrain_simclr.sh "path_to_synthetic_trainset" "path_to_testset"
```

### Pretrain resnet-18 on synthetic fv samples with BYOL
```bash
$ bash run_pretrain_byol.sh "path_to_synthetic_trainset" "path_to_testset"
```

### Finetune resnet-18 on FVUSM trainset with FusionAug 
```bash
$ bash run_finetune_fusionaug.sh "path_to_trainset" "path_to_testset" "path_to_pretrained_ckpt"
```

### Train resnet-18 on FVUSM trainset with FusionAug w/o ssl pretraining (supervised baseline) 
```bash
$ bash run_finetune_fusionaug.sh "path_to_trainset" "path_to_testset"
```

## Testing
```bash
$ python3 -u ./test.py --ckpt "path_to_checkpoint" --data "path_to_testset" --dataset_name "name of dataset, default: FVUSM" --network "name of network, default: resnet18"
```

## Results

We follow the open-set evaluation protocol and divide the FV-USM database by a training set (2952 images) and a testing set (2952 images) which consists of non-overlapped vein classes. The biometric verification performance of different approaches evaluated on the testing set are summarized below. 

|         Method         |      Network      | EER(%) | FRR(%)@FAR=0.01 | FRR(%)@FAR=0.001 | FRR(%)@FAR=0.0001 |
|:----------------------:|:-----------------:|:------:|:---------------:|:----------------:|:-----------------:|
|      FusionAug         |   [Resnet-18](https://portland-my.sharepoint.com/:u:/g/personal/weifengou2-c_my_cityu_edu_hk/EaVtgan6kf5Lp9c1hU03cDgBkgZhqYZDctpWEalpIgzMSw?e=kuLdoa)   |  0.33  |      0.16       |       0.99       |       4.51        |
|         SimCLR         |   [Resnet-18](https://portland-my.sharepoint.com/:u:/g/personal/weifengou2-c_my_cityu_edu_hk/EV7EYqd6Rr5Am5pLAFRB4kUBxe3QxsKkuhj9Ax-JlJC8Og?e=olKWV3)   |  1.03  |      1.03       |       5.32       |       11.39       |
| GSCL(SimCLR+FusionAug) |   [Resnet-18](https://portland-my.sharepoint.com/:u:/g/personal/weifengou2-c_my_cityu_edu_hk/ERUnbGwTn9JJh4f4QfT25lUB0YxiS0wdcs6KDQX5ZLgMXw?e=aKu6C7)   |  0.18  |      0.02       |       0.45       |       2.47        |
|          BYOL          |   [Resnet-18](https://portland-my.sharepoint.com/:u:/g/personal/weifengou2-c_my_cityu_edu_hk/Ef9vc9LNJi1OrzLibjoXnhQBvm_eygATNiJaXC6og9WyFg?e=HJJpIW)   |  4.00  |      8.72       |      22.13       |       37.82       |
|  GSCL(BYOL+FusionAug)  |   [Resnet-18](https://portland-my.sharepoint.com/:u:/g/personal/weifengou2-c_my_cityu_edu_hk/EaS0wCDucvVLk07Tguts2_IBJDh1KFIR59VXLx_p6MemJg?e=FXbwvB)   |  0.51  |      0.27       |       2.16       |       8.11        |


## Acknowledgement
* The copyright of the [FV-USM database](http://drfendi.com/fv_usm_database/) is owned by Dr. Bakhtiar Affendi Rosdi, School of Electrical and Electronic Engineering, USM.
* This code is inspired by and built upon several public projects, many thanks to the authors.
  * https://github.com/adambielski/siamese-triplet/tree/master
  * https://github.com/sthalles/PyTorch-BYOL/
  * https://github.com/sthalles/SimCLR


## Citation
```bibtex
@article{ou2024gscl,
  title={GSCL: Generative Self-Supervised Contrastive Learning for Vein-Based Biometric Verification},
  author={Ou, Wei-Feng and Po, Lai-Man and Huang, Xiu-Feng and Yu, Wing-Yin and Zhao, Yu-Zhi},
  journal={IEEE Transactions on Biometrics, Behavior, and Identity Science},
  year={2024},
  publisher={IEEE}
}

@article{ou2021fusion,
  title={Fusion loss and inter-class data augmentation for deep finger vein feature learning},
  author={Ou, Wei-Feng and Po, Lai-Man and Zhou, Chang and Rehman, Yasar Abbas Ur and Xian, Peng-Fei and Zhang, Yu-Jia},
  journal={Expert Systems with Applications},
  volume={171},
  pages={114584},
  year={2021},
  publisher={Elsevier}
}

@article{asaari2014fusion,
  title={Fusion of band limited phase only correlation and width centroid contour distance for finger based biometrics},
  author={Asaari, Mohd Shahrimie Mohd and Suandi, Shahrel A and Rosdi, Bakhtiar Affendi},
  journal={Expert Systems with Applications},
  volume={41},
  number={7},
  pages={3367--3382},
  year={2014},
  publisher={Elsevier}
}

@inproceedings{chen2020simple,
  title={A simple framework for contrastive learning of visual representations},
  author={Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
  booktitle={International conference on machine learning},
  pages={1597--1607},
  year={2020},
  organization={PMLR}
}

@article{grill2020bootstrap,
  title={Bootstrap your own latent-a new approach to self-supervised learning},
  author={Grill, Jean-Bastien and Strub, Florian and Altch{\'e}, Florent and Tallec, Corentin and Richemond, Pierre and Buchatskaya, Elena and Doersch, Carl and Avila Pires, Bernardo and Guo, Zhaohan and Gheshlaghi Azar, Mohammad and others},
  journal={Advances in neural information processing systems},
  volume={33},
  pages={21271--21284},
  year={2020}
}

@inproceedings{schroff2015facenet,
  title={Facenet: A unified embedding for face recognition and clustering},
  author={Schroff, Florian and Kalenichenko, Dmitry and Philbin, James},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={815--823},
  year={2015}
}

@inproceedings{wang2018cosface,
  title={Cosface: Large margin cosine loss for deep face recognition},
  author={Wang, Hao and Wang, Yitong and Zhou, Zheng and Ji, Xing and Gong, Dihong and Zhou, Jingchao and Li, Zhifeng and Liu, Wei},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={5265--5274},
  year={2018}
}

```
