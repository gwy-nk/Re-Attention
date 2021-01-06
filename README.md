# Pytorch Implementation of Re-Attention for Visual Question Answering
## Introduction

We propose a re-attention framework to utilize the information in answers for the VQA task. 
The framework first learns the initial attention weights for the objects by calculating the similarity 
of each word-object pair in the feature space. Then, the visual attention map is reconstructed by re-attending 
the objects in images based on the answer. Through keeping the initial visual attention map and the reconstructed 
one to be consistent, the learned visual attention map can be corrected by the answer information.

## Prerequisites

* Python 3.5
* Pytorch ≥ 1.0
* CUDA 10.0

## Installation
Thanks for the works of [Vision and Language Group@ MIL](http://mil.hdu.edu.cn/). 
Our code is based on the implementation of [MCAN](https://github.com/MILVLG/mcan-vqa). 
We use the Faster R-CNN model pre-trained on the Visual Genome dataset to represent the image as a set of object-level features. 
For simplicity, you can download the pre-processed data from [here](https://github.com/MILVLG/mcan-vqa). 
Please refer to [here](https://github.com/MILVLG/mcan-vqa) to install the required environment.

## Training

Train re-attention with ground-truth annotation:

```bash
python run.py --RUN=train --VERSION=${CKPT_NAME} --GPU=${GPU_ID} --SPLIT=${SAMPLE_SPLIT} --MAX_EPOCH=${MAX_EPOCH} --recon_rate=${RECON_RATE} --entropy_tho=${RECON_THRESHOLD} --AGCAN_MODE=${METHOD_TYPE} --DATASET=${DATASET}
```

To add：

1. ```--AGCAN_MODE={recon, recon_e}```, ```--AGCAN_MODE=recon``` is for the method displayed in our AAAI 2020 paper and ```--AGCAN_MODE=recon_e``` is the enhanced re-attention version in our further research.

2. ```--recon_rate=float``` and ```--entropy_tho=float``` are the threshold for the re-attention and the added gate mechanism, respectively.


## Evaluation

Evaluate re-attention with ground-truth annotation:

```bash
python run.py --RUN=test --CKPT_PATH=${CHECKPOINT_PATH}  --AGCAN_MODE=${METHOD_TYPE} --GPU=${GPU_ID}  --DATASET=${DATASET}
```


## Citation

    @inproceedings{GuoZWYCY20,
	author    = {Wenya Guo,	Ying Zhang ,Xiaoping Wu, Jufeng Yang, Xiangrui Cai, Xiaojie Yuan},
	title     = {Re-Attention for Visual Question Answering},
	booktitle = {The AAAI conference on artificial intelligence (AAAI)},
	pages     = {91--98},
	year      = {2020},
    }
