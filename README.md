## Action Selection Learning for Multi-label Multi-view Action Recognition
[[paper](Add Link Here)]

**Authors:** [Trung Thanh Nguyen](https://scholar.google.com/citations?user=QSV452QAAAAJ), [Yasutomo Kawanishi](https://scholar.google.com/citations?user=Tdfw6WMAAAAJ), [Takahiro Komamizu](https://scholar.google.com/citations?user=j4n_V44AAAAJ), [Ichiro Ide](https://scholar.google.com/citations?user=8PXJm98AAAAJ)


## Introduction
This repository contains the implementation of MultiASL (Multi-view Action Selection Learning) on the MM-Office dataset.


## Environment

The Python code is developed and tested in the environment specified in `environment.yml`. 
Experiments on the MM-Office dataset were conducted on a single NVIDIA RTX A6000 GPU with 48 GB of GPU memory. 
You can adjust the `batch_size` to accommodate GPUs with smaller memory.


## Dataset

Download the MM-Office dataset [here](https://github.com/nttrd-mdlab/mm-office) and place it in the `dataset/MM-Office` directory.

## Training
To train the model, execute the following command:
```
    bash ./scripts/train_MM_ViT_Transformer.sh
```

## Inference
To perform inference, use the following command:
```
    bash ./scripts/infer_MM_ViT_Transformer.sh
```

## Citation

If you find this code useful for your research, please cite the following paper:
```
@inproceedings{nguyen2024MultiASL,
    title={Action Selection Learning for Multi-label Multi-view Action Recognition},
    author={Trung Thanh Nguyen, Yasutomo Kawanishi, Takahiro Komamizu, Ichiro Ide},
    year={2024}
}
```

This source code is based on the following paper:
```
@inproceedings{ma2021asl,
  title={Weakly Supervised Action Selection Learning in Video},
  author={Ma, Junwei and Gorti, Satya Krishna and Volkovs, Maksims and Yu, Guangwei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```