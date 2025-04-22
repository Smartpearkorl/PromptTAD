# PromptTAD: Object-Prompt Enhanced Traffic Anomaly Detection

We propose **PromptTAD**, a brand new architecture for both frame-level and instance-level video anomaly detection. The code is developed based on the Pytorch framework.

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/memory-augmented-online-video-anomaly/online-video-anomaly-detection-on-detection)](https://paperswithcode.com/sota/online-video-anomaly-detection-on-detection?p=memory-augmented-online-video-anomaly)

<!-- ![Video Anomaly Detection with PromptTAD](images/demo.gif) -->


<video src="/images/demo.mp4" width="320" height="240" controls></video>  

## Introduction
![PromptTAD Architurecture](images/overview.png)
Ego-centric Traffic Anomaly Detection (TAD) aims to identify abnormal events in videos captured by dashboard-mounted cameras in vehicles. Compared to anomaly detection in roadside surveillance videos, ego-centric TAD poses greater challenges due to the dynamic backgrounds caused by vehicle motion. Previous frame-level methods are often vulnerable to interference from these dynamic backgrounds and struggle to detect small objects located at a distance or off-center. To address these challenges, we propose an object-prompt enhanced method that integrates detected traffic objects into a frame-level TAD framework. Our approach introduces an object-prompt scheme comprising an object prompt encoder, along with two cross-attention-based aggregation modules: an instance-wise aggregation module for fusing information between object instances and the scene, and a relation-wise aggregation module for capturing relationships inter-objects. Additionally, we design an instance-level loss to supervise anomaly detection at the object level. Our method effectively mitigates interference from dynamic backgrounds, improves the detection of distant or off-center anomalies, and enables precise spatial localization of anomalies. Experimental results on the DoTA and DADA-2000 datasets demonstrate that our method achieves state-of-the-art performance.


## Requirements
### 1. conda env preparation
```bash
git clone https://github.com/Smartpearkorl/PromptTAD.git
cd PromptTAD
conda env create -n poma --file environment.yml
conda activate poma
```
### 2. pathes preparation
TPromptTAD/runner/__init__.py determine the shared parameters, such as the path of the dataset, the path of the font, etc.
```python
'''
1. define some shared parameters
'''
from pathlib import Path
DATA_FOLDER = Path('/data/qh/DoTA/data')
FONT_FOLDER = "/data/qh/dependency/arial.ttf"
CHFONT_FOLDER = "/data/qh/dependency/microsoft_yahei.ttf"

# train / test folder
pretrained_weight_path = Path('/data/qh/pretrain_models/')
vst_pretrained_weight_path = pretrained_weight_path / 'swin_base_patch244_window1677_sthv2.pth'                        

DoTA_FOLDER = Path('/ssd/qh/DoTA/data') 
DADA_FOLDER = Path('/ssd/qh/DADA2000/')
META_FOLDER = DATA_FOLDER / 'metadata'

DEBUG_FOLDER = Path('/data/qh/DoTA/Intervl-TAD/debug')
```

### 2. pretrained weight preparation
```bash
# swin-transformer pretrained weights 
$ wget https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window1677_sthv2.pth -O pretrained/swin_base_patch244_window1677_sthv2.pth
# yolov9 pretrained weights
```

## Data and Bounding Boxes preparation
### 1. Download DoTa dataset

Please download from [official website](https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly)
the dataset and save inside `data/dota` directory.

You should obtain the following structure:

```
data/dota
├── annotations
│   ├── 0qfbmt4G8Rw_000306.json
│   ├── 0qfbmt4G8Rw_000435.json
│   ├── 0qfbmt4G8Rw_000602.json
│   ...
├── frames
│   ├── 0qfbmt4G8Rw_000072
│   ├── 0qfbmt4G8Rw_000306
│   ├── 0qfbmt4G8Rw_000435
│   .... 
└── metadata
    ├── metadata_train.json
    ├── metadata_val.json
    ├── train_split.txt
    └── val_split.txt
```


## Usage

###  Installation
```bash
$ git clone https://github.com/IMPLabUniPr/movad/tree/movad_vad
$ cd movad
$ wget https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window1677_sthv2.pth -O pretrained/swin_base_patch244_window1677_sthv2.pth
$ conda env create -n movad_env --file environment.yml
$ conda activate movad_env
```



# Download pretrained on DoTA dataset

Open [Release v1.0](https://github.com/IMPLabUniPr/movad/releases/tag/v1.0)
page and download .pt (pretrained) and .pkl (results) file.
Unzip them inside the `output` directory, obtaining the following directories
structure:

```
output/
├── v4_1
│   ├── checkpoints
│   │   └── model-640.pt
│   └── eval
│       └── results-640.pkl
└── v4_2
    ├── checkpoints
    │   └── model-690.pt
    └── eval
        └── results-690.pkl
```

### Train
```bash
python src/main.py --config cfgs/v4_2.yml --output output/v4_2/ --phase train --epochs 1000 --epoch -1
```

### Eval
```bash
python src/main.py --config cfgs/v4_2.yml --output output/v4_2/ --phase test --epoch 690
```

### Play: generate video
```bash
python src/main.py --config cfgs/v4_2.yml --output output/v4_2/ --phase play --epoch 690
```

## Results

### Table 1

Memory modules effectiveness.

| # | Short-term | Long-term | AUC | Conf |
|:---:|:---:|:---:|:---:|:---:|
| 1 |   |   | 66.53 | [conf](cfgs/v0_1.yml) |
| 2 | X |   | 74.46 | [conf](cfgs/v2_3.yml) |
| 3 |   | X | 68.76 | [conf](cfgs/v1_1.yml) |
| 4 | X | X | 79.21 | [conf](cfgs/v1_3.yml) |

### Figure 2

Short-term memory module.

| Name | Conf |
|:---:|:---:|
| NF 1 | [conf](cfgs/v1_1.yml) |
| NF 2 | [conf](cfgs/v1_2.yml) |
| NF 3 | [conf](cfgs/v1_3.yml) |
| NF 4 | [conf](cfgs/v1_4.yml) |
| NF 5 | [conf](cfgs/v1_5.yml) |

### Figure 3

Long-term memory module.

| Name | Conf |
|:---:|:---:|
| w/out LSTM     | [conf](cfgs/v2_1.yml) |
| LSTM (1 cell)  | [conf](cfgs/v2_2.yml) |
| LSTM (2 cells) | [conf](cfgs/v1_3.yml) |
| LSTM (3 cells) | [conf](cfgs/v2_3.yml) |
| LSTM (4 cells) | [conf](cfgs/v2_4.yml) |

### Figure 4

Video clip length (VCL).

| Name | Conf |
|:---:|:---:|
| 4 frames  | [conf](cfgs/v3_1.yml) |
| 8 frames  | [conf](cfgs/v1_3.yml) |
| 12 frames | [conf](cfgs/v3_2.yml) |
| 16 frames | [conf](cfgs/v3_3.yml) |

### Table 2

Comparison with the state of the art.

| # | Method | Input | AUC | Conf |
|:---:|:---:|:---:|:---:|:---:|
| 9  | Our (MOVAD) | RGB (320x240) | 80.09 | [conf](cfgs/v4_1.yml) |
| 10 | Our (MOVAD) | RGB (640x480) | 82.17 | [conf](cfgs/v4_2.yml) |

## License

See [GPL v2](./LICENSE) License.

## Acknowledgement

This research benefits from the HPC (High Performance Computing) facility
of the University of Parma, Italy.

## Citation
If you find our work useful in your research, please cite:

```
@misc{https://doi.org/10.48550/arxiv.2302.10719,
  doi = {10.48550/ARXIV.2302.10719},
  url = {https://arxiv.org/abs/2302.10719},
  author = {Rossi, Leonardo and Bernuzzi, Vittorio and Fontanini, Tomaso and Bertozzi, Massimo and Prati, Andrea},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences, F.1.1, 68-02, 68-04, 68-06, 68T07, 68T10, 68T45},
  title = {Memory-augmented Online Video Anomaly Detection},
  publisher = {arXiv},
  year = {2023},
  copyright = {Creative Commons Attribution Share Alike 4.0 International}
}
```
=======
# Prompt_TAD
Prompt for Traffic  Accident Detection
>>>>>>> b68512f20e00033c28d8c30d8a0962729e89bbc5