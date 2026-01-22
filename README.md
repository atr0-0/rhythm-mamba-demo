# RhythmMamba: Personal Implementation

This repository is a personal implementation of the RhythmMamba model for remote photoplethysmography (rPPG) - a contactless method for measuring vital signs like heart rate and respiration rate from facial videos. RhythmMamba uses Mamba (Selective State Space Model) to efficiently process variable-length video sequences and extract physiological signals.

This implementation focuses on making RhythmMamba accessible and practical for personal projects by providing:
- **Complete WSL/Linux environment setup** for Windows users (required for Mamba dependencies)
- **Training and testing pipelines** with the lightweight UBFC-rPPG dataset
- **Inference API and frontend demo** for uploading videos and visualizing vital signs in real-time
- **Detailed documentation** covering environment setup, model training, and deployment

**Original Work:** Based on [RhythmMamba: Fast, Lightweight, and Accurate Remote Physiological Measurement [AAAI 2025]](https://github.com/zizheng-guo/RhythmMamba)

<img src="./figures/framework.jpg" alt="framework" style="zoom: 30%;" />

## 🎯 This Implementation Adds

- **WSL/Linux Setup Guide**: Complete instructions for installing Mamba dependencies (Triton, causal-conv1d, mamba-ssm) on WSL/Linux, as these are not available on Windows
- **UBFC-rPPG Dataset Support**: Ready-to-use configs for the lightweight UBFC-rPPG Dataset 1 (8 videos)
- **Inference API**: FastAPI endpoint for video upload and vital sign extraction
- **Frontend Demo**: Web interface to visualize rPPG waveforms and extracted heart rate / respiration rate



## :wrench: Setup

### Prerequisites
- **WSL2 with Ubuntu** (for Windows users) or native Linux environment
- **NVIDIA GPU** with CUDA support
- **CUDA Toolkit** installed (nvcc available)
- **Conda** or Miniconda

### Why WSL/Linux?
The Mamba dependencies (`mamba-ssm`, `causal-conv1d`) require:
- **Triton** (only available on Linux, not Windows)
- **C++ compiler** and CUDA toolkit for compilation

Windows users must use WSL2; native Linux users can proceed directly.

### Installation Steps (WSL/Linux)

**STEP 1: Install system dependencies**
```bash
sudo apt update
sudo apt install -y nvidia-cuda-toolkit build-essential ninja-build git curl wget
nvcc --version  # Verify CUDA toolkit installation
```

**STEP 2: Create conda environment**
```bash
conda create -n rhythm python=3.10 -y
conda activate rhythm
```

**STEP 3: Install PyTorch with CUDA**
```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```

**STEP 4: Install Mamba dependencies** (requires nvcc and build tools)
```bash
pip install causal-conv1d==1.4.0 --no-build-isolation
pip install mamba-ssm==2.2.2 --no-build-isolation
```

**STEP 5: Install remaining requirements**
```bash
pip install -r requirements.txt
```

### Notes
- Access your Windows files from WSL at `/mnt/e/...` (adjust drive letter as needed) 




## :computer: Example of Using Pre-trained Models 

Please use config files under `./configs/infer_configs`

For example, if you want to run the pre-trained model for intra-dataset on MMPD, use `python main.py --config_file ./configs/infer_configs/MMPD_RHYTHMMAMBA.yaml`

**Note:** Pre-trained model checkpoints are available from the original repository or can be trained from scratch. Checkpoints will be included with the frontend demo for quick testing and inference.

## 🌐 Frontend Demo

A web-based demo for uploading videos and extracting vital signs will be added.


## :computer: Examples of Neural Network Training

Please use config files under `./configs/train_configs`

### Training on UBFC-rPPG (Recommended Starting Point)

**UBFC-rPPG Dataset 1** is a lightweight dataset ideal for getting started with RhythmMamba training.

STEP 1: Download the [UBFC-rPPG Dataset 1](https://sites.google.com/view/ybenezeth/ubfcrppg)

STEP 2: Place data in a local directory

STEP 3: Edit `./configs/train_configs/intra/2UBFC-rPPG_RHYTHMMAMBA.yaml` and update the `DATA_PATH` to your data directory

STEP 4: Run training:
```bash
python main.py --config_file ./configs/train_configs/intra/2UBFC-rPPG_RHYTHMMAMBA.yaml
```

**Dataset Citation:**
If you use the UBFC-rPPG dataset, please cite:
```
S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, J. Dubois, 
"Unsupervised skin tissue segmentation for remote photoplethysmography", 
Pattern Recognition Letters, 2017.
```

### Intra-dataset on MMPD With RhythmMamba

STEP 1: Download the MMPD raw data by asking the paper authors

STEP 2: Modify `./configs/train_configs/intra/0MMPD_RHYTHMMAMBA.yaml` 

STEP 3: Run `python main.py --config_file ./configs/train_configs/intra/0MMPD_RHYTHMMAMBA.yaml` 

### Cross-dataset - Training on PURE and testing on UBFC-rPPG With RhythmMamba

STEP 1: Download the PURE raw data by asking the [paper authors](https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/pulse-rate-detection-dataset-pure).

STEP 2: Download the UBFC-rPPG raw data via [link](https://sites.google.com/view/ybenezeth/ubfcrppg)

STEP 3: Modify `./configs/train_configs/cross/PURE_UBFC-rPPG_RHYTHMMAMBA.yaml` 

STEP 4: Run `python main.py --config_file ./configs/train_configs/cross/PURE_UBFC-rPPG_RHYTHMMAMBA.yaml` 




## 📚 References & Research Papers

The following papers and resources helped in understanding RhythmMamba and SSM-based architectures:

- [RhythmMamba: Fast, Lightweight, and Accurate Remote Physiological Measurement](https://arxiv.org/html/2404.06483v1) [AAAI 2025]
- [Getting on the SSM Train: Exploring State Space Models for NLP](https://huggingface.co/blog/lbourdois/get-on-the-ssm-train#introduction) - HuggingFace Blog
- [Mamba Model Overview](https://www.ibm.com/think/topics/mamba-model) - IBM Think

## 🎓 Acknowledgement

This is a personal implementation focused on WSL/Linux setup and training pipeline for RhythmMamba. The core model architecture, configs, and evaluation metrics are from the original authors.

**Original Work:**
- RhythmMamba: [Zou et al., 2024](https://github.com/zizheng-guo/RhythmMamba)
- rPPG-Toolbox foundation: [Liu et al., 2023](https://github.com/ubicomplab/rPPG-Toolbox)
```
@article{liu2024rppg,
  title={rppg-toolbox: Deep remote ppg toolbox},
  author={Liu, Xin and Narayanswamy, Girish and Paruchuri, Akshay and Zhang, Xiaoyu and Tang, Jiankai and Zhang, Yuzhe and Sengupta, Roni and Patel, Shwetak and Wang, Yuntao and McDuff, Daniel},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```



## 📜 Citation

If you find this repository helpful, please consider citing:

```
@article{zou2024rhythmmamba,
  title={Rhythmmamba: Fast remote physiological measurement with arbitrary length videos},
  author={Zou, Bochao and Guo, Zizheng and Hu, Xiaocheng and Ma, Huimin},
  journal={arXiv preprint arXiv:2404.06483},
  year={2024}
}
```
