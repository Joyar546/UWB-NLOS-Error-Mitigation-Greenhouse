# UWB NLOS Error Mitigation in Greenhouse Environments

[![Python 3.6](https://img.shields.io/badge/Python-3.6-blue.svg)](https://www.python.org/downloads/release/python-3613/)
[![PyTorch 1.10](https://img.shields.io/badge/PyTorch-1.10%2Bcu113-orange.svg)](https://pytorch.org/get-started/previous-versions/#v110)

This repository implements the ECA-ResNet model for mitigating Ultra-Wideband (UWB) ranging errors in greenhouse NLOS conditions, as proposed in our [paper](https://doi.org/10.1016/j.compag.2022.107573). The code is compatible with **PyTorch 1.10 + CUDA 11.3** and **Python 3.6**.

## 🛠️ Environment Setup
```bash
# Create a conda environment (recommended)
conda create -n uwb_nlos python=3.6.13
conda activate uwb_nlos

# Install PyTorch with CUDA 11.3
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Install other dependencies
pip install numpy pandas matplotlib scikit-learn tqdm
```

## 📊 Dataset
```bash
CIR : 1016-length vector (raw), representing Channel Impulse Response.

Edis: Estimated distance (in meters) calculated by UWB module (DWM1000).

Rdis: Ground-truth distance (in meters) measured by laser rangefinder.

Obstacle: Integer label indicating NLOS condition:

| Label | Obstacle Type    | Description                     |
|-------|------------------|---------------------------------|
| 0     | LOS              | Line-of-Sight (no obstacle)     |
| 1     | Human            | Blocked by human body           |
| 2     | Wood             | Wooden barriers                 |
| 3     | Steel            | Metal structures                |
| 4     | Wall             | Concrete/brick walls            |
| 5     | Glass            | Glass panels/doors              |
| 6     | Leaves           | Plant foliage                   |
| 7     | Leaves + Glass   | Mixed obstruction               |
```


## 📜 Citation
```bash
@article{niu2023deep,
  title={Deep learning-based ranging error mitigation method for UWB localization system in greenhouse},
  author={Niu, Ziang and Yang, Huizhen and Zhou, Lei and Taha, Mohamed Farag and He, Yong and Qiu, Zhengjun},
  journal={Computers and Electronics in Agriculture},
  volume={205},
  pages={107573},
  year={2023},
  publisher={Elsevier}
}
```