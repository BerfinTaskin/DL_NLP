#!/bin/bash -i
set -e

# Set up Conda, install Python
conda create -n dnlp python=3.10 -y
conda activate dnlp

# Check for CUDA and install appropriate PyTorch version
if command -v nvidia-smi &>/dev/null; then
    echo "CUDA detected, installing PyTorch with CUDA support."
    conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
else
    echo "CUDA not detected, installing CPU-only PyTorch."
    conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 cpuonly -c pytorch -y
fi

# Install additional packages
conda install tqdm==4.66.2 requests==2.31.0 transformers==4.38.2 tensorboard==2.16.2 tokenizers==0.15.1 -c conda-forge -c huggingface -y
pip install explainaboard-client==0.1.4 sacrebleu==2.4.0 numpy==1.26.4
pip install peft==0.11.1
pip install nltk==3.9.1
pip install inflect==7.5.0
pip install wordfreq==3.1.1