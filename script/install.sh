#!/bin/bash

set -e

# Conda setup and environment creation
eval "$(conda shell.bash hook)"

conda create --name svgrender python=3.10 --yes
conda activate svgrender
export PYTHONNOUSERSITE=1
echo "The conda environment was successfully created"

# Install PyTorch, xformers and related libraries
pip install torch torchvision xformers --index-url https://download.pytorch.org/whl/cu128
echo "Pytorch installation is complete."

# Install common Python dependencies
pip install hydra-core omegaconf
pip install freetype-py shapely svgutils
pip install opencv-python scikit-image matplotlib visdom wandb BeautifulSoup4
pip install triton numba
pip install numpy scipy scikit-fmm einops timm fairscale==0.4.13
pip install safetensors
pip install "huggingface_hub<0.25.0" "transformers==4.35.2" "accelerate==0.25.0" "datasets==2.15.0"
pip install easydict scikit-learn pytorch_lightning==2.1.0 webdataset
echo "The basic dependency library is installed."

# Additional utility libraries
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
echo "CLIP installation is complete."

# Install diffusers
pip install diffusers==0.20.2
echo "Diffusers installation is complete. version: 0.20.2"

# Clone and set up DiffVG, handling dependencies on Ubuntu
git clone https://github.com/BachiLi/diffvg.git
cd diffvg
git submodule update --init --recursive

sed -i 's/cmake_minimum_required(VERSION 3.4)/cmake_minimum_required(VERSION 3.10)/g' pybind11/CMakeLists.txt
export DIFFVG_CUDA=1

# Install system dependencies for Ubuntu (to avoid potential issues)
echo "Installing system dependencies for DiffVG..."
sudo apt update
sudo apt install -y cmake ffmpeg build-essential libjpeg-dev libpng-dev libtiff-dev

conda install -y -c anaconda cmake
conda install -y -c conda-forge ffmpeg

# Install DiffVG
python setup.py install
echo "DiffVG installation is complete."

pip install svgwrite svgpathtools cssutils

# Final confirmation
echo "The running environment has been successfully installed!!!"
