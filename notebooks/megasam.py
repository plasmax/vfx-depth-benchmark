"""
Testing out MegaSaM using a Runpod.io 48GB VRAM instance
https://github.com/mega-sam/mega-sam

First, I had to install miniconda

```bash
cd /workspace
wget -q --show-progress https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /workspace/miniconda3
git clone --recursive https://github.com/mega-sam/mega-sam.git
cd ./mega-sam/
/workspace/miniconda3/bin/conda env create -f environment.yml
wget https://anaconda.org/xformers/xformers/0.0.22.post7/download/linux-64/xformers-0.0.22.post7-py310_cu11.8.0_pyt2.0.1.tar.bz2
/workspace/miniconda3/bin/conda install xformers-0.0.22.post7-py310_cu11.8.0_pyt2.0.1.tar.bz2

```

"""