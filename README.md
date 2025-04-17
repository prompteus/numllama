# NumLlama

Project for exploring numerical embeddings for language models.


## Set up env

```shell
git clone ...
cd ...
conda create -n numllama python=3.12
conda activate numllama
pip install poetry
poetry install
```


## Train model

```shell
mkdir .wandb
mkdir checkpoints
CUDA_VISIBLE_DEVICES=... python scripts/train_calc.py
```
