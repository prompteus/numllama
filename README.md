# Remarkably Accurate

This is the official repository for the EMNLP 2025 paper: **Pre-trained Language Models Learn Remarkably Accurate Representations of Numbers.**

It introduces a sinusoidal probe - a probe suitable for decoding representations of numbers. This probe assumes a sinusoidal pattern in embeddings, which is learned by many models across diferent model families (Llama 3, Olmo 2, Phi 4) during pretraining.


## Get started

To get started playing around with sinusoidal probe, we recommend trying out going through a notebook `noteoboks/model_activations_probing_next_tok.ipynb`, which is self-contained (depends only on torch and transformers). The notebook showcases how you can start probing the internal activations of a model.


## Set up env

```shell
git clone ...
cd ...
conda create -n numllama python=3.12
conda activate numllama
pip install poetry
poetry install
```


## Citation
```bibtex
@misc{kadlcik2025remarkablyaccurate,
      title={Pre-trained Language Models Learn Remarkably Accurate Representations of Numbers}, 
      author={Marek Kadl\v{c}\'{i}k and Michal \v{S}tef\'{a}nik and Timothee Mickus and Michal Spiegel and Josef Kucha\v{r}},
      year={2025},
      eprint={2506.08966},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.08966}, 
}
```
