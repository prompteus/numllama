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
@inproceedings{kadlcik-etal-2025-pre,
    title = "Pre-trained Language Models Learn Remarkably Accurate Representations of Numbers",
    author = "Kadl{\v{c}}{\'i}k, Marek  and
      {\v{S}}tef{\'a}nik, Michal  and
      Mickus, Timothee  and
      Kucha{\v{r}}, Josef  and
      Spiegel, Michal",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.1356/",
    doi = "10.18653/v1/2025.emnlp-main.1356",
    pages = "26705--26714",
    ISBN = "979-8-89176-332-6"
}
```

