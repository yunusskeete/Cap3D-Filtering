# SF-Cap-3d-Filtering

## Overview
[Cap3D](https://cap3d-um.github.io/) provides detailed descriptions of 3D objects by leveraging pretrained models in captioning, alignment, and LLM to consolidate multi-view information.
These descriptions detail assets from a large-scale and broad-categoried collection of open-source 3D models (namely [Objaverse](https://arxiv.org/abs/2212.08051), [Objaverse-XL](https://arxiv.org/abs/2307.05663), and [ABO](https://arxiv.org/abs/2110.06199)).
Examples of the descriptions, along with their rendered images, are shown below:

![Example captioning results by Cap3D.](https://tiangeluo.github.io/projectpages/imgs/Cap3D/teaser.png)

This repo contains scripts for the filtering of the [Cap3D dataset](https://cap3d-um.github.io/) for domain alignment with an architectural/building/interior design use case, with basic natural language processing techniques.
CSV captions are downloaded, filtered, and the UID of the domain-alighned 3D assets are saved in JSON files. The following [Jupyter notebook](filter_dataset.ipynb) provides a step-by-step and visually-aided walk through the [Cap3D dataset](https://huggingface.co/datasets/tiange/Cap3D) filtering logic.

## Functionality
### [download_captions.py](download_captions.py)
This script provides functionality for the downloading of the latest set of captions (the dataset is continously expanded).
Example:
```bash
cd captions
python download_captions.py
cd ..
```
### [filter_dataset.py](filter_dataset.py)
This script provides functionality for the downloading of the latest set of captions (the dataset is continously expanded), and the filtering of 3D assets based on these captions using simple natural language processing techniques.
Example:
```bash
# Run download captions script to ensure latest captions are downloaded
cd captions
python download_captions.py
cd ..
# Run dataset filtering script
python filter_dataset.py --config_path filter_dataset.yaml
```

NOTE: A [configuration file](environment.yml) must be supplied, detailing paths for loading and saving files, as well as other runtime constants including filter categories.

## Setup
Create the conda environment:
```bash
conda env create
```

Activate the environment:
```bash
conda activate cap3d_filter
```
