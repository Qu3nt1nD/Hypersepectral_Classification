# Hyperspectral Imagery Classification

## Get started

### Installation

First download the repository.
Then install the requirements from the designated file.

```
pip install requirements.txt
```


## Demonstration

### Training

```
python main.py --data ./data/pavia_university/data.yaml --nn_mode 2D --train
```

### Evaluation

```
python main.py --data ./data/pavia_university/data.yaml --nn_mode 2D --resume path_to_model --full_map
```
