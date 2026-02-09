# GraphLand

This is the official repository for the paper [GraphLand: Evaluating Graph Machine Learning Models on Diverse Industrial Data](https://arxiv.org/abs/2409.14500) (accepted at NeurIPS 2025 Datasets & Benchmarks track).

> GraphLand datasets are available at [Zenodo](https://zenodo.org/records/16895532) and [Kaggle](https://kaggle.com/datasets/bazhenovgleb/graphland).

## How to use this repository

### If you want to use the GraphLand datasets

GraphLand datasets come from real-world industrial applications of Graph Machine Learning and are thus a bit more complex than most popular graph datasets. 
They come with node features of different types and have several realistic data splits including one for the inductive setting.
Thus, the raw data may require some preprocessing before it is ready to be passed into a neural model.
Our implementation of this preprocessing is provided in the `dataset.py` file. This file is self-contained.
It provides two classes: `Dataset` and `PyGDataset`.
The `Dataset` class is what this repository uses internally, however, most people are more familiar with datasets in the style of PyTorch Geometric (PyG), so what you probably need is the `PyGDataset` class.
This class provides GraphLand datasets in a format similar to PyG datasets.

How to use `PyGDataset`:

- Install the requirements from `environment_pyg.yaml` (this is a lightweight environment that only contains the libraries neccessary for `PyGDataset`, but not those neccessary for using the rest of the code in this repository).

- Download (some of) the GraphLand datasets from [Zenodo](https://zenodo.org/records/16895532) or [Kaggle](https://kaggle.com/datasets/bazhenovgleb/graphland) and put them in the `data` directory.

- Now you can use `PyGDataset`. A detailed description of its arguments is provided in the class docstring, but you most likely only need to specify the dataset name and the data split to use. Below we provide a couple simple examples. You can read more about the available data splits and the transductive and the inductive learning settings in our paper.

If you want to use a dataset in the transductive setting (that is, with either `RL`, `RH`, or `TH` split), your code might look like this:

```python
from dataset import PyGDataset

dataset = PyGDataset(name='artnet-views', split='RL')

# In the transductive setting, the dataset contains a single PyG Data object.
data = dataset[0]

best_val_metric = 0
corresponding_test_metric = 0
for _ in range(num_steps):
    model.train()
    preds = model(features=data.x, edges=data.edge_index)
    loss = compute_loss(input=preds[data.train_mask], target=data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    model.eval()
    with torch.no_grad():
        preds = model(features=data.x, edges=data.edge_index)
        val_metric = compute_metric(input=preds[data.val_mask], target=data.y[data.val_mask])
        if val_metric > best_val_metric:
            test_metric = compute_metric(input=preds[data.test_mask], target=data.y[data.test_mask])
            corresponding_test_metric = test_metric

print(f'Best val metric: {val_metric}, corresponding test metric: {corresponding_test_metric}')
```

If you want to use a dataset in the inductive setting (that is, with `THI` split), your code might look like this:

```python
from dataset import PyGDataset

dataset = PyGDataset(name='artnet-views', split='THI')

# In the inductive setting, the dataset contains 3 PyG Data objects - the train, val, and test snapshots of an evolving network.
train_data, val_data, test_data = dataset

best_val_metric = 0
corresponding_test_metric = 0
for _ in range(num_steps):
    model.train()
    preds = model(features=train_data.x, edges=train_data.edge_index)
    loss = compute_loss(input=preds[train_data.train_mask], target=train_data.y[train_data.train_mask])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    model.eval()
    with torch.no_grad():
        preds = model(features=val_data.x, edges=val_data.edge_index)
        val_metric = compute_metric(input=preds[val_data.val_mask], target=val_data.y[val_data.val_mask])
        if val_metric > best_val_metric:
            preds = model(features=test_data.x, edges=test_data.edge_index)
            test_metric = compute_metric(input=preds[test_data.test_mask], target=test_data.y[test_data.test_mask])
            corresponding_test_metric = test_metric

print(f'Best val metric: {val_metric}, corresponding test metric: {corresponding_test_metric}')
```

### If you want to reproduce our experimental results

This repository can be used to run experiments with GNNs on many graph datasets, not only those from the GraphLand benchmark.

How to run experiments in this repository:

- Install the requirements from `environment_dgl.yaml`. This environment includes DGL, which is a very efficient library for GNNs, but which is unfortunately no longer maintained and can only work with a relatively old version of PyTorch.

- If you want to run expriments with (some of) the GraphLand datasets, download them from [Zenodo](https://zenodo.org/records/16895532) or [Kaggle](https://kaggle.com/datasets/bazhenovgleb/graphland) and put them in the `data` directory. If you want to run experiments with other supported datasets, they will be downloaded automatically upon experiment launch.

- Run `main.py` with the neccessary arguments.

Executing `main.py` runs a single experiment, which might include hyperparameter search and multiple runs with the best hyperparameters to compute the mean and standard deviation of model performance. `main.py` can accept a number of arguments, see `get_args` function from `args.py` for a full list. A simple example is:

```bash
python main.py --name example_experiment --dataset hm-categories --split RL --model GT --lr 3e-4 --dropout 0.1 --device cuda:0
```

Many arguments can accept a list of values. Passing a list of values rather than a single value will trigger grid search over these values. For example, to run grid search over learning rate and dropout probability, run the following command:

```bash
python main.py --name example_experiment_with_hparam_search --dataset hm-categories --split RL --model GT --lr 3e-5 1e-4 3e-4 1e-3 3e-3 --dropout 0 0.1 0.2 --device cuda:0
```

The directory `scripts` contains commands to reproduce all the GNN experiments in our paper. Specifically, there are 4 file, one for each split and setting in our paper: `run_experiments_RL.sh`, `run_experiments_RH.sh`, `run_experiments_TH.sh`, `run_experiments_THI.sh`. Each file contains commands to reproduce all GNN experiments for the corresponding split and setting (`RL`, `RH`, `TH`, `THI`). Each line runs a single experiment (with a single model and a single dataset) including hyperparameter search.

Note that running all the experiments takes considerable time (multiple days on a single A100 GPU), although experimenting with smaller datasets is much faster than with larger ones.

The code for reproducing the results of non-GNN models from our paper, i.e., GBDTs and GFMs, can be found in [another repository](https://github.com/gvbazhenov/graphland-baselines).

## Citation

If you found GraphLand datasets or this repository useful, please cite the following work:

```
@inproceedings{bazhenov2025graphland,
  title={GraphLand: Evaluating graph machine learning models on diverse industrial data},
  author={Bazhenov, Gleb and Platonov, Oleg and Prokhorenkova, Liudmila},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2025}
}
```
