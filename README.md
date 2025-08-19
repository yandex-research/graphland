# GraphLand

This is the official repository for the paper "[GraphLand: Evaluating Graph Machine Learning Models on Diverse Industrial Data](https://arxiv.org/abs/2409.14500)".

GraphLand datasets are available at [Zenodo](https://zenodo.org/records/16895532).

This repository can be used to run experiments with GNNs on many graph datasets including those from the GraphLand benchmark.

How to use it:

If you want to use datasets from the GraphLand benchmark, download them from [Zenodo](https://zenodo.org/records/16895532) and put them in the `data` directory. If you want to use other supported datasets, they can be downloaded by the `Dataset` class automatically.

If you simply want to get access to GraphLand and other datasets in a unififed format, you can use the `Dataset` class from the `dataset.py` file — it is self-contained.

If you want to run experiments in this codebase, install the dependencies from `environment.yml` and run `main.py`.

Executing `main.py` runs a single experiment which might include hyperparameter search and multiple runs with the best hyperparameters to compute the mean and standard deviation of model performance. `main.py` can accept a number of arguments, see `get_args` function from `args.py` for a full list. A simple example is:

```
python main.py --name example_experiment --dataset hm-categories --split RL --transductive True --model GT --lr 3e-4 --dropout 0.1 --device cuda:0
```

Many arguments can accept a list of values. Passing a list of values rather than a single value will trigger grid search over these values. For example, to run grid search over learning rate and dropout probability, run the following command:

```
python main.py --name example_experiment_with_hparam_search --dataset hm-categories --split RL --transductive True --model GT --lr 3e-5 1e-4 3e-4 1e-3 3e-3 --dropout 0 0.1 0.2 --device cuda:0
```

The directory `scripts` contains commands to reproduce all the GNN experiments in our paper. Specifically, there are 4 file, one for each setting in our paper: `run_experiments_RL.sh`, `run_experiments_RH.sh`, `run_experiments_TH.sh`, `run_experiments_THI.sh`. Each file contains commands to reproduce all GNN experiments for the corresponding setting (`RL`, `RH`, `TH`, `THI`). Each line runs a single experiment (with a single model and a single dataset) including hyperparameter search. Note that running all the experiments takes considerable time (multiple days on a single A100 GPU), although experimenting with smaller datasets is much faster than with larger ones.

If you found GraphLand datasets or this repository useful, please cite the following work:

```
@article{bazhenov2025graphland,
  title={GraphLand: Evaluating graph machine learning models on diverse industrial data},
  author={Bazhenov, Gleb and Platonov, Oleg and Prokhorenkova, Liudmila},
  journal={arXiv preprint},
  year={2025}
}
```
