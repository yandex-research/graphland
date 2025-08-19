import os
import argparse
from dataclasses import dataclass
from optuna.distributions import BaseDistribution
from utils import read_yaml


def str_to_bool(string):
    if string == 'True':
        return True
    elif string == 'False':
        return False
    else:
        raise ValueError('Only strings "True" and "False" can be converted to a boolean value.')


@dataclass
class ExperimentConfig:
    name: str = 'experiment'
    save_dir: str = 'experiments'
    dataset: str = 'tolokers-2'
    split: str = 'RH'
    transductive: bool = True
    train_regime: str = 'full-graph'
    config: str = None

    # Data preprocessing.
    regression_targets_transform: str = 'standard-scaler'
    numerical_features_transform: str = 'quantile-transform-normal'
    fraction_features_transform: str = 'none'
    numerical_features_nan_imputation_strategy: str = 'most_frequent'
    fraction_features_nan_imputation_strategy: str = 'most_frequent'
    node_embeddings: str = None

    # PLR embeddings for numerical features.
    plr: bool = False
    plr_frequencies_dim: int = 48
    plr_frequencies_scale: float = 0.01
    plr_embedding_dim: int = 16
    plr_lite: bool = False

    # Model architecture.
    model: str = 'GraphSAGE'
    num_layers: int = 3
    hidden_dim: int = 512
    num_heads: int = 4
    hidden_dim_multiplier: float = 1
    normalization: str = 'layernorm'

    # Training hyperparameters.
    lr: float = 3e-4
    dropout: float = 0
    weight_decay: float = 0

    max_steps: int = 1000
    num_warmup_steps: int = None
    warmup_proportion: float = 0
    early_stopping: int = -1

    num_runs_with_best_hparams: int = None
    num_runs_with_each_hparams: int = 1
    num_optuna_trials: int = 100

    device: str = 'cuda:0'
    amp: bool = True
    compile: bool = False


def get_default_num_runs_value(dataset_name):
    if dataset_name in ['pokec-regions', 'web-fraud', 'web-traffic', 'web-topics']:
        return 5
    else:
        return 10


def get_args():
    """
    The argument processing logic is a bit tricky. First, arguments are loaded from the default config. Thus, the
    values provided there are the default values. Then, another config provided in the --config argument is loaded and
    all the values provided in it overwrite the default ones. Then, any values provided as command line arguments
    overwrite the previous values (all command line arguments have None as the default value to skip overwriting).
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default=None, help='Experiment name.')
    parser.add_argument('--save_dir', type=str, default=None, help='Base directory for saving information.')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=[
                            # GraphLand datasets
                            'hm-categories', 'hm-prices', 'avazu-ctr', 'tolokers-2', 'artnet-views', 'artnet-exp',
                            'twitch-views', 'city-roads-M', 'city-roads-L', 'city-reviews', 'pokec-regions',
                            'web-fraud', 'web-traffic', 'web-topics',
                            # Heterophilous graphs benchmark datasets
                            'roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions',
                            # Classic datasets
                            'cora', 'citeseer', 'pubmed', 'coauthor-cs', 'coauthor-physics',
                            'amazon-computers', 'amazon-photo', 'lastfm-asia', 'facebook', 'wiki-cs', 'flickr',
                            # OGB datasets
                            'ogbn-arxiv', 'ogbn-products'
                        ])
    parser.add_argument('--split', type=str, default=None)
    parser.add_argument('--transductive', type=str_to_bool, default=None)
    parser.add_argument('--train_regime', type=str, default=None, choices=['full-graph', 'minibatch'],
                        help='WIP. Minibatch training regime has not been implemented yet.')
    parser.add_argument('--config', type=str, default=None,
                        help='Name of a config yaml file in the configs directory. '
                             'WIP. Reading arguments from configs has not been tested yet.')

    # Data preprocessing.
    parser.add_argument('--regression_targets_transform', nargs='+', type=str, default=None,
                        choices=['none', 'standard-scaler', 'min-max-scaler', 'robust-scaler',
                                 'power-transform-yeo-johnson', 'quantile-transform-normal',
                                 'quantile-transform-uniform'],
                        help='Only used for regression datasets.')
    parser.add_argument('--numerical_features_transform', nargs='+', type=str, default=None,
                        choices=['none', 'standard-scaler', 'min-max-scaler', 'robust-scaler',
                                 'power-transform-yeo-johnson', 'quantile-transform-normal',
                                 'quantile-transform-uniform'],
                        help='Only used for GraphLand datasets that have numerical features which are not also '
                             'fraction features.')
    parser.add_argument('--fraction_features_transform', nargs='+', type=str, default=None,
                        choices=['none', 'standard-scaler', 'min-max-scaler', 'robust-scaler',
                                 'power-transform-yeo-johnson', 'quantile-transform-normal',
                                 'quantile-transform-uniform'],
                        help='Only used for GraphLand datasets that have fraction features.')
    parser.add_argument('--numerical_features_nan_imputation_strategy', nargs='+', type=str, default=None,
                        choices=['mean', 'median', 'most_frequent'],
                        help='Only used for GraphLand datasets that have NaNs in numerical features which are not also '
                             'fraction features.')
    parser.add_argument('--fraction_features_nan_imputation_strategy', nargs='+', type=str, default=None,
                        choices=['mean', 'median', 'most_frequent'],
                        help='Only used for GraphLand datasets that have NaNs in fraction features.')
    parser.add_argument('--node_embeddings', nargs='+', type=str, default=None,
                        help='Name of npy file containing node embeddings to use as additional node features.')

    # PLR embeddings for numerical features.
    parser.add_argument('--plr', nargs='+', type=str_to_bool, default=None,
                        help='Use PLR embeddings for numerical features which are not also fraction features. '
                             'Can only be used for GraphLand datasets.')
    parser.add_argument('--plr_frequencies_dim', nargs='+', type=int, default=None,
                        help='Only used if plr is True.')
    parser.add_argument('--plr_frequencies_scale', nargs='+', type=float, default=None,
                        help='Only used if plr is True.')
    parser.add_argument('--plr_embedding_dim', nargs='+', type=int, default=None,
                        help='Only used if plr is True.')
    parser.add_argument('--plr_lite', nargs='+', type=str_to_bool, default=None,
                        help='Only used if plr is True.')

    # Model architecture.
    parser.add_argument('--model', nargs='+', type=str, default=None,
                        choices=['ResMLP', 'GCN', 'GraphSAGE', 'GAT', 'GAT-sep', 'GT', 'GT-sep'])
    parser.add_argument('--num_layers', nargs='+', type=int, default=None)
    parser.add_argument('--hidden_dim', nargs='+', type=int, default=None)
    parser.add_argument('--num_heads', nargs='+', type=int, default=None)
    parser.add_argument('--hidden_dim_multiplier', nargs='+', type=float, default=None)
    parser.add_argument('--normalization', nargs='+', type=str, default=None,
                        choices=['none', 'layernorm', 'batchnorm'])

    # Training hyperparameters.
    parser.add_argument('--lr', nargs='+', type=float, default=None)
    parser.add_argument('--dropout', nargs='+', type=float, default=None)
    parser.add_argument('--weight_decay', nargs='+', type=float, default=None)

    parser.add_argument('--max_steps', nargs='+', type=int, default=None)
    parser.add_argument('--num_warmup_steps', nargs='+', type=int, default=None,
                        help='If None, warmup_proportion is used instead.')
    parser.add_argument('--warmup_proportion', nargs='+', type=float, default=None,
                        help='Only used if num_warmup_steps is None.')
    parser.add_argument('--early_stopping', nargs='+', type=int, default=None,
                        help='Stop training after this many steps without improvement in validation metric. '
                             'If -1, early stopping is disabled.')

    parser.add_argument('--num_runs_with_best_hparams', type=int, default=None,
                        help='If None, dataset-specific default values is used.')
    parser.add_argument('--num_runs_with_each_hparams', type=int, default=None)
    parser.add_argument('--num_optuna_trials', type=int, default=None)

    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--amp', type=str_to_bool, default=None)
    parser.add_argument('--compile', type=str_to_bool, default=None)

    args = parser.parse_args()

    if args.config is not None:
        config_path = os.path.join('configs', args.config)
        config_dict = read_yaml(config_path)
        config = ExperimentConfig(**config_dict)
    else:
        config = ExperimentConfig()

    # Fill empty args with values from the config.
    for key, value in vars(config).items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    if args.num_runs_with_best_hparams is None:
        args.num_runs_with_best_hparams = get_default_num_runs_value(args.dataset)

    grid_search, optuna = False, False
    num_grid_search_trials = 1
    for key, value in vars(args).items():
        if isinstance(value, (list, tuple)):
            if len(value) == 1:
                setattr(args, key, value[0])
            else:
                grid_search = True
                num_grid_search_trials *= len(value)

        elif isinstance(value, BaseDistribution):
            optuna = True

    if grid_search and optuna:
        raise ValueError('Lists of argument values and Optuna distributions cannot be used simultaneously.')
    elif grid_search:
        args.hparam_search_strategy = 'grid-search'
        args.num_hparam_search_trials = num_grid_search_trials
    elif optuna:
        # WIP. Optuna hyperparameter search has not been tested yet.
        args.hparam_search_strategy = 'optuna'
        args.num_hparam_search_trials = args.num_optuna_trials
    else:
        args.hparam_search_strategy = 'fixed'
        args.num_hparam_search_trials = None

    return args
