import os
import yaml
from functools import partial

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import dgl

from sklearn.preprocessing import (FunctionTransformer, StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer,
                                   QuantileTransformer, OneHotEncoder)
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, r2_score
from sklearn.model_selection import train_test_split

from torch_geometric import datasets as pyg_datasets
from ogb.nodeproppred import NodePropPredDataset


class Dataset:
    # Datasets by source.
    # Automatic downloading is currently not supported for GraphLand datasets. If you want to use one of these datasets,
    # put it in the data directory.
    graphland_datasets_names = [
        'hm-categories', 'hm-prices', 'avazu-ctr', 'tolokers-2', 'artnet-views', 'artnet-exp', 'twitch-views',
        'city-roads-M', 'city-roads-L', 'city-reviews', 'pokec-regions', 'web-fraud', 'web-traffic', 'web-topics'
    ]
    pyg_datasets_names = [
        'roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions', 'cora', 'citeseer', 'pubmed',
        'coauthor-cs', 'coauthor-physics', 'amazon-computers', 'amazon-photo', 'lastfm-asia', 'facebook', 'wiki-cs',
        'flickr'
    ]
    ogb_datasets_names = ['ogbn-arxiv', 'ogbn-products']

    # Datasets by task.
    multiclass_classification_datasets_names = [
        'hm-categories', 'pokec-regions', 'web-topics', 'roman-empire', 'amazon-ratings', 'cora', 'citeseer', 'pubmed',
        'coauthor-cs', 'coauthor-physics', 'amazon-computers', 'amazon-photo', 'lastfm-asia', 'facebook', 'wiki-cs',
        'flickr', 'ogbn-arxiv', 'ogbn-products'
    ]
    binary_classification_datasets_names = [
        'tolokers-2', 'artnet-exp', 'city-reviews', 'web-fraud', 'minesweeper', 'tolokers', 'questions',
    ]
    regression_datasets_names = [
        'hm-prices', 'avazu-ctr', 'artnet-views', 'twitch-views', 'city-roads-M', 'city-roads-L', 'web-traffic'
    ]

    # Not all datasets obtained from PyG have predefined data splits. Random class stratified splits will be used for
    # other datasets.
    pyg_datasets_with_predefined_splits_names = [
        'roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions', 'flickr'
    ]

    transforms = {
        'none': partial(FunctionTransformer, func=lambda x: x, inverse_func=lambda x: x),
        'standard-scaler': partial(StandardScaler, copy=False),
        'min-max-scaler': partial(MinMaxScaler, clip=False, copy=False),
        'robust-scaler': partial(RobustScaler, copy=False),
        'power-transform-yeo-johnson': partial(PowerTransformer, method='yeo-johnson', standardize=True, copy=False),
        'quantile-transform-normal': partial(QuantileTransformer, output_distribution='normal', subsample=None,
                                             random_state=0, copy=False),
        'quantile-transform-uniform': partial(QuantileTransformer, output_distribution='uniform', subsample=None,
                                              random_state=0, copy=False)
    }

    def __init__(self, name, split=None, transductive=True, add_self_loops=False, node_embeddings=None,
                 regression_targets_transform='none', numerical_features_transform='none',
                 fraction_features_transform='none', numerical_features_nan_imputation_strategy='most_frequent',
                 fraction_features_nan_imputation_strategy='most_frequent', device='cpu'):
        print('Preparing data...')
        if name in self.graphland_datasets_names:
            if transductive:
                (graph, features, targets, train_mask, val_mask, test_mask,
                 numerical_features_mask, fraction_features_mask) = \
                    self.get_graphland_transductive_dataset(name=name, split=split, add_self_loops=add_self_loops,
                                                            node_embeddings_name=node_embeddings)

            else:
                (train_graph, train_features, train_targets, train_mask,
                 val_graph, val_features, val_targets, val_mask,
                 test_graph, test_features, test_targets, test_mask,
                 numerical_features_mask, fraction_features_mask) = \
                    self.get_graphland_inductive_dataset(name=name, split=split, add_self_loops=add_self_loops,
                                                         node_embeddings_name=node_embeddings)

        elif name in self.pyg_datasets_names:
            graph, features, targets, train_mask, val_mask, test_mask = self.get_pyg_dataset(
                name=name, add_self_loops=add_self_loops
            )
            numerical_features_mask, fraction_features_mask = None, None

        elif name in self.ogb_datasets_names:
            graph, features, targets, train_mask, val_mask, test_mask = self.get_ogb_dataset(
                name=name, add_self_loops=add_self_loops
            )
            numerical_features_mask, fraction_features_mask = None, None

        else:
            raise ValueError(f'Unkown dataset name: {name}.')

        if name in self.multiclass_classification_datasets_names:
            task = 'multiclass_classification'
            metric_name = 'accuracy'
            loss_fn = F.cross_entropy
            if transductive:
                targets_dim = len(targets[~targets.isnan()].unique())
                targets = targets.to(torch.int64)
            else:
                targets_dim = len(train_targets[~train_targets.isnan()].unique())
                train_targets = train_targets.to(torch.int64)
                val_targets = val_targets.to(torch.int64)
                test_targets = test_targets.to(torch.int64)

        elif name in self.binary_classification_datasets_names:
            task = 'binary_classification'
            metric_name = 'AP'
            loss_fn = F.binary_cross_entropy_with_logits
            targets_dim = 1
            if transductive:
                targets = targets.to(torch.float32)
            else:
                train_targets = train_targets.to(torch.float32)
                val_targets = val_targets.to(torch.float32)
                test_targets = test_targets.to(torch.float32)

        elif name in self.regression_datasets_names:
            task = 'regression'
            metric_name = 'R2'
            loss_fn = F.mse_loss
            targets_dim = 1

        else:
            raise RuntimeError(f'The task for dataset {name} is not known.')

        self.name = name
        self.task = task
        self.metric_name = metric_name
        self.loss_fn = loss_fn
        self.transductive = transductive
        self.device = device

        if transductive:
            self.graph = graph.to(device)
            self.features = features.to(device)
            self.targets = targets.to(device)
            self.train_mask = train_mask.to(device)
            self.val_mask = val_mask.to(device)
            self.test_mask = test_mask.to(device)

        else:
            self.train_graph = train_graph.to(device)
            self.train_features = train_features.to(device)
            self.train_targets = train_targets.to(device)
            self.train_mask = train_mask.to(device)

            self.val_graph = val_graph.to(device)
            self.val_features = val_features.to(device)
            self.val_targets = val_targets.to(device)
            self.val_mask = val_mask.to(device)

            self.test_graph = test_graph.to(device)
            self.test_features = test_features.to(device)
            self.test_targets = test_targets.to(device)
            self.test_mask = test_mask.to(device)

        self.features_dim = features.shape[1] if transductive else train_features.shape[1]
        self.targets_dim = targets_dim

        if task == 'regression':
            self.regression_targets_transform_name = None
            self.regression_targets_transform = None
            if transductive:
                self.targets_orig = targets.clone().numpy()
            else:
                self.train_targets_orig = train_targets.clone().numpy()
                self.val_targets_orig = val_targets.clone().numpy()
                self.test_targets_orig = test_targets.clone().numpy()

        if numerical_features_mask is None:
            self.numerical_features_mask = None
        else:
            self.numerical_features_mask = numerical_features_mask.to(device)
            self.numerical_features_transform_name = None
            self.numerical_features_nan_imputation_strategy = None
            if transductive:
                self.numerical_features_orig = features[:, numerical_features_mask].clone().numpy()
            else:
                self.train_numerical_features_orig = train_features[:, numerical_features_mask].clone().numpy()
                self.val_numerical_features_orig = val_features[:, numerical_features_mask].clone().numpy()
                self.test_numerical_features_orig = test_features[:, numerical_features_mask].clone().numpy()

        if fraction_features_mask is None:
            self.fraction_features_mask = None
        else:
            self.fraction_features_mask = fraction_features_mask.to(device)
            self.fraction_features_transform_name = None
            self.fraction_features_nan_imputation_strategy = None
            if transductive:
                self.fraction_features_orig = features[:, fraction_features_mask].clone().numpy()
            else:
                self.train_fraction_features_orig = train_features[:, fraction_features_mask].clone().numpy()
                self.val_fraction_features_orig = val_features[:, fraction_features_mask].clone().numpy()
                self.test_fraction_features_orig = test_features[:, fraction_features_mask].clone().numpy()

        self.apply_transforms(regression_targets_transform_name=regression_targets_transform,
                              numerical_features_transform_name=numerical_features_transform,
                              fraction_features_transform_name=fraction_features_transform,
                              numerical_features_nan_imputation_strategy=numerical_features_nan_imputation_strategy,
                              fraction_features_nan_imputation_strategy=fraction_features_nan_imputation_strategy)

    def apply_transforms(self, regression_targets_transform_name, numerical_features_transform_name,
                         fraction_features_transform_name, numerical_features_nan_imputation_strategy,
                         fraction_features_nan_imputation_strategy):
        if self.task == 'regression' and regression_targets_transform_name != self.regression_targets_transform_name:
            self.transform_targets(transform_name=regression_targets_transform_name)

        if (
                self.numerical_features_mask is not None and
                (
                        numerical_features_transform_name != self.numerical_features_transform_name or
                        numerical_features_nan_imputation_strategy != self.numerical_features_nan_imputation_strategy
                )
        ):
            self.transform_features(features_type='numerical',
                                    transform_name=numerical_features_transform_name,
                                    nan_imputation_strategy=numerical_features_nan_imputation_strategy)

        if (
                self.fraction_features_mask is not None and
                (
                        fraction_features_transform_name != self.fraction_features_transform_name or
                        fraction_features_nan_imputation_strategy != self.fraction_features_nan_imputation_strategy
                )
        ):
            self.transform_features(features_type='fraction',
                                    transform_name=fraction_features_transform_name,
                                    nan_imputation_strategy=fraction_features_nan_imputation_strategy)

    def apply_transforms_from_args(self, args):
        self.apply_transforms(
            regression_targets_transform_name=args.regression_targets_transform,
            numerical_features_transform_name=args.numerical_features_transform,
            fraction_features_transform_name=args.fraction_features_transform,
            numerical_features_nan_imputation_strategy=args.numerical_features_nan_imputation_strategy,
            fraction_features_nan_imputation_strategy=args.fraction_features_nan_imputation_strategy
        )

    def transform_targets(self, transform_name):
        transform = self.transforms[transform_name]()

        if self.transductive:
            train_mask = self.train_mask.cpu().numpy()
            transform.fit(self.targets_orig[train_mask, None])
            targets = transform.transform(self.targets_orig.copy()[:, None]).squeeze(1)
            self.targets = torch.tensor(targets, device=self.device)

        else:
            transform.fit(self.train_targets_orig[:, None])

            train_targets = transform.transform(self.train_targets_orig.copy()[:, None]).squeeze(1)
            val_targets = transform.transform(self.val_targets_orig.copy()[:, None]).squeeze(1)
            test_targets = transform.transform(self.test_targets_orig.copy()[:, None]).squeeze(1)

            self.train_targets = torch.tensor(train_targets, device=self.device)
            self.val_targets = torch.tensor(val_targets, device=self.device)
            self.test_targets = torch.tensor(test_targets, device=self.device)

        self.regression_targets_transform_name = transform_name
        self.regression_targets_transform = transform

    def transform_features(self, features_type, transform_name, nan_imputation_strategy):
        transform = self.transforms[transform_name]()
        imputer = SimpleImputer(missing_values=np.nan, strategy=nan_imputation_strategy, copy=False)

        if features_type == 'numerical':
            mask = self.numerical_features_mask
        elif features_type == 'fraction':
            mask = self.fraction_features_mask
        else:
            raise ValueError(
                f'Unknown features type: {features_type}. Supported values are "numerical" and "fraction".'
            )

        if self.transductive:
            if features_type == 'numerical':
                features_orig = self.numerical_features_orig
            elif features_type == 'fraction':
                features_orig = self.fraction_features_orig
            else:
                raise ValueError(
                    f'Unknown features type: {features_type}. Supported values are "numerical" and "fraction".'
                )

            transform.fit(features_orig)
            features = transform.transform(features_orig.copy())

            if np.isnan(features).any():
                imputer.fit(features)
                features = imputer.transform(features)

            self.features[:, mask] = torch.tensor(features, device=self.device)

        else:
            if features_type == 'numerical':
                train_features_orig = self.train_numerical_features_orig
                val_features_orig = self.val_numerical_features_orig
                test_features_orig = self.test_numerical_features_orig
            elif features_type == 'fraction':
                train_features_orig = self.train_fraction_features_orig
                val_features_orig = self.val_fraction_features_orig
                test_features_orig = self.test_fraction_features_orig
            else:
                raise ValueError(
                    f'Unknown features type: {features_type}. Supported values are "numerical" and "fraction".'
                )

            transform.fit(train_features_orig)
            train_features = transform.transform(train_features_orig.copy())
            val_features = transform.transform(val_features_orig.copy())
            test_features = transform.transform(test_features_orig.copy())

            if np.isnan(train_features).any() or np.isnan(val_features).any() or np.isnan(test_features).any():
                imputer.fit(train_features)
                train_features = imputer.transform(train_features)
                val_features = imputer.transform(val_features)
                test_features = imputer.transform(test_features)

            self.train_features[:, mask] = torch.tensor(train_features, device=self.device)
            self.val_features[:, mask] = torch.tensor(val_features, device=self.device)
            self.test_features[:, mask] = torch.tensor(test_features, device=self.device)

        if features_type == 'numerical':
            self.numerical_features_transform_name = transform_name
            self.numerical_features_nan_imputation_strategy = nan_imputation_strategy
        elif features_type == 'fraction':
            self.fraction_features_transform_name = transform_name
            self.fraction_features_nan_imputation_strategy = nan_imputation_strategy
        else:
            raise ValueError(
                f'Unknown features type: {features_type}. Supported values are "numerical" and "fraction".'
            )

    def compute_metrics_transductive(self, preds):
        if self.metric_name == 'accuracy':
            preds = preds.argmax(axis=1)
            train_metric = (preds[self.train_mask] == self.targets[self.train_mask]).float().mean().item()
            val_metric = (preds[self.val_mask] == self.targets[self.val_mask]).float().mean().item()
            test_metric = (preds[self.test_mask] == self.targets[self.test_mask]).float().mean().item()

        elif self.metric_name == 'AP':
            targets = self.targets.cpu().numpy()
            preds = preds.cpu().numpy()

            train_mask = self.train_mask.cpu().numpy()
            val_mask = self.val_mask.cpu().numpy()
            test_mask = self.test_mask.cpu().numpy()

            train_metric = average_precision_score(y_true=targets[train_mask], y_score=preds[train_mask]).item()
            val_metric = average_precision_score(y_true=targets[val_mask], y_score=preds[val_mask]).item()
            test_metric = average_precision_score(y_true=targets[test_mask], y_score=preds[test_mask]).item()

        elif self.metric_name == 'R2':
            preds_orig = self.regression_targets_transform.inverse_transform(preds.cpu().numpy()[:, None]).squeeze(1)

            train_mask = self.train_mask.cpu().numpy()
            val_mask = self.val_mask.cpu().numpy()
            test_mask = self.test_mask.cpu().numpy()

            train_metric = r2_score(y_true=self.targets_orig[train_mask], y_pred=preds_orig[train_mask])
            val_metric = r2_score(y_true=self.targets_orig[val_mask], y_pred=preds_orig[val_mask])
            test_metric = r2_score(y_true=self.targets_orig[test_mask], y_pred=preds_orig[test_mask])

        else:
            raise ValueError(f'Unknown metric: {self.metric_name}.')

        metrics = {
            f'train {self.metric_name}': train_metric,
            f'val {self.metric_name}': val_metric,
            f'test {self.metric_name}': test_metric
        }

        return metrics

    def compute_val_metric_inductive(self, preds):
        if self.metric_name == 'accuracy':
            preds = preds.argmax(axis=1)
            val_metric = (preds[self.val_mask] == self.val_targets[self.val_mask]).float().mean().item()

        elif self.metric_name == 'AP':
            val_targets = self.val_targets.cpu().numpy()
            preds = preds.cpu().numpy()
            val_mask = self.val_mask.cpu().numpy()
            val_metric = average_precision_score(y_true=val_targets[val_mask], y_score=preds[val_mask]).item()

        elif self.metric_name == 'R2':
            preds_orig = self.regression_targets_transform.inverse_transform(preds.cpu().numpy()[:, None]).squeeze(1)
            val_mask = self.val_mask.cpu().numpy()
            val_metric = r2_score(y_true=self.val_targets_orig[val_mask], y_pred=preds_orig[val_mask])

        else:
            raise ValueError(f'Unknown metric: {self.metric_name}.')

        return val_metric

    def compute_test_metric_inductive(self, preds):
        if self.metric_name == 'accuracy':
            preds = preds.argmax(axis=1)
            test_metric = (preds[self.test_mask] == self.test_targets[self.test_mask]).float().mean().item()

        elif self.metric_name == 'AP':
            test_targets = self.test_targets.cpu().numpy()
            preds = preds.cpu().numpy()
            test_mask = self.test_mask.cpu().numpy()
            test_metric = average_precision_score(y_true=test_targets[test_mask], y_score=preds[test_mask]).item()

        elif self.metric_name == 'R2':
            preds_orig = self.regression_targets_transform.inverse_transform(preds.cpu().numpy()[:, None]).squeeze(1)
            test_mask = self.test_mask.cpu().numpy()
            test_metric = r2_score(y_true=self.test_targets_orig[test_mask], y_pred=preds_orig[test_mask])

        else:
            raise ValueError(f'Unknown metric: {self.metric_name}.')

        return test_metric

    @staticmethod
    def get_graphland_transductive_dataset(name, split, add_self_loops, node_embeddings_name):
        with open(f'data/{name}/info.yaml', 'r') as file:
            info = yaml.safe_load(file)

        fraction_features_names_set = set(info['fraction_features_names'])
        numerical_features_names = [
            name for name in info['numerical_features_names'] if name not in fraction_features_names_set
        ]

        features_df = pd.read_csv(f'data/{name}/features.csv', index_col=0)
        numerical_features = features_df[numerical_features_names].values.astype(np.float32)
        fraction_features = features_df[info['fraction_features_names']].values.astype(np.float32)
        categorical_features = features_df[info['categorical_features_names']].values.astype(np.float32)

        if categorical_features.shape[1] > 0:
            one_hot_encoder = OneHotEncoder(drop='if_binary', sparse_output=False, dtype=np.float32)
            categorical_features = one_hot_encoder.fit_transform(categorical_features)

        features = np.concatenate([numerical_features, fraction_features, categorical_features], axis=1)

        if node_embeddings_name is not None:
            node_embeddings = np.load(f'data/{name}/{node_embeddings_name}.npy')
            features = np.concatenate([features, node_embeddings], axis=1)

        if numerical_features.shape[1] > 0:
            numerical_features_mask = np.zeros(features.shape[1], dtype=bool)
            numerical_features_mask[:numerical_features.shape[1]] = True
        else:
            numerical_features_mask = None

        if fraction_features.shape[1] > 0:
            fraction_features_mask = np.zeros(features.shape[1], dtype=bool)
            fraction_features_mask[
                numerical_features.shape[1]:numerical_features.shape[1] + fraction_features.shape[1]
            ] = True
        else:
            fraction_features_mask = None

        targets = pd.read_csv(f'data/{name}/targets.csv', index_col=0).values.squeeze(1).astype(np.float32)

        edges_df = pd.read_csv(f'data/{name}/edgelist.csv')
        edges = edges_df.values[:, :2]

        split_masks_df = pd.read_csv(f'data/{name}/split_masks_{split}.csv', index_col=0)
        train_mask_orig = split_masks_df['train'].values
        val_mask_orig = split_masks_df['val'].values
        test_mask_orig = split_masks_df['test'].values

        labeled_mask = ~np.isnan(targets)
        train_mask = (train_mask_orig & labeled_mask)
        val_mask = (val_mask_orig & labeled_mask)
        test_mask = (test_mask_orig & labeled_mask)

        graph = Dataset.get_graph(edges=edges, num_nodes=len(features), add_self_loops=add_self_loops)

        features = torch.tensor(features)
        targets = torch.tensor(targets)
        train_mask = torch.tensor(train_mask)
        val_mask = torch.tensor(val_mask)
        test_mask = torch.tensor(test_mask)

        if numerical_features_mask is not None:
            numerical_features_mask = torch.tensor(numerical_features_mask)

        if fraction_features_mask is not None:
            fraction_features_mask = torch.tensor(fraction_features_mask)

        return (
            graph, features, targets, train_mask, val_mask, test_mask, numerical_features_mask, fraction_features_mask
        )

    @staticmethod
    def get_graphland_inductive_dataset(name, split, add_self_loops, node_embeddings_name):
        with open(f'data/{name}/info.yaml', 'r') as file:
            info = yaml.safe_load(file)

        split_masks_df = pd.read_csv(f'data/{name}/split_masks_{split}.csv', index_col=0)
        train_mask_orig = split_masks_df['train'].values
        val_mask_orig = split_masks_df['val'].values
        test_mask_orig = split_masks_df['test'].values

        fraction_features_names_set = set(info['fraction_features_names'])
        numerical_features_names = [
            name for name in info['numerical_features_names'] if name not in fraction_features_names_set
        ]

        features_df = pd.read_csv(f'data/{name}/features.csv', index_col=0)
        numerical_features = features_df[numerical_features_names].values.astype(np.float32)
        fraction_features = features_df[info['fraction_features_names']].values.astype(np.float32)
        categorical_features = features_df[info['categorical_features_names']].values.astype(np.float32)

        if categorical_features.shape[1] > 0:
            one_hot_encoder = OneHotEncoder(drop='if_binary', sparse_output=False, dtype=np.float32,
                                            handle_unknown='ignore')
            one_hot_encoder = one_hot_encoder.fit(categorical_features[train_mask_orig])
            categorical_features = one_hot_encoder.transform(categorical_features)

        features = np.concatenate([numerical_features, fraction_features, categorical_features], axis=1)

        if node_embeddings_name is not None:
            node_embeddings = np.load(f'data/{name}/{node_embeddings_name}.npy')
            features = np.concatenate([features, node_embeddings], axis=1)

        if numerical_features.shape[1] > 0:
            numerical_features_mask = np.zeros(features.shape[1], dtype=bool)
            numerical_features_mask[:numerical_features.shape[1]] = True
        else:
            numerical_features_mask = None

        if fraction_features.shape[1] > 0:
            fraction_features_mask = np.zeros(features.shape[1], dtype=bool)
            fraction_features_mask[
                numerical_features.shape[1]:numerical_features.shape[1] + fraction_features.shape[1]
            ] = True
        else:
            fraction_features_mask = None

        targets = pd.read_csv(f'data/{name}/targets.csv', index_col=0).values.squeeze(1).astype(np.float32)

        edges_df = pd.read_csv(f'data/{name}/edgelist.csv')
        edges = edges_df.values[:, :2]

        train_and_val_mask_orig = (train_mask_orig | val_mask_orig)

        train_edges = Dataset.get_induced_subgraph_edges(edges=edges, nodes_to_keep=np.where(train_mask_orig)[0])
        train_features = features[train_mask_orig]
        train_targets = targets[train_mask_orig]

        val_edges = Dataset.get_induced_subgraph_edges(edges=edges, nodes_to_keep=np.where(train_and_val_mask_orig)[0])
        val_features = features[train_and_val_mask_orig]
        val_targets = targets[train_and_val_mask_orig]

        test_edges = edges
        test_features = features
        test_targets = targets

        labeled_mask = ~np.isnan(targets)
        train_mask = labeled_mask[train_mask_orig]
        val_mask = (val_mask_orig & labeled_mask)[train_and_val_mask_orig]
        test_mask = (test_mask_orig & labeled_mask)

        train_graph = Dataset.get_graph(edges=train_edges, num_nodes=len(train_features), add_self_loops=add_self_loops)
        val_graph = Dataset.get_graph(edges=val_edges, num_nodes=len(val_features), add_self_loops=add_self_loops)
        test_graph = Dataset.get_graph(edges=test_edges, num_nodes=len(test_features), add_self_loops=add_self_loops)

        train_features = torch.tensor(train_features)
        train_targets = torch.tensor(train_targets)
        train_mask = torch.tensor(train_mask)

        val_features = torch.tensor(val_features)
        val_targets = torch.tensor(val_targets)
        val_mask = torch.tensor(val_mask)

        test_features = torch.tensor(test_features)
        test_targets = torch.tensor(test_targets)
        test_mask = torch.tensor(test_mask)

        if numerical_features_mask is not None:
            numerical_features_mask = torch.tensor(numerical_features_mask)

        if fraction_features_mask is not None:
            fraction_features_mask = torch.tensor(fraction_features_mask)

        return (
            train_graph, train_features, train_targets, train_mask,
            val_graph, val_features, val_targets, val_mask,
            test_graph, test_features, test_targets, test_mask,
            numerical_features_mask, fraction_features_mask
        )

    @staticmethod
    def get_pyg_dataset(name, add_self_loops):
        if name in ['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions']:
            dataset = pyg_datasets.HeterophilousGraphDataset(name=name, root='data')
        elif name in ['cora', 'citeseer', 'pubmed']:
            dataset = pyg_datasets.Planetoid(name=name, root='data')
        elif name in ['coauthor-cs', 'coauthor-physics']:
            dataset = pyg_datasets.Coauthor(name=name.split('-')[1], root=os.path.join('data', 'coauthor'))
        elif name in ['amazon-computers', 'amazon-photo']:
            dataset = pyg_datasets.Amazon(name=name.split('-')[1], root=os.path.join('data', 'amazon'))
        elif name == 'lastfm-asia':
            dataset = pyg_datasets.LastFMAsia(root=os.path.join('data', name))
        elif name == 'facebook':
            dataset = pyg_datasets.FacebookPagePage(root=os.path.join('data', name))
        elif name == 'wiki-cs':
            dataset = pyg_datasets.WikiCS(root=os.path.join('data', name), is_undirected=True)
        elif name == 'flickr':
            dataset = pyg_datasets.Flickr(root=os.path.join('data', name))
        else:
            raise ValueError(f'Unknown PyG dataset name: {name}.')

        pyg_data = dataset[0]
        features = pyg_data.x
        targets = pyg_data.y
        num_nodes = len(features)
        edges = pyg_data.edge_index.T
        graph = Dataset.get_graph(edges=edges, num_nodes=num_nodes, add_self_loops=add_self_loops)

        # Get data splits.
        if name in Dataset.pyg_datasets_with_predefined_splits_names:
            if pyg_data.train_mask.dim() == 1:
                # These datasets have a single predefined data split.
                train_mask = pyg_data.train_mask
                val_mask = pyg_data.val_mask
                test_mask = pyg_data.test_mask

            else:
                # These datasets have several predefined data splits, but we will only use the first one.
                train_mask = pyg_data.train_mask[:, 0]
                val_mask = pyg_data.val_mask[:, 0]
                test_mask = pyg_data.test_mask[:, 0]

        else:
            # A random stratified by class data split will be created.
            train_idx, val_and_test_idx = train_test_split(torch.arange(num_nodes), test_size=0.5, random_state=0,
                                                           stratify=targets)
            val_idx, test_idx = train_test_split(val_and_test_idx, test_size=0.5, random_state=0,
                                                 stratify=targets[val_and_test_idx])

            train_mask = torch.zeros_like(targets, dtype=torch.bool)
            train_mask[train_idx] = True

            val_mask = torch.zeros_like(targets, dtype=torch.bool)
            val_mask[val_idx] = True

            test_mask = torch.zeros_like(targets, dtype=torch.bool)
            test_mask[test_idx] = True

        return graph, features, targets, train_mask, val_mask, test_mask

    @staticmethod
    def get_ogb_dataset(name, add_self_loops):
        dataset = NodePropPredDataset(name=name, root='data')
        data, targets = dataset[0]
        targets = torch.tensor(targets.squeeze(1))
        features = torch.tensor(data['node_feat'])
        edges = data['edge_index'].T
        graph = Dataset.get_graph(edges=edges, num_nodes=len(features), add_self_loops=add_self_loops)

        split = dataset.get_idx_split()
        train_idx = split['train']
        val_idx = split['valid']
        test_idx = split['test']

        train_mask = torch.zeros_like(targets, dtype=torch.bool)
        train_mask[train_idx] = True

        val_mask = torch.zeros_like(targets, dtype=torch.bool)
        val_mask[val_idx] = True

        test_mask = torch.zeros_like(targets, dtype=torch.bool)
        test_mask[test_idx] = True

        return graph, features, targets, train_mask, val_mask, test_mask

    @staticmethod
    def get_graph(edges, num_nodes, add_self_loops):
        graph = dgl.graph((edges[:, 0], edges[:, 1]), num_nodes=num_nodes, idtype=torch.int32)

        graph = dgl.remove_self_loop(graph)
        graph = dgl.to_simple(graph)
        graph = dgl.to_bidirected(graph)

        if add_self_loops:
            graph = dgl.add_self_loop(graph)

        return graph

    @staticmethod
    def get_induced_subgraph_edges(edges, nodes_to_keep):
        old_node_id_to_new_node_id = {
            old_node_id: new_node_id for new_node_id, old_node_id in enumerate(sorted(nodes_to_keep))
        }

        nodes_to_keep = set(nodes_to_keep)

        edges = np.array([
            (old_node_id_to_new_node_id[u], old_node_id_to_new_node_id[v]) for u, v in edges
            if u in nodes_to_keep and v in nodes_to_keep
        ])

        return edges
