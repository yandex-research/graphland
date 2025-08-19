from abc import ABC, abstractmethod
from copy import deepcopy
from itertools import product
import optuna


class BaseHparamGenerator(ABC):
    @abstractmethod
    def __init__(self, args):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def start_trial(self):
        raise NotImplementedError

    @abstractmethod
    def finish_trial(self, val_metric):
        raise NotImplementedError


class GridSearchHparamGenerator(BaseHparamGenerator):
    def __init__(self, args):
        # Get hparams that have multiple values.
        hparam_lists = {key: value for key, value in vars(args).items() if isinstance(value, (list, tuple))}

        # Dict of lists to list of dicts.
        hparam_values = product(*hparam_lists.values())
        hparam_dicts = [dict(zip(hparam_lists.keys(), values)) for values in hparam_values]

        self.hparam_dicts = hparam_dicts
        self.hparam_id = -1

    def __len__(self):
        return len(self.hparam_dicts)

    def start_trial(self):
        self.hparam_id += 1
        hparams = deepcopy(self.hparam_dicts[self.hparam_id])

        return hparams

    def finish_trial(self, val_metric):
        pass


class OptunaHparamGenerator(BaseHparamGenerator):
    """WIP. Optuna hyperparameter generator has not been tested yet."""
    lr_map = [1e-5, 2e-5, 3e-5, 5e-5, 7e-5, 1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3, 2e-3, 3e-3, 5e-3, 7e-3, 1e-2]

    def __init__(self, args):
        distributions = {
            key: value for key, value in vars(args).items() if isinstance(value, optuna.distributions.BaseDistribution)
        }

        sampler = optuna.samplers.TPESampler(seed=0, n_startup_trials=10)
        study = optuna.create_study(sampler=sampler, direction='maximize')

        if args.predefined_hparam_combs is not None:
            for hparams in args.predefined_hparam_combs:
                if hparams.keys() != distributions.keys():
                    raise ValueError(
                        f'The set of predefined hparams {set(hparams.keys())} does not match the set of hparams to be '
                        f'searched {set(distributions.keys())}.'
                    )

                study.enqueue_trial(hparams)

        self.study = study
        self.distributions = distributions
        self.num_trials = args.num_trials
        self.cur_trial = None

        self.use_lr_map = (
                'lr' in distributions and isinstance(distributions['lr'], optuna.distributions.IntDistribution)
        )

    def __len__(self):
        return self.num_trials

    def start_trial(self):
        self.cur_trial = self.study.ask(self.distributions)
        hparams = deepcopy(self.cur_trial.params)

        if self.use_lr_map:
            hparams['lr'] = self.lr_map[hparams['lr']]

        return hparams

    def finish_trial(self, val_metric):
        self.study.tell(trial=self.cur_trial, values=val_metric)


def get_hparam_generator(args):
    if args.hparam_search_strategy == 'grid-search':
        return GridSearchHparamGenerator(args)
    elif args.hparam_search_strategy == 'optuna':
        return OptunaHparamGenerator(args)
    else:
        raise ValueError(
            f'Hparam generator can only be provided if hparam_search_strategy is "grid-search" or "optuna", '
            f'but hparam_search_strategy is {args.hparam_search_strategy}.'
        )
