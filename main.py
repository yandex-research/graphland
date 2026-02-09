import torch
from args import get_args
from dataset import Dataset
from model import get_model
from train_loops import get_train_fn
from logger import Logger
from hparam_generators import get_hparam_generator
from utils import update_hparams


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def main():
    args = get_args()

    dataset = Dataset(name=args.dataset,
                      split=args.split,
                      add_self_loops=(args.model in ['GCN', 'GAT', 'GT']),
                      node_embeddings=args.node_embeddings,
                      device=args.device)

    train_fn = get_train_fn(train_regime=args.train_regime, transductive=dataset.transductive)

    logger = Logger(args=args, metric_name=dataset.metric_name)

    if args.hparam_search_strategy != 'fixed':
        hparam_generator = get_hparam_generator(args)
        logger.start_search_phase()
        for trial_id in range(1, len(hparam_generator) + 1):
            hparams = hparam_generator.start_trial()
            logger.start_search_trial(hparams=hparams, trial_id=trial_id)
            args = update_hparams(args=args, hparams=hparams)
            dataset.apply_transforms_from_args(args)
            for run_id in range(1, args.num_runs_with_each_hparams + 1):
                logger.start_search_run(run_id=run_id)
                torch.manual_seed(run_id)
                model = get_model(args=args, dataset=dataset)
                run_results = train_fn(model=model, dataset=dataset, args=args, run_id=run_id)
                logger.finish_search_run(run_results=run_results)

            trial_results = logger.finish_search_trial()
            hparam_generator.finish_trial(val_metric=trial_results[f'val {dataset.metric_name} mean'])

        best_hparams = logger.finish_search_phase()
        args = update_hparams(args=args, hparams=best_hparams)

    dataset.apply_transforms_from_args(args)
    num_finished_runs_with_best_hparams = logger.start_main_trial()
    for run_id in range(num_finished_runs_with_best_hparams + 1, args.num_runs_with_best_hparams + 1):
        logger.start_main_run(run_id=run_id)
        torch.manual_seed(run_id)
        model = get_model(args=args, dataset=dataset)
        run_results = train_fn(model=model, dataset=dataset, args=args, run_id=run_id)
        logger.finish_main_run(run_results=run_results)

    logger.finish_main_trial()


if __name__ == '__main__':
    main()
