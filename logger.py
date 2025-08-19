import os
from copy import deepcopy
import numpy as np
from utils import write_yaml


class Logger:
    def __init__(self, args, metric_name):
        self.save_dir = self._get_save_dir(base_dir=args.save_dir, dataset=args.dataset, name=args.name)
        self.metric_name = metric_name
        self.num_runs_with_best_hparams = args.num_runs_with_best_hparams
        self.num_runs_with_each_hparams = args.num_runs_with_each_hparams
        self.num_hparam_search_trials = args.num_hparam_search_trials

        self.results = {
            'main results': None,
            'best hparams': None,
            'hparam search results': None
        }

        print(f'Info will be saved to {self.save_dir}.')
        write_yaml(vars(args), os.path.join(self.save_dir, 'args.yaml'))

    def start_main_trial(self):
        print('Starting the main trial...\n')

        if self.results['main results'] is None:
            self.results['main results'] = self._get_empty_trial_results_dict()
            num_finished_runs_with_best_hparams = 0
        else:
            num_finished_runs_with_best_hparams = self.results['main results']['num runs']
            for run_id, (val_metric, test_metric, step, successful) in enumerate(
                    zip(
                        self.results['main results'][f'val {self.metric_name} values'],
                        self.results['main results'][f'test {self.metric_name} values'],
                        self.results['main results']['best steps'],
                        self.results['main results']['successful']
                    ),
                    start=1
            ):
                print(f'Run {run_id} finished during hparam search phase. '
                      f'Best val {self.metric_name}: {val_metric:.4f}, '
                      f'corresponding test {self.metric_name}: {test_metric:.4f} '
                      f'(step {step}).')

                if not successful:
                    print('An error occured during the run!')

                print()

        return num_finished_runs_with_best_hparams

    def start_main_run(self, run_id):
        print(f'Starting run {run_id}/{self.num_runs_with_best_hparams}...')

    def finish_main_run(self, run_results):
        self._finish_run(run_results=run_results, trial_results=self.results['main results'])

    def finish_main_trial(self):
        self._print_results_summary(results=self.results['main results'])

        return deepcopy(self.results['main results'])

    def start_search_phase(self):
        self.results['hparam search results'] = []

        print('\n' * 3)
        print(f'Starting hparam search ({self.num_hparam_search_trials} trials)...')
        print('\n' * 3)

    def start_search_trial(self, hparams, trial_id):
        self.results['hparam search results'].append(
            {'hparams': hparams, 'results': self._get_empty_trial_results_dict()}
        )

        print(f'Starting hparam search trial {trial_id}/{self.num_hparam_search_trials} using the following hparams:')
        print(', '.join(f'{key}={value}' for key, value in hparams.items()))
        print()

    def start_search_run(self, run_id):
        print(f'Starting run {run_id}/{self.num_runs_with_each_hparams}...')

    def finish_search_run(self, run_results):
        self._finish_run(run_results=run_results, trial_results=self.results['hparam search results'][-1]['results'])

    def finish_search_trial(self):
        self._print_results_summary(results=self.results['hparam search results'][-1]['results'])

        return deepcopy(self.results['hparam search results'][-1]['results'])

    def finish_search_phase(self):
        best_search_trial_info = max(
            self.results['hparam search results'], key=lambda x: x['results'][f'val {self.metric_name} mean']
        )

        self.results['best hparams'] = deepcopy(best_search_trial_info['hparams'])
        self.results['main results'] = deepcopy(best_search_trial_info['results'])

        write_yaml(self.results, os.path.join(self.save_dir, 'results.yaml'))

        print('\n' * 3)
        print('Finished hparam search.')
        print('The best hparams found are:')
        print(', '.join(f'{key}={value}' for key, value in self.results['best hparams'].items()))
        print('\n' * 10)

        return deepcopy(self.results['best hparams'])

    def _finish_run(self, run_results, trial_results):
        trial_results['num runs'] += 1
        trial_results[f'val {self.metric_name} values'].append(run_results[f'val {self.metric_name}'])
        trial_results[f'test {self.metric_name} values'].append(run_results[f'test {self.metric_name}'])
        trial_results['best steps'].append(run_results['step'])
        trial_results['successful'].append(run_results['successful'])
        trial_results[f'val {self.metric_name} mean'] = np.mean(
            trial_results[f'val {self.metric_name} values']
        ).item()
        trial_results[f'test {self.metric_name} mean'] = np.mean(
            trial_results[f'test {self.metric_name} values']
        ).item()
        if trial_results['num runs'] > 1:
            trial_results[f'val {self.metric_name} std'] = np.std(
                trial_results[f'val {self.metric_name} values'], ddof=1
            ).item()
            trial_results[f'test {self.metric_name} std'] = np.std(
                trial_results[f'test {self.metric_name} values'], ddof=1
            ).item()

        write_yaml(self.results, os.path.join(self.save_dir, 'results.yaml'))

        print(f'Finished run {trial_results["num runs"]}. '
              f'Best val {self.metric_name}: {run_results[f"val {self.metric_name}"]:.4f}, '
              f'corresponding test {self.metric_name}: {run_results[f"test {self.metric_name}"]:.4f} '
              f'(step {run_results["step"]}).')

        if not run_results['successful']:
            print('An error occured during the run!')

        print()

    def _print_results_summary(self, results):
        if results['num runs'] <= 1:
            print('\n')
            return

        print(f'Finished {results["num runs"]} runs.')
        print(f'Val {self.metric_name} mean: {results[f"val {self.metric_name} mean"]:.4f}')
        print(f'Val {self.metric_name} std: {results[f"val {self.metric_name} std"]:.4f}')
        print(f'Test {self.metric_name} mean: {results[f"test {self.metric_name} mean"]:.4f}')
        print(f'Test {self.metric_name} std: {results[f"test {self.metric_name} std"]:.4f}')
        print('\n' * 5)

    @staticmethod
    def _get_save_dir(base_dir, dataset, name):
        idx = 1
        save_dir = os.path.join(base_dir, dataset, f'{name}_{idx:02d}')
        while os.path.exists(save_dir):
            idx += 1
            save_dir = os.path.join(base_dir, dataset, f'{name}_{idx:02d}')

        os.makedirs(save_dir)

        return save_dir

    def _get_empty_trial_results_dict(self):
        return {
            'num runs': 0,
            f'val {self.metric_name} mean': None,
            f'val {self.metric_name} std': None,
            f'test {self.metric_name} mean': None,
            f'test {self.metric_name} std': None,
            f'val {self.metric_name} values': [],
            f'test {self.metric_name} values': [],
            'best steps': [],
            'successful': []
        }
