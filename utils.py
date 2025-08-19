import yaml
import torch


def read_yaml(path):
    with open(path, 'r') as file:
        data = yaml.load(file, Loader=yaml.UnsafeLoader)

    return data


class YamlCustomDumper(yaml.Dumper):
    """
    Like yaml.Dumper, but with slightly improved formatting.

    Compared to yaml.Dumper, this class:

        Adds blank lines between elements of the top-level data structure if block style is used - see
        write_line_break method, which is taken from https://github.com/yaml/pyyaml/issues/127#issuecomment-525800484.

        Adds additional indent for list items if block style is used - see increase_indent method, which is taken from
        https://reorx.com/blog/python-yaml-tips/.
    """

    def write_line_break(self, data=None):
        super().write_line_break(data=data)

        if len(self.indents) == 1:
            super().write_line_break()

    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow=flow, indentless=False)


class YamlBlockStyleDict(dict):
    @staticmethod
    def represent_block_style_dict(dumper, data):
        return dumper.represent_mapping(tag='tag:yaml.org,2002:map', mapping=data, flow_style=False)


yaml.add_representer(data_type=YamlBlockStyleDict, representer=YamlBlockStyleDict.represent_block_style_dict)


def write_yaml(data, path):
    # If the top-level item is a dict, ensure that block style is used for it.
    if isinstance(data, dict):
        data = YamlBlockStyleDict(data)

    with open(path, 'w') as file:
        yaml.dump(data, file, sort_keys=False, default_flow_style=None, width=float('inf'), Dumper=YamlCustomDumper)


def get_parameter_groups(model):
    no_weight_decay_names = ['bias', 'normalization', 'frequencies']

    parameter_groups = [
        {
            'params': [param for name, param in model.named_parameters()
                       if not any(no_weight_decay_name in name for no_weight_decay_name in no_weight_decay_names)]
        },
        {
            'params': [param for name, param in model.named_parameters()
                       if any(no_weight_decay_name in name for no_weight_decay_name in no_weight_decay_names)],
            'weight_decay': 0
        },
    ]

    return parameter_groups


def get_lr_scheduler_with_warmup(optimizer, num_warmup_steps=None, num_steps=None, warmup_proportion=None,
                                 last_step=-1):

    if num_warmup_steps is None and (num_steps is None or warmup_proportion is None):
        raise ValueError('Either num_warmup_steps or num_steps and warmup_proportion should be provided.')

    if num_warmup_steps is None:
        num_warmup_steps = int(num_steps * warmup_proportion)

    def get_lr_multiplier(step):
        if step < num_warmup_steps:
            return (step + 1) / (num_warmup_steps + 1)
        else:
            return 1

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_multiplier, last_epoch=last_step)

    return lr_scheduler


def update_hparams(args, hparams):
    for key, value in hparams.items():
        setattr(args, key, value)

    return args
