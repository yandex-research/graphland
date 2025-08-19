import torch
from torch import nn
from modules import (ResidualModuleWrapper, FeedForwardModule, GCNModule, GraphSAGEModule, GATModule, GATSepModule,
                     TransformerAttentionModule, TransformerAttentionSepModule)
from plr_embeddings import PLREmbeddings


class Model(nn.Module):
    modules = {
        'ResMLP': [FeedForwardModule],
        'GCN': [GCNModule],
        'GraphSAGE': [GraphSAGEModule],
        'GAT': [GATModule],
        'GAT-sep': [GATSepModule],
        'GT': [TransformerAttentionModule, FeedForwardModule],
        'GT-sep': [TransformerAttentionSepModule, FeedForwardModule]
    }

    normalization = {
        'none': nn.Identity,
        'layernorm': nn.LayerNorm,
        'batchnorm': nn.BatchNorm1d
    }

    def __init__(self, model_name, num_layers, features_dim, hidden_dim, output_dim, num_heads, hidden_dim_multiplier,
                 normalization, dropout, use_plr, numerical_features_mask, plr_frequencies_dim, plr_frequencies_scale,
                 plr_embedding_dim, use_plr_lite):
        super().__init__()

        normalization = self.normalization[normalization]

        self.use_plr = use_plr
        if use_plr:
            if numerical_features_mask is None:
                raise ValueError('If PLR embeddings for numerical features are used, num_features_mask should be '
                                 'a torch tensor with dtype bool, not None.')

            numerical_features_dim = numerical_features_mask.sum()
            input_dim = features_dim - numerical_features_dim + numerical_features_dim * plr_embedding_dim
            self.register_buffer('numerical_features_mask', numerical_features_mask)
            self.plr_embeddings = PLREmbeddings(features_dim=numerical_features_dim,
                                                frequencies_dim=plr_frequencies_dim,
                                                frequencies_scale=plr_frequencies_scale,
                                                embedding_dim=plr_embedding_dim,
                                                lite=use_plr_lite)
        else:
            input_dim = features_dim

        self.input_module = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.Dropout(p=dropout),
            nn.GELU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        )

        self.residual_modules = nn.ModuleList()
        for _ in range(num_layers):
            for module in self.modules[model_name]:
                residual_module = ResidualModuleWrapper(module=module,
                                                        normalization=normalization,
                                                        dim=hidden_dim,
                                                        hidden_dim_multiplier=hidden_dim_multiplier,
                                                        num_heads=num_heads,
                                                        dropout=dropout)

                self.residual_modules.append(residual_module)

        self.output_module = nn.Sequential(
            normalization(hidden_dim),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.GELU(),
            nn.Linear(in_features=hidden_dim, out_features=output_dim)
        )

    def forward(self, graph, x):
        if self.use_plr:
            x_numerical = x[:, self.numerical_features_mask]
            x_numerical_embedded = self.plr_embeddings(x_numerical).flatten(start_dim=1)
            x = torch.cat([x_numerical_embedded, x[:, ~self.numerical_features_mask]], axis=1)

        x = self.input_module(x)

        for residual_module in self.residual_modules:
            x = residual_module(graph, x)

        x = self.output_module(x).squeeze(1)

        return x


def get_model(args, dataset):
    model = Model(model_name=args.model,
                  num_layers=args.num_layers,
                  features_dim=dataset.features_dim,
                  hidden_dim=args.hidden_dim,
                  output_dim=dataset.targets_dim,
                  num_heads=args.num_heads,
                  hidden_dim_multiplier=args.hidden_dim_multiplier,
                  normalization=args.normalization,
                  dropout=args.dropout,
                  use_plr=args.plr,
                  numerical_features_mask=dataset.numerical_features_mask,
                  plr_frequencies_dim=args.plr_frequencies_dim,
                  plr_frequencies_scale=args.plr_frequencies_scale,
                  plr_embedding_dim=args.plr_embedding_dim,
                  use_plr_lite=args.plr_lite)

    model.to(args.device)

    if args.compile:
        model.compile(dynamic=False, mode='default')

    return model
